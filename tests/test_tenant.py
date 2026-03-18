"""
tests/test_tenant.py
====================

Unit tests for ninetrix._internals.tenant.

Tests cover:
- TenantContext is a frozen dataclass with correct defaults
- set_tenant / get_tenant round-trip
- require_tenant raises ConfigurationError when not set
- tenant_scope async context manager sets and restores tenant
- Nested tenant_scope restores outer tenant correctly
- _auto_init_from_env sets tenant from env vars
- _auto_init_from_env is a no-op when env vars are missing
- Context is task-local (separate asyncio tasks get separate values)
- Top-level re-exports work
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

import pytest

from ninetrix._internals.tenant import (
    TenantContext,
    _auto_init_from_env,
    _current_tenant,
    get_tenant,
    require_tenant,
    set_tenant,
    tenant_scope,
)
from ninetrix._internals.types import ConfigurationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_tenant() -> None:
    """Reset the context var to None for test isolation."""
    _current_tenant.set(None)


# ===========================================================================
# TenantContext dataclass
# ===========================================================================

class TestTenantContext:
    def test_required_field(self):
        tc = TenantContext(workspace_id="ws-1")
        assert tc.workspace_id == "ws-1"

    def test_defaults(self):
        tc = TenantContext(workspace_id="ws-1")
        assert tc.org_id == ""
        assert tc.api_key == ""
        assert tc.region == "us"
        assert tc.db_schema == "public"

    def test_all_fields(self):
        tc = TenantContext(
            workspace_id="ws-2",
            org_id="org-x",
            api_key="nxt_abc",
            region="eu",
            db_schema="tenant_ws2",
        )
        assert tc.org_id == "org-x"
        assert tc.api_key == "nxt_abc"
        assert tc.region == "eu"
        assert tc.db_schema == "tenant_ws2"

    def test_frozen(self):
        tc = TenantContext(workspace_id="ws-3")
        with pytest.raises((AttributeError, TypeError)):
            tc.workspace_id = "mutated"  # type: ignore[misc]

    def test_equality(self):
        a = TenantContext(workspace_id="ws-1", api_key="k1")
        b = TenantContext(workspace_id="ws-1", api_key="k1")
        assert a == b

    def test_inequality(self):
        a = TenantContext(workspace_id="ws-1")
        b = TenantContext(workspace_id="ws-2")
        assert a != b


# ===========================================================================
# set_tenant / get_tenant
# ===========================================================================

class TestSetGetTenant:
    def setup_method(self):
        _clear_tenant()

    def teardown_method(self):
        _clear_tenant()

    def test_get_tenant_returns_none_when_not_set(self):
        assert get_tenant() is None

    def test_set_get_round_trip(self):
        tc = TenantContext(workspace_id="ws-set")
        set_tenant(tc)
        assert get_tenant() is tc

    def test_set_returns_token(self):
        tc = TenantContext(workspace_id="ws-token")
        token = set_tenant(tc)
        assert token is not None

    def test_token_can_restore_previous_value(self):
        tc1 = TenantContext(workspace_id="ws-1")
        tc2 = TenantContext(workspace_id="ws-2")
        set_tenant(tc1)
        token = set_tenant(tc2)
        assert get_tenant() is tc2
        _current_tenant.reset(token)
        assert get_tenant() is tc1

    def test_set_tenant_overwrites(self):
        set_tenant(TenantContext(workspace_id="ws-a"))
        set_tenant(TenantContext(workspace_id="ws-b"))
        assert get_tenant().workspace_id == "ws-b"  # type: ignore[union-attr]


# ===========================================================================
# require_tenant
# ===========================================================================

class TestRequireTenant:
    def setup_method(self):
        _clear_tenant()

    def teardown_method(self):
        _clear_tenant()

    def test_returns_tenant_when_set(self):
        tc = TenantContext(workspace_id="ws-req")
        set_tenant(tc)
        assert require_tenant() is tc

    def test_raises_configuration_error_when_not_set(self):
        with pytest.raises(ConfigurationError) as exc_info:
            require_tenant()
        msg = str(exc_info.value)
        assert "No TenantContext" in msg
        assert "tenant_scope" in msg
        assert "set_tenant" in msg
        assert "AgentSandbox" in msg

    def test_error_message_contains_fix_instructions(self):
        with pytest.raises(ConfigurationError) as exc_info:
            require_tenant()
        msg = str(exc_info.value)
        # All three fix paths must be described
        assert "FastAPI" in msg or "handler" in msg
        assert "script" in msg or "set_tenant" in msg
        assert "test" in msg or "AgentSandbox" in msg


# ===========================================================================
# tenant_scope async context manager
# ===========================================================================

class TestTenantScope:
    def setup_method(self):
        _clear_tenant()

    def teardown_method(self):
        _clear_tenant()

    async def test_sets_tenant_inside_block(self):
        tc = TenantContext(workspace_id="ws-scope")
        async with tenant_scope(tc):
            assert get_tenant() is tc

    async def test_restores_none_after_block(self):
        tc = TenantContext(workspace_id="ws-scope")
        async with tenant_scope(tc):
            pass
        assert get_tenant() is None

    async def test_restores_previous_tenant_after_nested_block(self):
        outer = TenantContext(workspace_id="ws-outer")
        inner = TenantContext(workspace_id="ws-inner")
        async with tenant_scope(outer):
            async with tenant_scope(inner):
                assert get_tenant() is inner
            assert get_tenant() is outer
        assert get_tenant() is None

    async def test_restores_tenant_on_exception(self):
        tc = TenantContext(workspace_id="ws-exc")
        with pytest.raises(ValueError):
            async with tenant_scope(tc):
                raise ValueError("boom")
        assert get_tenant() is None

    async def test_yields_context(self):
        tc = TenantContext(workspace_id="ws-yield")
        async with tenant_scope(tc) as yielded:
            assert yielded is tc

    async def test_three_levels_of_nesting(self):
        t1 = TenantContext(workspace_id="ws-1")
        t2 = TenantContext(workspace_id="ws-2")
        t3 = TenantContext(workspace_id="ws-3")
        async with tenant_scope(t1):
            assert get_tenant() is t1
            async with tenant_scope(t2):
                assert get_tenant() is t2
                async with tenant_scope(t3):
                    assert get_tenant() is t3
                assert get_tenant() is t2
            assert get_tenant() is t1

    async def test_require_tenant_works_inside_scope(self):
        tc = TenantContext(workspace_id="ws-req-scope")
        async with tenant_scope(tc):
            assert require_tenant() is tc


# ===========================================================================
# Task isolation
# ===========================================================================

class TestTaskIsolation:
    def setup_method(self):
        _clear_tenant()

    def teardown_method(self):
        _clear_tenant()

    async def test_separate_tasks_have_independent_tenants(self):
        """Each asyncio task gets its own context copy."""
        results: list[TenantContext | None] = []

        async def task_a():
            async with tenant_scope(TenantContext(workspace_id="ws-a")):
                await asyncio.sleep(0)  # yield control
                results.append(get_tenant())

        async def task_b():
            # No tenant set in this task
            await asyncio.sleep(0)
            results.append(get_tenant())

        await asyncio.gather(task_a(), task_b())
        # One result should have workspace_id "ws-a", the other None
        ids = {r.workspace_id if r else None for r in results}
        assert "ws-a" in ids
        assert None in ids

    async def test_child_task_inherits_parent_tenant(self):
        """
        asyncio tasks copy the parent context at creation time.
        The child sees the parent's tenant but cannot mutate the parent's
        context var.
        """
        tc = TenantContext(workspace_id="ws-parent")
        token = set_tenant(tc)
        try:
            child_tenant: list[TenantContext | None] = []

            async def child():
                child_tenant.append(get_tenant())

            await asyncio.create_task(child())
            assert child_tenant[0] is tc
        finally:
            _current_tenant.reset(token)


# ===========================================================================
# _auto_init_from_env
# ===========================================================================

class TestAutoInitFromEnv:
    def setup_method(self):
        _clear_tenant()

    def teardown_method(self):
        _clear_tenant()

    def test_sets_tenant_when_both_env_vars_present(self):
        with patch.dict(os.environ, {
            "NINETRIX_WORKSPACE_ID": "ws-env-123",
            "NINETRIX_API_KEY": "nxt_test_key",
        }):
            _auto_init_from_env()
        tc = get_tenant()
        assert tc is not None
        assert tc.workspace_id == "ws-env-123"
        assert tc.api_key == "nxt_test_key"

    def test_noop_when_workspace_id_missing(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("NINETRIX_WORKSPACE_ID", "NINETRIX_API_KEY")}
        env["NINETRIX_API_KEY"] = "nxt_key"
        with patch.dict(os.environ, env, clear=True):
            _auto_init_from_env()
        assert get_tenant() is None

    def test_noop_when_api_key_missing(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("NINETRIX_WORKSPACE_ID", "NINETRIX_API_KEY")}
        env["NINETRIX_WORKSPACE_ID"] = "ws-123"
        with patch.dict(os.environ, env, clear=True):
            _auto_init_from_env()
        assert get_tenant() is None

    def test_noop_when_both_missing(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("NINETRIX_WORKSPACE_ID", "NINETRIX_API_KEY")}
        with patch.dict(os.environ, env, clear=True):
            _auto_init_from_env()
        assert get_tenant() is None


# ===========================================================================
# Re-exports
# ===========================================================================

class TestReExports:
    def test_top_level_exports(self):
        import ninetrix
        assert hasattr(ninetrix, "TenantContext")
        assert hasattr(ninetrix, "set_tenant")
        assert hasattr(ninetrix, "get_tenant")
        assert hasattr(ninetrix, "require_tenant")
        assert hasattr(ninetrix, "tenant_scope")

    def test_tenant_context_is_same_class(self):
        from ninetrix import TenantContext as TC
        assert TC is TenantContext

    def test_tenant_scope_is_same_function(self):
        from ninetrix import tenant_scope as ts
        assert ts is tenant_scope
