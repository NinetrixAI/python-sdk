"""Tests for AgentContext, AuthResolver, and ToolSource lifecycle methods."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

from ninetrix.tools.agent_context import AgentContext
from ninetrix.tools.auth_resolver import AuthResolver
from ninetrix.runtime.dispatcher import (
    ToolSource,
    ToolDispatcher,
    LocalToolSource,
    MCPToolSource,
    ComposioToolSource,
    RegistryToolSource,
)
from ninetrix.registry import ToolDef, _registry


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_registry():
    _registry.clear()
    yield
    _registry.clear()


def _fake_env(key: str, default: str = "") -> str:
    """Fake env resolver for tests."""
    store = {
        "MY_TOKEN": "sk-test-123",
        "MY_USER": "admin",
        "MY_PASS": "secret",
        "API_KEY": "key-456",
    }
    return store.get(key, default)


# ── AuthResolver ──────────────────────────────────────────────────────────


class TestAuthResolver:
    def test_none_auth_returns_empty(self):
        r = AuthResolver()
        assert r.resolve_headers(None) == {}
        assert r.resolve_headers({}) == {}

    def test_bearer_with_literal(self):
        r = AuthResolver()
        h = r.resolve_headers({"type": "bearer", "token": "sk-literal"})
        assert h == {"Authorization": "Bearer sk-literal"}

    def test_bearer_with_env_var(self):
        r = AuthResolver(env=_fake_env)
        h = r.resolve_headers({"type": "bearer", "token": "${MY_TOKEN}"})
        assert h == {"Authorization": "Bearer sk-test-123"}

    def test_bearer_empty_token_returns_empty(self):
        r = AuthResolver(env=lambda k, d="": "")
        h = r.resolve_headers({"type": "bearer", "token": "${MISSING}"})
        assert h == {}

    def test_basic_auth(self):
        r = AuthResolver()
        h = r.resolve_headers({"type": "basic", "username": "user", "password": "pass"})
        expected = base64.b64encode(b"user:pass").decode()
        assert h == {"Authorization": f"Basic {expected}"}

    def test_basic_auth_with_env_vars(self):
        r = AuthResolver(env=_fake_env)
        h = r.resolve_headers({"type": "basic", "username": "${MY_USER}", "password": "${MY_PASS}"})
        expected = base64.b64encode(b"admin:secret").decode()
        assert h == {"Authorization": f"Basic {expected}"}

    def test_header_auth(self):
        r = AuthResolver()
        h = r.resolve_headers({"type": "header", "header_name": "X-API-Key", "token": "my-key"})
        assert h == {"X-API-Key": "my-key"}

    def test_header_auth_missing_name_returns_empty(self):
        r = AuthResolver()
        h = r.resolve_headers({"type": "header", "token": "my-key"})
        assert h == {}

    def test_api_key_query_returns_no_headers(self):
        r = AuthResolver()
        h = r.resolve_headers({"type": "api_key_query", "query_param": "key", "token": "val"})
        assert h == {}

    def test_api_key_query_params(self):
        r = AuthResolver(env=_fake_env)
        p = r.resolve_query_params({"type": "api_key_query", "query_param": "api_key", "token": "${API_KEY}"})
        assert p == {"api_key": "key-456"}

    def test_query_params_non_query_type_returns_empty(self):
        r = AuthResolver()
        p = r.resolve_query_params({"type": "bearer", "token": "x"})
        assert p == {}

    def test_unknown_type_returns_empty(self):
        r = AuthResolver()
        h = r.resolve_headers({"type": "oauth2_fancy"})
        assert h == {}

    def test_default_type_is_bearer(self):
        r = AuthResolver()
        h = r.resolve_headers({"token": "sk-123"})
        assert h == {"Authorization": "Bearer sk-123"}


# ── AgentContext ──────────────────────────────────────────────────────────


class TestAgentContext:
    def test_construction(self):
        http = MagicMock()
        auth = AuthResolver()
        ctx = AgentContext(
            http=http,
            auth=auth,
            agent_name="test-agent",
            org_id="org-1",
        )
        assert ctx.http is http
        assert ctx.auth is auth
        assert ctx.agent_name == "test-agent"
        assert ctx.org_id == "org-1"

    def test_default_env(self):
        import os
        ctx = AgentContext(http=MagicMock())
        # Default env callable should read from os.environ
        os.environ["_NINETRIX_TEST_KEY"] = "test_val"
        try:
            assert ctx.env("_NINETRIX_TEST_KEY") == "test_val"
        finally:
            del os.environ["_NINETRIX_TEST_KEY"]

    def test_custom_env(self):
        ctx = AgentContext(http=MagicMock(), env=_fake_env)
        assert ctx.env("MY_TOKEN") == "sk-test-123"


# ── ToolSource lifecycle methods ──────────────────────────────────────────


class TestToolSourceLifecycle:
    def test_source_type_defaults(self):
        assert ToolSource.source_type == "unknown"
        assert LocalToolSource.source_type == "local"
        assert MCPToolSource.source_type == "mcp"
        assert ComposioToolSource.source_type == "composio"
        assert RegistryToolSource.source_type == "registry"

    @pytest.mark.asyncio
    async def test_default_initialize_is_noop(self):
        """Concrete subclass with no initialize override — should not raise."""
        class DummySource(ToolSource):
            def tool_definitions(self): return []
            async def call(self, n, a): return ""
            def handles(self, n): return False

        s = DummySource()
        await s.initialize()  # should not raise

    def test_default_validate_config_is_noop(self):
        class DummySource(ToolSource):
            def tool_definitions(self): return []
            async def call(self, n, a): return ""
            def handles(self, n): return False

        s = DummySource()
        s.validate_config()  # should not raise

    @pytest.mark.asyncio
    async def test_default_health_check_returns_true(self):
        class DummySource(ToolSource):
            def tool_definitions(self): return []
            async def call(self, n, a): return ""
            def handles(self, n): return False

        s = DummySource()
        assert await s.health_check() is True

    @pytest.mark.asyncio
    async def test_default_shutdown_is_noop(self):
        class DummySource(ToolSource):
            def tool_definitions(self): return []
            async def call(self, n, a): return ""
            def handles(self, n): return False

        s = DummySource()
        await s.shutdown()  # should not raise


# ── ToolDispatcher lifecycle ──────────────────────────────────────────────


def _make_tool_def(name: str) -> ToolDef:
    return ToolDef(
        name=name,
        description="test",
        parameters={"type": "object", "properties": {}, "required": []},
        fn=lambda: None,
    )


class TestToolDispatcherLifecycle:
    @pytest.mark.asyncio
    async def test_initialize_calls_validate_then_init(self):
        """validate_config() is called on all sources before any initialize()."""
        call_order = []

        class TrackedSource(ToolSource):
            source_type = "tracked"
            def __init__(self, name):
                self._name = name
            def tool_definitions(self): return []
            async def call(self, n, a): return ""
            def handles(self, n): return False
            def validate_config(self):
                call_order.append(f"validate:{self._name}")
            async def initialize(self):
                call_order.append(f"init:{self._name}")

        d = ToolDispatcher([TrackedSource("a"), TrackedSource("b")])
        await d.initialize()

        assert call_order == [
            "validate:a", "validate:b",
            "init:a", "init:b",
        ]

    @pytest.mark.asyncio
    async def test_shutdown_calls_in_reverse_order(self):
        call_order = []

        class TrackedSource(ToolSource):
            def __init__(self, name):
                self._name = name
            def tool_definitions(self): return []
            async def call(self, n, a): return ""
            def handles(self, n): return False
            async def shutdown(self):
                call_order.append(self._name)

        d = ToolDispatcher([TrackedSource("a"), TrackedSource("b"), TrackedSource("c")])
        await d.shutdown()

        assert call_order == ["c", "b", "a"]

    @pytest.mark.asyncio
    async def test_shutdown_continues_on_error(self):
        """If one source's shutdown() raises, others still get called."""
        call_order = []

        class FailSource(ToolSource):
            def tool_definitions(self): return []
            async def call(self, n, a): return ""
            def handles(self, n): return False
            async def shutdown(self):
                call_order.append("fail")
                raise RuntimeError("boom")

        class OkSource(ToolSource):
            def tool_definitions(self): return []
            async def call(self, n, a): return ""
            def handles(self, n): return False
            async def shutdown(self):
                call_order.append("ok")

        d = ToolDispatcher([OkSource(), FailSource()])
        await d.shutdown()  # should not raise

        # Both called (reverse order: FailSource first, then OkSource)
        assert call_order == ["fail", "ok"]

    @pytest.mark.asyncio
    async def test_health_check_aggregates(self):
        class HealthySource(ToolSource):
            source_type = "healthy"
            def tool_definitions(self): return []
            async def call(self, n, a): return ""
            def handles(self, n): return False
            async def health_check(self): return True

        class UnhealthySource(ToolSource):
            source_type = "unhealthy"
            def tool_definitions(self): return []
            async def call(self, n, a): return ""
            def handles(self, n): return False
            async def health_check(self): return False

        d = ToolDispatcher([HealthySource(), UnhealthySource()])
        result = await d.health_check()

        assert result["HealthySource:healthy"] is True
        assert result["UnhealthySource:unhealthy"] is False

    @pytest.mark.asyncio
    async def test_health_check_catches_exceptions(self):
        class CrashSource(ToolSource):
            source_type = "crash"
            def tool_definitions(self): return []
            async def call(self, n, a): return ""
            def handles(self, n): return False
            async def health_check(self): raise RuntimeError("boom")

        d = ToolDispatcher([CrashSource()])
        result = await d.health_check()

        assert result["CrashSource:crash"] is False


# ── MCPToolSource with ctx ────────────────────────────────────────────────


class TestMCPToolSourceCtx:
    def test_ctx_http_used_when_provided(self):
        mock_http = MagicMock()
        ctx = AgentContext(http=mock_http)
        source = MCPToolSource("http://gw:8080", "tok", "org", ctx=ctx)
        assert source._http() is mock_http

    def test_fallback_to_singleton_when_no_ctx(self):
        source = MCPToolSource("http://gw:8080", "tok", "org")
        # _http() should call get_http_client() — we just verify it doesn't crash
        # (it returns the singleton httpx client)
        client = source._http()
        assert client is not None


class TestRegistryToolSourceCtx:
    def test_ctx_http_used_when_provided(self):
        mock_http = MagicMock()
        ctx = AgentContext(http=mock_http)
        source = RegistryToolSource(["s1"], "http://reg", "key", ctx=ctx)
        assert source._http() is mock_http
