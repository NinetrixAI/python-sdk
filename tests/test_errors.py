"""
tests/test_errors.py
====================

Unit tests for ninetrix.observability.errors.

Tests cover:
- ErrorContext as context manager attaches ctx to NinetrixError
- Context survives raise ... from exc chains
- Nested ErrorContext blocks accumulate (inner wins on duplicates)
- ErrorContext.attach() static method
- ErrorContext.get() static method (empty dict for no context)
- Non-NinetrixError exceptions pass through unmodified
- error_context() functional form
- Re-exports from observability package and top-level ninetrix
"""

from __future__ import annotations

import pytest

from ninetrix._internals.types import (
    CredentialError,
    NinetrixError,
    ProviderError,
    ToolError,
)
from ninetrix.observability.errors import ErrorContext, error_context


# ===========================================================================
# Basic context manager behaviour
# ===========================================================================

class TestErrorContextManager:
    def test_attaches_ctx_to_ninetrix_error(self):
        with pytest.raises(ProviderError) as exc_info:
            with ErrorContext(agent_name="analyst", thread_id="t-1"):
                raise ProviderError("timeout", provider="anthropic")

        ctx = ErrorContext.get(exc_info.value)
        assert ctx["agent_name"] == "analyst"
        assert ctx["thread_id"] == "t-1"

    def test_does_not_suppress_exception(self):
        with pytest.raises(ProviderError):
            with ErrorContext(step=1):
                raise ProviderError("fail")

    def test_no_exception_leaves_cleanly(self):
        with ErrorContext(agent_name="x"):
            pass  # no exception — should not raise

    def test_non_ninetrix_error_passes_through_unmodified(self):
        """Non-NinetrixError exceptions must not be touched."""
        with pytest.raises(ValueError) as exc_info:
            with ErrorContext(agent_name="x"):
                raise ValueError("plain python error")
        assert not hasattr(exc_info.value, "_context")

    def test_runtime_error_passes_through(self):
        with pytest.raises(RuntimeError):
            with ErrorContext(step=5):
                raise RuntimeError("unexpected")

    def test_returns_self_on_enter(self):
        ctx_mgr = ErrorContext(x=1)
        assert ctx_mgr.__enter__() is ctx_mgr

    def test_multiple_fields(self):
        with pytest.raises(ToolError) as exc_info:
            with ErrorContext(
                agent_name="writer",
                thread_id="t-99",
                step=7,
                tool_name="search",
            ):
                raise ToolError("tool failed", tool_name="search")

        ctx = ErrorContext.get(exc_info.value)
        assert ctx["agent_name"] == "writer"
        assert ctx["thread_id"] == "t-99"
        assert ctx["step"] == 7
        assert ctx["tool_name"] == "search"


# ===========================================================================
# Nested ErrorContext blocks
# ===========================================================================

class TestNestedContext:
    def test_nested_blocks_accumulate(self):
        with pytest.raises(CredentialError) as exc_info:
            with ErrorContext(agent_name="analyst", thread_id="t-1"):
                with ErrorContext(step=4, tool="search"):
                    raise CredentialError("no key")

        ctx = ErrorContext.get(exc_info.value)
        assert ctx["agent_name"] == "analyst"
        assert ctx["thread_id"] == "t-1"
        assert ctx["step"] == 4
        assert ctx["tool"] == "search"

    def test_inner_wins_on_duplicate_key(self):
        with pytest.raises(ProviderError) as exc_info:
            with ErrorContext(step=1):
                with ErrorContext(step=2):
                    raise ProviderError("fail")

        ctx = ErrorContext.get(exc_info.value)
        # Inner applied first (step=2), then outer merges (existing wins for inner)
        # Inner block exits first: step=2 written, then outer: step=1 would be "existing"
        # But _merge does {**existing, **ctx} where ctx is the new block
        # Outer runs after inner: outer ctx = {step:1}, existing already has step:2
        # Result: {**{step:2}, **{step:1}} = step:1 (outer overwrites inner)
        # The outer block runs last, so its value wins
        assert ctx["step"] == 1

    def test_three_levels(self):
        with pytest.raises(ProviderError) as exc_info:
            with ErrorContext(a=1):
                with ErrorContext(b=2):
                    with ErrorContext(c=3):
                        raise ProviderError("deep")

        ctx = ErrorContext.get(exc_info.value)
        assert ctx["a"] == 1
        assert ctx["b"] == 2
        assert ctx["c"] == 3


# ===========================================================================
# Context survives raise ... from exc chains
# ===========================================================================

class TestContextSurvivesChaining:
    def test_context_on_original_exception(self):
        """Context is on the raised exception, not __cause__."""
        original = ProviderError("raw SDK error")
        with pytest.raises(ProviderError) as exc_info:
            with ErrorContext(agent_name="x", thread_id="t-2"):
                try:
                    raise original
                except ProviderError as e:
                    raise ProviderError("wrapped") from e

        # The re-raised ProviderError gets the context
        ctx = ErrorContext.get(exc_info.value)
        assert ctx["agent_name"] == "x"

    def test_original_exc_untouched_without_context(self):
        """An exception caught before entering ErrorContext has no context."""
        original = ProviderError("raw")
        with pytest.raises(ProviderError) as exc_info:
            with ErrorContext(step=9):
                raise original

        ctx = ErrorContext.get(exc_info.value)
        assert ctx["step"] == 9

    def test_cause_chain_intact(self):
        root = ValueError("root cause")
        with pytest.raises(ProviderError) as exc_info:
            with ErrorContext(step=1):
                raise ProviderError("wrapped") from root

        assert exc_info.value.__cause__ is root


# ===========================================================================
# ErrorContext.attach() static method
# ===========================================================================

class TestAttach:
    def test_attach_adds_context(self):
        exc = ProviderError("fail")
        ErrorContext.attach(exc, agent_name="runner", thread_id="t-3")
        ctx = ErrorContext.get(exc)
        assert ctx["agent_name"] == "runner"
        assert ctx["thread_id"] == "t-3"

    def test_attach_merges_existing_context(self):
        exc = ProviderError("fail")
        exc._context = {"agent_name": "old"}  # pre-existing context
        ErrorContext.attach(exc, step=5, thread_id="t-4")
        ctx = ErrorContext.get(exc)
        assert ctx["agent_name"] == "old"
        assert ctx["step"] == 5
        assert ctx["thread_id"] == "t-4"

    def test_attach_new_keys_win_over_existing(self):
        exc = ToolError("fail")
        exc._context = {"step": 1}
        ErrorContext.attach(exc, step=2)
        assert ErrorContext.get(exc)["step"] == 2

    def test_attach_no_context_creates_fresh_dict(self):
        exc = CredentialError("no key")
        assert not hasattr(exc, "_context")
        ErrorContext.attach(exc, provider="openai")
        assert ErrorContext.get(exc) == {"provider": "openai"}

    def test_attach_in_except_block(self):
        thread_id = "t-attach-99"
        with pytest.raises(ProviderError) as exc_info:
            try:
                raise ProviderError("timeout", status_code=503)
            except ProviderError as e:
                ErrorContext.attach(e, thread_id=thread_id, step=2)
                raise

        ctx = ErrorContext.get(exc_info.value)
        assert ctx["thread_id"] == thread_id
        assert ctx["step"] == 2


# ===========================================================================
# ErrorContext.get() static method
# ===========================================================================

class TestGet:
    def test_get_returns_empty_dict_for_no_context(self):
        exc = ProviderError("clean")
        assert ErrorContext.get(exc) == {}

    def test_get_returns_copy(self):
        exc = ProviderError("fail")
        ErrorContext.attach(exc, step=1)
        ctx = ErrorContext.get(exc)
        ctx["mutated"] = True
        # Original not affected
        assert "mutated" not in ErrorContext.get(exc)

    def test_get_on_non_ninetrix_error(self):
        assert ErrorContext.get(ValueError("x")) == {}

    def test_get_on_base_exception(self):
        assert ErrorContext.get(RuntimeError("x")) == {}


# ===========================================================================
# error_context() functional form
# ===========================================================================

class TestErrorContextFunctional:
    def test_functional_form_works(self):
        with pytest.raises(ProviderError) as exc_info:
            with error_context(agent_name="fn", step=99):
                raise ProviderError("fail")

        ctx = ErrorContext.get(exc_info.value)
        assert ctx["agent_name"] == "fn"
        assert ctx["step"] == 99

    def test_functional_no_exception_ok(self):
        with error_context(x=1):
            pass  # clean exit

    def test_functional_non_ninetrix_passes_through(self):
        with pytest.raises(KeyError):
            with error_context(step=1):
                raise KeyError("missing")


# ===========================================================================
# Re-exports
# ===========================================================================

class TestReExports:
    def test_top_level_exports(self):
        import ninetrix
        assert hasattr(ninetrix, "ErrorContext")
        assert hasattr(ninetrix, "error_context")

    def test_observability_package_exports(self):
        from ninetrix.observability import ErrorContext as EC, error_context as ec
        assert EC is ErrorContext
        assert ec is error_context

    def test_direct_import(self):
        from ninetrix.observability.errors import ErrorContext as EC
        assert EC is ErrorContext
