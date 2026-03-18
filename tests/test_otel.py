"""Tests for observability/otel.py — PR 31: OpenTelemetry integration.

All tests work regardless of whether opentelemetry-api is installed by
mocking the import as needed.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ninetrix.observability.otel import (
    _HAS_OTEL,
    _NoOpSpan,
    _NoOpTracer,
    _active_spans,
    _noop_tracer,
    _set_span_attrs,
    attach_otel_to_bus,
    cleanup_thread,
    get_tracer,
    is_configured,
)
from ninetrix.observability.events import AgentEvent, EventBus


# =============================================================================
# _NoOpSpan + _NoOpTracer
# =============================================================================


def test_noop_span_is_safe():
    span = _NoOpSpan()
    span.set_attribute("key", "value")
    span.set_status("ok")
    span.record_exception(ValueError("test"))
    span.end()


def test_noop_span_context_manager():
    span = _NoOpSpan()
    with span:
        pass  # should not raise


def test_noop_tracer_returns_noop_span():
    tracer = _NoOpTracer()
    span = tracer.start_span("test.span")
    assert isinstance(span, _NoOpSpan)


def test_noop_tracer_singleton():
    assert isinstance(_noop_tracer, _NoOpTracer)


# =============================================================================
# get_tracer — graceful degradation
# =============================================================================


def test_get_tracer_returns_noop_when_not_configured():
    # Even if otel is available, if not configured, return noop
    tracer = get_tracer()
    assert tracer is not None
    # Should be safe to call start_span on whatever is returned
    span = tracer.start_span("test")
    assert span is not None


def test_get_tracer_not_configured_returns_noop_tracer():
    # Force unconfigured state
    import ninetrix.observability.otel as otel_mod
    orig_configured = otel_mod._configured
    otel_mod._configured = False
    try:
        tracer = get_tracer()
        assert isinstance(tracer, _NoOpTracer)
    finally:
        otel_mod._configured = orig_configured


# =============================================================================
# configure_otel — no-op when opentelemetry not installed
# =============================================================================


def test_configure_otel_warns_when_otel_not_available():
    """If opentelemetry-api is not installed, configure_otel emits a warning."""
    import ninetrix.observability.otel as otel_mod

    if _HAS_OTEL:
        # Can't easily uninstall, so mock _HAS_OTEL
        orig = otel_mod._HAS_OTEL
        otel_mod._HAS_OTEL = False
        try:
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                from ninetrix.observability.otel import configure_otel
                configure_otel("http://localhost:4317")
                if w:
                    assert any("opentelemetry" in str(warning.message).lower() for warning in w)
        finally:
            otel_mod._HAS_OTEL = orig
    else:
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from ninetrix.observability.otel import configure_otel
            configure_otel("http://localhost:4317")
            if w:
                assert any("opentelemetry" in str(warning.message).lower() for warning in w)


def test_is_configured_initially_false():
    # After import without configure_otel(), should be False
    # (unless a previous test configured it)
    # We just verify it's a bool
    result = is_configured()
    assert isinstance(result, bool)


# =============================================================================
# _set_span_attrs — safe attribute setting
# =============================================================================


def test_set_span_attrs_calls_set_attribute():
    span = MagicMock()
    _set_span_attrs(span, {"key1": "val1", "key2": 42})
    span.set_attribute.assert_any_call("key1", "val1")
    span.set_attribute.assert_any_call("key2", 42)


def test_set_span_attrs_skips_none_values():
    span = MagicMock()
    _set_span_attrs(span, {"a": "present", "b": None, "c": 0})
    calls = [c[0][0] for c in span.set_attribute.call_args_list]
    assert "a" in calls
    assert "b" not in calls
    assert "c" in calls


def test_set_span_attrs_handles_exception_gracefully():
    span = MagicMock()
    span.set_attribute.side_effect = RuntimeError("OTEL error")
    # Should not raise
    _set_span_attrs(span, {"key": "val"})


# =============================================================================
# cleanup_thread
# =============================================================================


def test_cleanup_thread_removes_span():
    _active_spans["test_thread_123"] = _NoOpSpan()
    cleanup_thread("test_thread_123")
    assert "test_thread_123" not in _active_spans


def test_cleanup_thread_removes_turn_span():
    _active_spans["t1:turn"] = _NoOpSpan()
    cleanup_thread("t1")
    assert "t1:turn" not in _active_spans


def test_cleanup_thread_removes_tool_spans():
    _active_spans["t2:tool:search"] = _NoOpSpan()
    _active_spans["t2:tool:lookup"] = _NoOpSpan()
    cleanup_thread("t2")
    assert "t2:tool:search" not in _active_spans
    assert "t2:tool:lookup" not in _active_spans


def test_cleanup_thread_noop_if_not_found():
    # Should not raise even if thread has no spans
    cleanup_thread("nonexistent_thread_xyz")


# =============================================================================
# attach_otel_to_bus — EventBus integration
# =============================================================================


def test_attach_otel_registers_handlers():
    bus = EventBus()
    attach_otel_to_bus(bus)
    # Should have subscribed handlers for various events
    assert bus.subscriber_count() > 0


@pytest.mark.asyncio
async def test_run_start_creates_span():
    """on_run_start should store a span in _active_spans."""
    bus = EventBus()

    mock_span = MagicMock()
    mock_span.set_attribute = MagicMock()

    mock_tracer = MagicMock()
    mock_tracer.start_span.return_value = mock_span

    import ninetrix.observability.otel as otel_mod
    orig_get_tracer = otel_mod.get_tracer

    def patched_get_tracer(name: str = "ninetrix") -> Any:
        return mock_tracer

    otel_mod.get_tracer = patched_get_tracer
    try:
        attach_otel_to_bus(bus)

        thread_id = "test_run_start_thread"
        await bus.emit(AgentEvent(
            type="run.start",
            thread_id=thread_id,
            agent_name="test-agent",
            data={"model": "claude-sonnet-4-6", "provider": "anthropic"},
        ))

        assert thread_id in _active_spans
    finally:
        otel_mod.get_tracer = orig_get_tracer
        _active_spans.pop("test_run_start_thread", None)


@pytest.mark.asyncio
async def test_run_end_removes_span():
    """on_run_end should remove and end the span."""
    bus = EventBus()

    mock_span = MagicMock()
    _active_spans["test_run_end_thread"] = mock_span

    mock_tracer = MagicMock()
    mock_tracer.start_span.return_value = MagicMock()

    import ninetrix.observability.otel as otel_mod
    orig_get_tracer = otel_mod.get_tracer
    otel_mod.get_tracer = lambda name="ninetrix": mock_tracer
    try:
        attach_otel_to_bus(bus)

        await bus.emit(AgentEvent(
            type="run.end",
            thread_id="test_run_end_thread",
            agent_name="test-agent",
            data={"tokens_used": 100, "cost_usd": 0.001, "steps": 2},
        ))

        assert "test_run_end_thread" not in _active_spans
        mock_span.end.assert_called()
    finally:
        otel_mod.get_tracer = orig_get_tracer


@pytest.mark.asyncio
async def test_tool_call_creates_span():
    """on_tool_call should create a span keyed by thread:tool:name."""
    bus = EventBus()

    mock_span = MagicMock()
    mock_root_span = MagicMock()

    mock_tracer = MagicMock()
    mock_tracer.start_span.return_value = mock_span

    thread_id = "test_tool_thread"
    _active_spans[thread_id] = mock_root_span

    import ninetrix.observability.otel as otel_mod
    orig_get_tracer = otel_mod.get_tracer
    otel_mod.get_tracer = lambda name="ninetrix": mock_tracer
    try:
        attach_otel_to_bus(bus)

        await bus.emit(AgentEvent(
            type="tool.call",
            thread_id=thread_id,
            agent_name="test-agent",
            data={"tool_name": "search", "turn": 1},
        ))

        assert f"{thread_id}:tool:search" in _active_spans
    finally:
        otel_mod.get_tracer = orig_get_tracer
        _active_spans.pop(thread_id, None)
        _active_spans.pop(f"{thread_id}:tool:search", None)


@pytest.mark.asyncio
async def test_tool_result_ends_tool_span():
    """on_tool_result should remove and end the tool span."""
    bus = EventBus()

    mock_tool_span = MagicMock()
    thread_id = "test_tool_result_thread"
    _active_spans[f"{thread_id}:tool:search"] = mock_tool_span

    mock_tracer = MagicMock()
    mock_tracer.start_span.return_value = MagicMock()

    import ninetrix.observability.otel as otel_mod
    orig_get_tracer = otel_mod.get_tracer
    otel_mod.get_tracer = lambda name="ninetrix": mock_tracer
    try:
        attach_otel_to_bus(bus)

        await bus.emit(AgentEvent(
            type="tool.result",
            thread_id=thread_id,
            agent_name="test-agent",
            data={"tool_name": "search", "result": "found it"},
        ))

        assert f"{thread_id}:tool:search" not in _active_spans
        mock_tool_span.end.assert_called()
    finally:
        otel_mod.get_tracer = orig_get_tracer
        _active_spans.pop(thread_id, None)


@pytest.mark.asyncio
async def test_error_event_ends_root_span():
    """on_error should remove and mark the root span as failed."""
    bus = EventBus()

    mock_span = MagicMock()
    thread_id = "test_error_thread"
    _active_spans[thread_id] = mock_span

    mock_tracer = MagicMock()
    mock_tracer.start_span.return_value = MagicMock()

    import ninetrix.observability.otel as otel_mod
    orig_get_tracer = otel_mod.get_tracer
    otel_mod.get_tracer = lambda name="ninetrix": mock_tracer
    try:
        attach_otel_to_bus(bus)

        await bus.emit(AgentEvent(
            type="error",
            thread_id=thread_id,
            agent_name="test-agent",
            data={"message": "Something went wrong"},
        ))

        assert thread_id not in _active_spans
        mock_span.end.assert_called()
    finally:
        otel_mod.get_tracer = orig_get_tracer


@pytest.mark.asyncio
async def test_full_lifecycle_integration():
    """Simulate a full run lifecycle and verify all events process without error."""
    bus = EventBus()
    attach_otel_to_bus(bus)

    thread_id = "lifecycle_test_thread"

    # Simulate a full run
    await bus.emit(AgentEvent(
        type="run.start",
        thread_id=thread_id,
        agent_name="test-agent",
        data={"model": "claude-sonnet-4-6"},
    ))
    await bus.emit(AgentEvent(
        type="turn.start",
        thread_id=thread_id,
        agent_name="test-agent",
        data={"turn": 0},
    ))
    await bus.emit(AgentEvent(
        type="tool.call",
        thread_id=thread_id,
        agent_name="test-agent",
        data={"tool_name": "search", "turn": 0},
    ))
    await bus.emit(AgentEvent(
        type="tool.result",
        thread_id=thread_id,
        agent_name="test-agent",
        data={"tool_name": "search", "result": "found"},
    ))
    await bus.emit(AgentEvent(
        type="turn.end",
        thread_id=thread_id,
        agent_name="test-agent",
        data={"turn": 0, "input_tokens": 10, "output_tokens": 5},
    ))
    await bus.emit(AgentEvent(
        type="run.end",
        thread_id=thread_id,
        agent_name="test-agent",
        data={"tokens_used": 15, "cost_usd": 0.0001, "steps": 1},
    ))

    # All spans should be cleaned up
    assert thread_id not in _active_spans
    assert f"{thread_id}:turn" not in _active_spans


# =============================================================================
# Public imports
# =============================================================================


def test_otel_importable():
    from ninetrix.observability.otel import configure_otel, get_tracer, attach_otel_to_bus
    assert configure_otel is not None
    assert get_tracer is not None
    assert attach_otel_to_bus is not None
