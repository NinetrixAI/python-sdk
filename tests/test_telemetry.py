"""
tests/test_telemetry.py
=======================

Unit tests for ninetrix.observability.telemetry.

Tests cover:
- TelemetryEvent construction and to_dict()
- TelemetryCollector.record() enqueues events
- Opt-out via NINETRIX_TELEMETRY=off disables recording
- Queue bounded at _MAX_QUEUE (oldest dropped)
- flush() sends batch and clears queue
- flush() is no-op when queue is empty
- All send errors are swallowed silently
- start_flush_loop() schedules background task
- stop() cancels loop and flushes remaining
- record_event() convenience wrapper
- _stable_machine_id() returns 16-char hex string
- No PII fields in TelemetryEvent
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest

from ninetrix.observability.telemetry import (
    TelemetryCollector,
    TelemetryEvent,
    _stable_machine_id,
    record_event,
    _telemetry,
)


# ===========================================================================
# TelemetryEvent
# ===========================================================================

class TestTelemetryEvent:
    def test_event_name_required(self):
        evt = TelemetryEvent(event="agent_run")
        assert evt.event == "agent_run"

    def test_defaults_populated(self):
        evt = TelemetryEvent(event="test")
        assert evt.sdk_version != ""
        assert evt.python_version != ""
        assert evt.platform != ""
        assert evt.machine_id != ""
        assert evt.provider == ""
        assert evt.model == ""
        assert evt.success is True
        assert evt.error_code == ""

    def test_to_dict_has_all_fields(self):
        evt = TelemetryEvent(
            event="agent_run",
            provider="anthropic",
            model="claude-sonnet-4-6",
            tool_count=3,
            duration_ms=1200,
            tokens_used=420,
            steps=4,
            success=True,
            error_code="",
        )
        d = evt.to_dict()
        assert d["event"] == "agent_run"
        assert d["provider"] == "anthropic"
        assert d["model"] == "claude-sonnet-4-6"
        assert d["tool_count"] == 3
        assert d["duration_ms"] == 1200
        assert d["tokens_used"] == 420
        assert d["steps"] == 4
        assert d["success"] is True
        assert d["error_code"] == ""

    def test_no_pii_fields_in_to_dict(self):
        """to_dict() must not expose prompts, outputs, API keys, or user data."""
        evt = TelemetryEvent(event="test")
        d = evt.to_dict()
        forbidden = {"prompt", "output", "content", "api_key", "email",
                     "user_id", "ip", "message", "history"}
        for key in d:
            assert key not in forbidden, f"PII field '{key}' found in telemetry payload"

    def test_error_event(self):
        evt = TelemetryEvent(
            event="agent_run",
            success=False,
            error_code="ProviderError",
        )
        assert evt.success is False
        assert evt.error_code == "ProviderError"
        d = evt.to_dict()
        assert d["success"] is False
        assert d["error_code"] == "ProviderError"


# ===========================================================================
# _stable_machine_id
# ===========================================================================

class TestMachineId:
    def test_returns_16_char_hex(self):
        mid = _stable_machine_id()
        assert len(mid) == 16
        assert all(c in "0123456789abcdef" for c in mid)

    def test_stable_across_calls(self):
        assert _stable_machine_id() == _stable_machine_id()

    def test_fallback_on_socket_error(self):
        with patch("socket.gethostname", side_effect=OSError("no hostname")):
            mid = _stable_machine_id()
        # Should return "unknown" fallback
        assert mid == "unknown"


# ===========================================================================
# TelemetryCollector — opt-out
# ===========================================================================

class TestOptOut:
    def test_disabled_when_env_off(self):
        with patch.dict(os.environ, {"NINETRIX_TELEMETRY": "off"}):
            col = TelemetryCollector()
        assert col.enabled is False

    def test_disabled_when_env_0(self):
        with patch.dict(os.environ, {"NINETRIX_TELEMETRY": "0"}):
            col = TelemetryCollector()
        assert col.enabled is False

    def test_disabled_when_env_false(self):
        with patch.dict(os.environ, {"NINETRIX_TELEMETRY": "false"}):
            col = TelemetryCollector()
        assert col.enabled is False

    def test_enabled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("NINETRIX_TELEMETRY", None)
            with patch.dict(os.environ, env, clear=True):
                col = TelemetryCollector()
        assert col.enabled is True

    def test_record_noop_when_disabled(self):
        with patch.dict(os.environ, {"NINETRIX_TELEMETRY": "off"}):
            col = TelemetryCollector()
        col.record(TelemetryEvent(event="test"))
        assert col.queue_size == 0


# ===========================================================================
# TelemetryCollector — record / queue
# ===========================================================================

class TestRecord:
    def _make_collector(self) -> TelemetryCollector:
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("NINETRIX_TELEMETRY", None)
            with patch.dict(os.environ, env, clear=True):
                return TelemetryCollector(_http_post=AsyncMock())

    def test_record_enqueues_event(self):
        col = self._make_collector()
        col.record(TelemetryEvent(event="test"))
        assert col.queue_size == 1

    def test_record_multiple(self):
        col = self._make_collector()
        for i in range(5):
            col.record(TelemetryEvent(event=f"evt_{i}"))
        assert col.queue_size == 5

    def test_queue_bounded_drops_oldest(self):
        from ninetrix.observability.telemetry import _MAX_QUEUE
        col = self._make_collector()
        # Fill past the limit
        for i in range(_MAX_QUEUE + 10):
            col.record(TelemetryEvent(event=f"e{i}"))
        assert col.queue_size == _MAX_QUEUE
        # Oldest (e0) should be gone, newest should be present
        last = col._queue[-1]
        assert last.event == f"e{_MAX_QUEUE + 9}"


# ===========================================================================
# TelemetryCollector — flush
# ===========================================================================

class TestFlush:
    def _make_collector(self, http_post=None) -> TelemetryCollector:
        mock_post = http_post or AsyncMock()
        env = os.environ.copy()
        env.pop("NINETRIX_TELEMETRY", None)
        with patch.dict(os.environ, env, clear=True):
            return TelemetryCollector(_http_post=mock_post)

    async def test_flush_sends_batch(self):
        mock_post = AsyncMock()
        col = self._make_collector(mock_post)
        col.record(TelemetryEvent(event="run1"))
        col.record(TelemetryEvent(event="run2"))

        await col.flush()

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        payload = call_args[0][1]  # positional arg: (url, payload)
        assert len(payload["events"]) == 2
        assert payload["events"][0]["event"] == "run1"
        assert payload["events"][1]["event"] == "run2"

    async def test_flush_clears_queue(self):
        col = self._make_collector()
        col.record(TelemetryEvent(event="x"))
        await col.flush()
        assert col.queue_size == 0

    async def test_flush_noop_when_empty(self):
        mock_post = AsyncMock()
        col = self._make_collector(mock_post)
        await col.flush()
        mock_post.assert_not_called()

    async def test_flush_swallows_send_error(self):
        mock_post = AsyncMock(side_effect=Exception("network down"))
        col = self._make_collector(mock_post)
        col.record(TelemetryEvent(event="x"))
        # Must not raise
        await col.flush()

    async def test_flush_noop_when_disabled(self):
        with patch.dict(os.environ, {"NINETRIX_TELEMETRY": "off"}):
            col = TelemetryCollector(_http_post=AsyncMock())
        col._queue.append(TelemetryEvent(event="sneaked"))  # bypass record()
        col._enabled = False
        mock_post = AsyncMock()
        col._http_post = mock_post
        await col.flush()
        mock_post.assert_not_called()

    async def test_send_includes_machine_id(self):
        mock_post = AsyncMock()
        col = self._make_collector(mock_post)
        col.record(TelemetryEvent(event="x"))
        await col.flush()
        payload = mock_post.call_args[0][1]
        assert "machine_id" in payload["events"][0]
        assert len(payload["events"][0]["machine_id"]) == 16


# ===========================================================================
# TelemetryCollector — flush loop
# ===========================================================================

class TestFlushLoop:
    async def test_start_flush_loop_creates_task(self):
        env = os.environ.copy()
        env.pop("NINETRIX_TELEMETRY", None)
        with patch.dict(os.environ, env, clear=True):
            col = TelemetryCollector(_http_post=AsyncMock(), flush_interval=0.05)

        col.start_flush_loop()
        assert col._flush_task is not None
        assert not col._flush_task.done()
        col._flush_task.cancel()
        try:
            await col._flush_task
        except asyncio.CancelledError:
            pass

    async def test_start_flush_loop_idempotent(self):
        env = os.environ.copy()
        env.pop("NINETRIX_TELEMETRY", None)
        with patch.dict(os.environ, env, clear=True):
            col = TelemetryCollector(_http_post=AsyncMock(), flush_interval=0.05)

        col.start_flush_loop()
        task1 = col._flush_task
        col.start_flush_loop()  # second call
        assert col._flush_task is task1  # same task, not replaced
        task1.cancel()
        try:
            await task1
        except asyncio.CancelledError:
            pass

    async def test_stop_cancels_loop_and_flushes(self):
        mock_post = AsyncMock()
        env = os.environ.copy()
        env.pop("NINETRIX_TELEMETRY", None)
        with patch.dict(os.environ, env, clear=True):
            col = TelemetryCollector(_http_post=mock_post, flush_interval=60.0)

        col.start_flush_loop()
        col.record(TelemetryEvent(event="pending"))
        await col.stop()

        # Queue flushed
        assert col.queue_size == 0
        mock_post.assert_called_once()

    async def test_flush_loop_triggers_send_on_interval(self):
        mock_post = AsyncMock()
        env = os.environ.copy()
        env.pop("NINETRIX_TELEMETRY", None)
        with patch.dict(os.environ, env, clear=True):
            col = TelemetryCollector(_http_post=mock_post, flush_interval=0.05)

        col.start_flush_loop()
        col.record(TelemetryEvent(event="auto"))
        await asyncio.sleep(0.15)  # allow ≥1 flush
        await col.stop()

        assert mock_post.call_count >= 1


# ===========================================================================
# record_event convenience function
# ===========================================================================

class TestRecordEvent:
    def test_record_event_enqueues_to_singleton(self):
        original_size = _telemetry.queue_size
        if _telemetry.enabled:
            record_event("unit_test_event", provider="test", success=True)
            assert _telemetry.queue_size == original_size + 1
            # Clean up
            _telemetry._queue.clear()

    def test_record_event_ignores_unknown_kwargs(self):
        """Unknown fields must be silently dropped — forward-compatible."""
        if _telemetry.enabled:
            before = _telemetry.queue_size
            record_event("test", unknown_future_field="value", provider="x")
            after = _telemetry.queue_size
            assert after == before + 1
            _telemetry._queue.clear()

    def test_record_event_maps_known_fields(self):
        if _telemetry.enabled:
            record_event(
                "agent_run",
                provider="openai",
                model="gpt-4o",
                tool_count=2,
                steps=3,
                success=False,
                error_code="ToolError",
            )
            evt = _telemetry._queue[-1]
            assert evt.provider == "openai"
            assert evt.model == "gpt-4o"
            assert evt.tool_count == 2
            assert evt.success is False
            assert evt.error_code == "ToolError"
            _telemetry._queue.clear()


# ===========================================================================
# lifespan integration — hooks fire correctly
# ===========================================================================

class TestLifespanIntegration:
    async def test_start_telemetry_called_on_startup(self):
        """lifespan.startup() should activate the telemetry flush loop."""
        from ninetrix._internals.lifespan import _start_telemetry_if_available
        # Should not raise — telemetry module now exists
        _start_telemetry_if_available()
        if _telemetry.enabled and _telemetry._flush_task:
            _telemetry._flush_task.cancel()
            try:
                await _telemetry._flush_task
            except asyncio.CancelledError:
                pass

    async def test_flush_telemetry_called_on_shutdown(self):
        """lifespan.shutdown() should flush pending events."""
        from ninetrix._internals.lifespan import _flush_telemetry_if_available
        _telemetry._queue.clear()
        if _telemetry.enabled:
            _telemetry.record(TelemetryEvent(event="shutdown_test"))
        # flush should clear the queue (even if HTTP fails — swallowed)
        await _flush_telemetry_if_available()
        assert _telemetry.queue_size == 0


# ===========================================================================
# Top-level re-exports
# ===========================================================================

class TestReExports:
    def test_top_level_exports(self):
        import ninetrix
        assert hasattr(ninetrix, "TelemetryEvent")
        assert hasattr(ninetrix, "TelemetryCollector")
        assert hasattr(ninetrix, "record_event")

    def test_observability_package_exports(self):
        from ninetrix.observability import (
            TelemetryEvent, TelemetryCollector, record_event
        )
        assert TelemetryEvent is not None
        assert TelemetryCollector is not None
        assert callable(record_event)
