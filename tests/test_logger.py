"""
tests/test_logger.py
====================

Unit tests for ninetrix.observability.logger.

Tests cover:
- NinetrixLogger construction and log methods
- Structured context (**ctx) forwarded to records
- Human and JSON formatters produce expected output
- enable_debug() sets root logger to DEBUG
- get_logger() returns a NinetrixLogger
- Module-level logger exists
- Level filtering (DEBUG suppressed at WARNING level)
- tool_call / tool_result / turn_end helpers emit correctly
"""

from __future__ import annotations

import io
import json
import logging
import sys
from unittest.mock import patch

import pytest

# Reset _handler_installed between tests so fresh handlers can be set up
import ninetrix.observability.logger as _logger_mod


def _reset_logger_state() -> None:
    """Clear handler-installed flag and root logger handlers for test isolation."""
    _logger_mod._handler_installed = False
    root = logging.getLogger("ninetrix")
    root.handlers.clear()
    root.setLevel(logging.WARNING)


# ===========================================================================
# NinetrixLogger construction
# ===========================================================================

class TestNinetrixLoggerConstruction:
    def setup_method(self):
        _reset_logger_state()

    def test_default_name_is_ninetrix(self):
        from ninetrix.observability.logger import NinetrixLogger
        log = NinetrixLogger()
        assert log._logger.name == "ninetrix"

    def test_custom_name_prefixed_with_ninetrix(self):
        from ninetrix.observability.logger import NinetrixLogger
        log = NinetrixLogger("mymodule")
        assert log._logger.name == "ninetrix.mymodule"

    def test_already_prefixed_name_not_doubled(self):
        from ninetrix.observability.logger import NinetrixLogger
        log = NinetrixLogger("ninetrix.providers.anthropic")
        assert log._logger.name == "ninetrix.providers.anthropic"

    def test_get_logger_returns_ninetrix_logger(self):
        from ninetrix.observability.logger import get_logger, NinetrixLogger
        log = get_logger("test.mod")
        assert isinstance(log, NinetrixLogger)
        assert "test.mod" in log._logger.name

    def test_module_level_logger_exists(self):
        from ninetrix.observability.logger import logger, NinetrixLogger
        assert isinstance(logger, NinetrixLogger)

    def test_top_level_imports(self):
        import ninetrix
        assert hasattr(ninetrix, "NinetrixLogger")
        assert hasattr(ninetrix, "enable_debug")
        assert hasattr(ninetrix, "get_logger")


# ===========================================================================
# enable_debug
# ===========================================================================

class TestEnableDebug:
    def setup_method(self):
        _reset_logger_state()

    def teardown_method(self):
        # Restore to WARNING after each test
        logging.getLogger("ninetrix").setLevel(logging.WARNING)

    def test_enable_debug_sets_root_to_debug(self):
        from ninetrix.observability.logger import enable_debug
        enable_debug()
        root = logging.getLogger("ninetrix")
        assert root.level == logging.DEBUG

    def test_enable_debug_callable_multiple_times(self):
        from ninetrix.observability.logger import enable_debug
        enable_debug()
        enable_debug()  # idempotent
        assert logging.getLogger("ninetrix").level == logging.DEBUG

    def test_setlevel_on_logger(self):
        from ninetrix.observability.logger import NinetrixLogger
        log = NinetrixLogger("ltest")
        log.setLevel(logging.INFO)
        assert log.level == logging.INFO

    def test_is_enabled_for(self):
        from ninetrix.observability.logger import NinetrixLogger, enable_debug
        enable_debug()
        log = NinetrixLogger("ltest2")
        assert log.isEnabledFor(logging.DEBUG) is True


# ===========================================================================
# Level filtering
# ===========================================================================

class TestLevelFiltering:
    def setup_method(self):
        _reset_logger_state()

    def test_debug_suppressed_at_warning_level(self):
        """DEBUG records should not be emitted when level is WARNING."""
        buf = io.StringIO()
        from ninetrix.observability.logger import NinetrixLogger
        log = NinetrixLogger("filter_test")
        # Install a test handler at DEBUG to capture if something slips through
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.DEBUG)
        log._logger.addHandler(handler)
        log._logger.setLevel(logging.WARNING)  # explicit level on child

        log.debug("should not appear")
        assert buf.getvalue() == ""

    def test_warning_emitted_at_warning_level(self):
        buf = io.StringIO()
        from ninetrix.observability.logger import NinetrixLogger
        log = NinetrixLogger("emit_test")
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.DEBUG)
        log._logger.addHandler(handler)
        log._logger.propagate = False
        log._logger.setLevel(logging.DEBUG)

        log.warning("hello warning")
        assert "hello warning" in buf.getvalue()

    def test_info_emitted_when_level_is_info(self):
        buf = io.StringIO()
        from ninetrix.observability.logger import NinetrixLogger
        log = NinetrixLogger("info_test")
        handler = logging.StreamHandler(buf)
        log._logger.addHandler(handler)
        log._logger.propagate = False
        log._logger.setLevel(logging.INFO)

        log.info("info message")
        assert "info message" in buf.getvalue()

    def test_error_always_emitted(self):
        buf = io.StringIO()
        from ninetrix.observability.logger import NinetrixLogger
        log = NinetrixLogger("err_test")
        handler = logging.StreamHandler(buf)
        log._logger.addHandler(handler)
        log._logger.propagate = False
        log._logger.setLevel(logging.DEBUG)

        log.error("something failed")
        assert "something failed" in buf.getvalue()


# ===========================================================================
# Structured context (**ctx)
# ===========================================================================

class TestStructuredContext:
    """Verify **ctx kwargs are attached to log records."""

    def _capture_records(self, log_fn, msg, **ctx):
        """Call log_fn(msg, **ctx) and return the LogRecord produced."""
        records = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        from ninetrix.observability.logger import NinetrixLogger
        log = NinetrixLogger("ctx_test")
        handler = _Capture()
        handler.setLevel(logging.DEBUG)
        log._logger.addHandler(handler)
        log._logger.propagate = False
        log._logger.setLevel(logging.DEBUG)

        getattr(log, log_fn)(msg, **ctx)
        return records

    def test_ctx_attached_to_record(self):
        records = self._capture_records("info", "test msg", tool_name="search", thread_id="t1")
        assert len(records) == 1
        assert records[0]._ctx == {"tool_name": "search", "thread_id": "t1"}

    def test_empty_ctx_is_empty_dict(self):
        records = self._capture_records("warning", "bare msg")
        assert records[0]._ctx == {}

    def test_debug_ctx(self):
        records = self._capture_records("debug", "debug msg", step=3, cost=0.01)
        assert records[0]._ctx == {"step": 3, "cost": 0.01}

    def test_error_ctx(self):
        records = self._capture_records("error", "fail", provider="anthropic", status=500)
        assert records[0]._ctx["provider"] == "anthropic"
        assert records[0]._ctx["status"] == 500


# ===========================================================================
# Human formatter
# ===========================================================================

class TestHumanFormatter:
    def test_formats_message_and_ctx(self):
        from ninetrix.observability.logger import _HumanFormatter
        formatter = _HumanFormatter()
        record = logging.LogRecord(
            name="ninetrix.test", level=logging.INFO,
            pathname="", lineno=0, msg="hello world", args=(), exc_info=None,
        )
        record._ctx = {"tool": "search", "thread": "t1"}  # type: ignore[attr-defined]
        output = formatter.format(record)
        assert "hello world" in output
        assert "tool=search" in output
        assert "thread=t1" in output
        assert "[INFO" in output

    def test_no_ctx_no_trailing_spaces(self):
        from ninetrix.observability.logger import _HumanFormatter
        formatter = _HumanFormatter()
        record = logging.LogRecord(
            name="ninetrix", level=logging.WARNING,
            pathname="", lineno=0, msg="bare msg", args=(), exc_info=None,
        )
        record._ctx = {}  # type: ignore[attr-defined]
        output = formatter.format(record)
        assert "bare msg" in output
        assert "=" not in output

    def test_debug_label(self):
        from ninetrix.observability.logger import _HumanFormatter
        formatter = _HumanFormatter()
        record = logging.LogRecord(
            name="ninetrix", level=logging.DEBUG,
            pathname="", lineno=0, msg="dbg", args=(), exc_info=None,
        )
        record._ctx = {}  # type: ignore[attr-defined]
        assert "DEBUG" in formatter.format(record)

    def test_error_label(self):
        from ninetrix.observability.logger import _HumanFormatter
        formatter = _HumanFormatter()
        record = logging.LogRecord(
            name="ninetrix", level=logging.ERROR,
            pathname="", lineno=0, msg="err", args=(), exc_info=None,
        )
        record._ctx = {}  # type: ignore[attr-defined]
        assert "ERROR" in formatter.format(record)


# ===========================================================================
# JSON formatter
# ===========================================================================

class TestJSONFormatter:
    def test_output_is_valid_json(self):
        from ninetrix.observability.logger import _JSONFormatter
        formatter = _JSONFormatter()
        record = logging.LogRecord(
            name="ninetrix.test", level=logging.INFO,
            pathname="", lineno=0, msg="json msg", args=(), exc_info=None,
        )
        record._ctx = {"key": "val", "n": 42}  # type: ignore[attr-defined]
        output = formatter.format(record)
        data = json.loads(output)
        assert data["msg"] == "json msg"
        assert data["level"] == "INFO"
        assert data["key"] == "val"
        assert data["n"] == 42

    def test_ts_field_present(self):
        from ninetrix.observability.logger import _JSONFormatter
        formatter = _JSONFormatter()
        record = logging.LogRecord(
            name="ninetrix", level=logging.DEBUG,
            pathname="", lineno=0, msg="ts test", args=(), exc_info=None,
        )
        record._ctx = {}  # type: ignore[attr-defined]
        data = json.loads(formatter.format(record))
        assert "ts" in data
        assert "T" in data["ts"]

    def test_logger_field(self):
        from ninetrix.observability.logger import _JSONFormatter
        formatter = _JSONFormatter()
        record = logging.LogRecord(
            name="ninetrix.providers", level=logging.WARNING,
            pathname="", lineno=0, msg="warn", args=(), exc_info=None,
        )
        record._ctx = {}  # type: ignore[attr-defined]
        data = json.loads(formatter.format(record))
        assert data["logger"] == "ninetrix.providers"

    def test_ctx_keys_at_top_level(self):
        from ninetrix.observability.logger import _JSONFormatter
        formatter = _JSONFormatter()
        record = logging.LogRecord(
            name="ninetrix", level=logging.ERROR,
            pathname="", lineno=0, msg="fail", args=(), exc_info=None,
        )
        record._ctx = {"provider": "anthropic", "status_code": 500}  # type: ignore[attr-defined]
        data = json.loads(formatter.format(record))
        assert data["provider"] == "anthropic"
        assert data["status_code"] == 500

    def test_non_serializable_ctx_uses_str_fallback(self):
        from ninetrix.observability.logger import _JSONFormatter
        formatter = _JSONFormatter()
        record = logging.LogRecord(
            name="ninetrix", level=logging.INFO,
            pathname="", lineno=0, msg="obj", args=(), exc_info=None,
        )

        class _Unserializable:
            def __repr__(self): return "<obj>"

        record._ctx = {"obj": _Unserializable()}  # type: ignore[attr-defined]
        output = formatter.format(record)
        data = json.loads(output)
        assert "<obj>" in data["obj"]


# ===========================================================================
# Structured event helpers
# ===========================================================================

class TestEventHelpers:
    def _setup_capture(self, name="helper_test"):
        records = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        from ninetrix.observability.logger import NinetrixLogger
        log = NinetrixLogger(name)
        handler = _Capture()
        handler.setLevel(logging.DEBUG)
        log._logger.addHandler(handler)
        log._logger.propagate = False
        log._logger.setLevel(logging.DEBUG)
        return log, records

    def test_tool_call_emits_debug_record(self):
        log, records = self._setup_capture("tc_test")
        log.tool_call("search", {"q": "hello"}, thread_id="t1")
        assert len(records) == 1
        assert records[0].levelno == logging.DEBUG
        assert records[0]._ctx["tool_name"] == "search"
        assert records[0]._ctx["thread_id"] == "t1"

    def test_tool_result_truncates_long_result(self):
        log, records = self._setup_capture("tr_test")
        log.tool_result("search", "x" * 200, elapsed_ms=42.5, thread_id="t2")
        assert len(records) == 1
        assert len(records[0]._ctx["result_preview"]) <= 120
        assert records[0]._ctx["elapsed_ms"] == 42.5

    def test_turn_end_emits_correct_fields(self):
        log, records = self._setup_capture("te_test")
        log.turn_end(step=3, input_tokens=100, output_tokens=50, cost_usd=0.001234)
        assert records[0]._ctx["step"] == 3
        assert records[0]._ctx["input_tokens"] == 100
        assert records[0]._ctx["output_tokens"] == 50
        assert "cost_usd" in records[0]._ctx

    def test_tool_call_without_thread_id(self):
        log, records = self._setup_capture("tc2_test")
        log.tool_call("calc", {"a": 1})
        assert records[0]._ctx["tool_name"] == "calc"


# ===========================================================================
# NINETRIX_LOG_FORMAT env var
# ===========================================================================

class TestEnvFormat:
    def setup_method(self):
        _reset_logger_state()

    def test_json_format_forced_by_env(self):
        """When NINETRIX_LOG_FORMAT=json, handler should use _JSONFormatter."""
        import os
        from ninetrix.observability.logger import _JSONFormatter
        with patch.dict(os.environ, {"NINETRIX_LOG_FORMAT": "json"}):
            _reset_logger_state()
            from ninetrix.observability.logger import NinetrixLogger
            log = NinetrixLogger("env_json")
        # Check the root handler's formatter type
        root = logging.getLogger("ninetrix")
        if root.handlers:
            assert isinstance(root.handlers[0].formatter, _JSONFormatter)

    def test_human_format_when_no_env(self):
        """Without NINETRIX_LOG_FORMAT, non-TTY should use JSON, TTY uses human."""
        import os
        with patch.dict(os.environ, {}, clear=False):
            if "NINETRIX_LOG_FORMAT" in os.environ:
                del os.environ["NINETRIX_LOG_FORMAT"]
            _reset_logger_state()
            from ninetrix.observability.logger import NinetrixLogger
            NinetrixLogger("env_human")
        # Just verify no crash — format depends on TTY state

    def test_log_level_from_env(self):
        import os
        with patch.dict(os.environ, {"NINETRIX_LOG_LEVEL": "DEBUG"}):
            _reset_logger_state()
            from ninetrix.observability.logger import NinetrixLogger
            NinetrixLogger("env_debug")
        root = logging.getLogger("ninetrix")
        assert root.level == logging.DEBUG


# ===========================================================================
# observability package re-exports
# ===========================================================================

class TestObservabilityPackage:
    def test_package_exports_ninetrix_logger(self):
        from ninetrix.observability import NinetrixLogger, enable_debug, get_logger
        assert NinetrixLogger is not None
        assert callable(enable_debug)
        assert callable(get_logger)
