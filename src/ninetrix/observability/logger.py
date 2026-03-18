"""
ninetrix.observability.logger
=============================
L6 kernel — stdlib only.

``NinetrixLogger`` wraps the stdlib ``logging`` module and adds:

- Structured context via ``**ctx`` kwargs on every log call (thread_id,
  tool_name, agent_name, etc.)
- Two output formats, auto-selected:
    - **Human** (default when stderr is a TTY): ``[LEVEL] message  key=value …``
    - **JSON** (CI/prod): ``{"ts": "…", "level": "…", "msg": "…", "key": …}``
  Force JSON with ``NINETRIX_LOG_FORMAT=json``.
- ``enable_debug()`` — flip the root ninetrix logger to DEBUG in one call.

Quick start::

    from ninetrix.observability.logger import NinetrixLogger, enable_debug

    log = NinetrixLogger(__name__)
    log.info("tool called", tool_name="search", thread_id="t-123")
    log.warning("budget warning", spent_usd=0.40, budget_usd=0.50)

    enable_debug()   # now DEBUG messages are visible too
    log.debug("token chunk", content="Paris")
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROOT_LOGGER_NAME = "ninetrix"
_ENV_FORMAT = "NINETRIX_LOG_FORMAT"   # set to "json" to force JSON output
_ENV_LEVEL = "NINETRIX_LOG_LEVEL"     # set to "DEBUG" / "INFO" / etc.


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class _HumanFormatter(logging.Formatter):
    """
    Human-readable format for terminal output.

    ``[LEVEL]  name — message  key=value key=value``
    """

    _LEVEL_LABELS = {
        logging.DEBUG:    "DEBUG",
        logging.INFO:     "INFO ",
        logging.WARNING:  "WARN ",
        logging.ERROR:    "ERROR",
        logging.CRITICAL: "CRIT ",
    }

    def format(self, record: logging.LogRecord) -> str:
        level = self._LEVEL_LABELS.get(record.levelno, record.levelname)
        ctx: dict[str, Any] = getattr(record, "_ctx", {})
        ctx_str = "  " + "  ".join(f"{k}={v}" for k, v in ctx.items()) if ctx else ""
        name_part = f"{record.name} — " if record.name != _ROOT_LOGGER_NAME else ""
        return f"[{level}]  {name_part}{record.getMessage()}{ctx_str}"


class _JSONFormatter(logging.Formatter):
    """JSON-lines format for machine consumption."""

    def format(self, record: logging.LogRecord) -> str:
        ctx: dict[str, Any] = getattr(record, "_ctx", {})
        payload: dict[str, Any] = {
            "ts": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)
            ),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        payload.update(ctx)
        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Handler + root logger setup (lazy, done once)
# ---------------------------------------------------------------------------

_handler_installed = False


def _ensure_handler() -> None:
    """Install a single handler on the root ninetrix logger (idempotent)."""
    global _handler_installed
    if _handler_installed:
        return

    root = logging.getLogger(_ROOT_LOGGER_NAME)

    # Determine format: JSON when forced via env OR when stderr is not a TTY
    force_json = os.environ.get(_ENV_FORMAT, "").lower() == "json"
    is_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    use_json = force_json or not is_tty

    formatter: logging.Formatter = (
        _JSONFormatter() if use_json else _HumanFormatter()
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    # Avoid adding duplicate handlers if the user configured logging themselves
    if not root.handlers:
        root.addHandler(handler)

    # Default level from env, else WARNING (quiet by default)
    env_level = os.environ.get(_ENV_LEVEL, "").upper()
    level = getattr(logging, env_level, logging.WARNING)
    root.setLevel(level)

    _handler_installed = True


# ---------------------------------------------------------------------------
# Context-injecting log record factory
# ---------------------------------------------------------------------------

class _CtxFilter(logging.Filter):
    """Passes through; context is attached directly to records by NinetrixLogger."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "_ctx"):
            record._ctx = {}  # type: ignore[attr-defined]
        return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class NinetrixLogger:
    """
    Thin structured-logging wrapper around ``logging.Logger``.

    Usage::

        log = NinetrixLogger(__name__)
        log.info("agent started", agent_name="analyst", model="claude-sonnet-4-6")
        log.debug("tool call", tool_name="search", args={"q": "ESG"})
        log.warning("budget 80%", spent_usd=0.40, budget_usd=0.50)
        log.error("provider failed", provider="anthropic", status_code=500)

    All keyword arguments beyond the message are serialized as structured
    context fields (key=value in human mode, extra JSON keys in JSON mode).
    """

    def __init__(self, name: str = _ROOT_LOGGER_NAME) -> None:
        _ensure_handler()
        # Ensure name is under the ninetrix namespace so the root handler fires
        if name and not name.startswith(_ROOT_LOGGER_NAME):
            name = f"{_ROOT_LOGGER_NAME}.{name}"
        self._logger = logging.getLogger(name)
        self._logger.addFilter(_CtxFilter())

    # ------------------------------------------------------------------
    # Log methods
    # ------------------------------------------------------------------

    def debug(self, msg: str, **ctx: Any) -> None:
        self._log(logging.DEBUG, msg, ctx)

    def info(self, msg: str, **ctx: Any) -> None:
        self._log(logging.INFO, msg, ctx)

    def warning(self, msg: str, **ctx: Any) -> None:
        self._log(logging.WARNING, msg, ctx)

    def error(self, msg: str, **ctx: Any) -> None:
        self._log(logging.ERROR, msg, ctx)

    def exception(self, msg: str, **ctx: Any) -> None:
        """Log ERROR with current exception info attached."""
        self._log(logging.ERROR, msg, ctx, exc_info=True)

    # ------------------------------------------------------------------
    # Convenience: structured event helpers used internally by the SDK
    # ------------------------------------------------------------------

    def tool_call(self, tool_name: str, args: dict[str, Any], thread_id: str = "") -> None:
        """Emit a structured DEBUG record for a tool invocation."""
        self.debug(
            "tool_call",
            tool_name=tool_name,
            args=args,
            thread_id=thread_id,
        )

    def tool_result(
        self,
        tool_name: str,
        result_preview: str,
        elapsed_ms: float,
        thread_id: str = "",
    ) -> None:
        """Emit a structured DEBUG record for a tool result."""
        self.debug(
            "tool_result",
            tool_name=tool_name,
            result_preview=result_preview[:120],
            elapsed_ms=round(elapsed_ms, 1),
            thread_id=thread_id,
        )

    def turn_end(
        self,
        step: int,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        thread_id: str = "",
    ) -> None:
        """Emit a structured DEBUG record after each LLM turn."""
        self.debug(
            "turn_end",
            step=step,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost_usd, 6),
            thread_id=thread_id,
        )

    # ------------------------------------------------------------------
    # stdlib passthrough (for frameworks that set NinetrixLogger directly)
    # ------------------------------------------------------------------

    @property
    def level(self) -> int:
        return self._logger.level

    def setLevel(self, level: int | str) -> None:  # noqa: N802
        self._logger.setLevel(level)

    def isEnabledFor(self, level: int) -> bool:  # noqa: N802
        return self._logger.isEnabledFor(level)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log(
        self,
        level: int,
        msg: str,
        ctx: dict[str, Any],
        *,
        exc_info: bool = False,
    ) -> None:
        if not self._logger.isEnabledFor(level):
            return
        record = self._logger.makeRecord(
            self._logger.name,
            level,
            fn="",
            lno=0,
            msg=msg,
            args=(),
            exc_info=sys.exc_info() if exc_info else None,
        )
        record._ctx = ctx  # type: ignore[attr-defined]
        self._logger.handle(record)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def enable_debug() -> None:
    """
    Set the root ninetrix logger to DEBUG level.

    Call once at app startup to see per-turn, per-tool, per-token logs::

        from ninetrix.observability.logger import enable_debug
        enable_debug()
    """
    _ensure_handler()
    logging.getLogger(_ROOT_LOGGER_NAME).setLevel(logging.DEBUG)


def get_logger(name: str) -> NinetrixLogger:
    """
    Convenience constructor — mirrors ``logging.getLogger(name)``.

    Preferred over ``NinetrixLogger(name)`` for module-level usage::

        log = get_logger(__name__)
    """
    return NinetrixLogger(name)


# Module-level default logger — used inside SDK internals
logger: NinetrixLogger = get_logger(_ROOT_LOGGER_NAME)
