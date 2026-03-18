"""
ninetrix.observability.telemetry
=================================
L6 kernel — stdlib only.

Anonymous, privacy-safe usage telemetry for the ninetrix SDK.

**What is collected** (no PII, no content):

    event           string — event name (e.g. "agent_run")
    machine_id      16-char opaque hash of hostname+MAC (stable across runs)
    sdk_version     ninetrix-sdk version string
    python_version  "3.11.9"
    platform        "darwin" / "linux" / "win32"
    provider        "anthropic" / "openai" / etc.
    model           model string (e.g. "claude-sonnet-4-6")
    tool_count      int
    duration_ms     int
    tokens_used     int
    steps           int
    success         bool
    error_code      error class name or "" on success

**What is never collected**: prompts, outputs, tool arguments, API keys,
user IDs, email addresses, IP addresses.

Opt-out::

    export NINETRIX_TELEMETRY=off

Batch flush every 30 s (or at shutdown).  All network errors are silently
swallowed — telemetry must never affect agent execution.

Usage inside the SDK::

    from ninetrix.observability.telemetry import record_event

    record_event(
        "agent_run",
        provider="anthropic",
        model="claude-sonnet-4-6",
        tool_count=3,
        duration_ms=1200,
        tokens_used=420,
        steps=4,
        success=True,
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import platform
import socket
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any

try:
    from importlib.metadata import version as _pkg_version
    _SDK_VERSION: str = _pkg_version("ninetrix-sdk")
except Exception:
    _SDK_VERSION = "0.0.0"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENV_OPT_OUT = "NINETRIX_TELEMETRY"
_ENDPOINT = "https://telemetry.ninetrix.io/v1/events"
_FLUSH_INTERVAL = 30.0   # seconds between automatic batch flushes
_MAX_QUEUE = 500          # drop oldest when queue exceeds this


# ---------------------------------------------------------------------------
# Machine ID (stable, opaque, no PII)
# ---------------------------------------------------------------------------

def _stable_machine_id() -> str:
    """
    Return a 16-char opaque identifier stable across SDK runs on the same machine.

    Derived from SHA-256(hostname + MAC address).  Neither hostname nor MAC is
    transmitted — only the truncated hash.
    """
    try:
        hostname = socket.gethostname()
        mac = uuid.getnode()
        raw = f"{hostname}:{mac}".encode()
        return hashlib.sha256(raw).hexdigest()[:16]
    except Exception:
        return "unknown"


_MACHINE_ID: str = _stable_machine_id()


# ---------------------------------------------------------------------------
# TelemetryEvent
# ---------------------------------------------------------------------------

@dataclass
class TelemetryEvent:
    """
    A single anonymous telemetry event.  No PII, no content fields.

    All fields have safe defaults so callers can populate only what they know.
    """

    event: str                          # event name, e.g. "agent_run"
    machine_id: str = field(default_factory=lambda: _MACHINE_ID)
    sdk_version: str = field(default_factory=lambda: _SDK_VERSION)
    python_version: str = field(
        default_factory=lambda: ".".join(str(v) for v in sys.version_info[:3])
    )
    platform: str = field(default_factory=lambda: sys.platform)
    provider: str = ""
    model: str = ""
    tool_count: int = 0
    duration_ms: int = 0
    tokens_used: int = 0
    steps: int = 0
    success: bool = True
    error_code: str = ""                # exception class name, or "" on success

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "machine_id": self.machine_id,
            "sdk_version": self.sdk_version,
            "python_version": self.python_version,
            "platform": self.platform,
            "provider": self.provider,
            "model": self.model,
            "tool_count": self.tool_count,
            "duration_ms": self.duration_ms,
            "tokens_used": self.tokens_used,
            "steps": self.steps,
            "success": self.success,
            "error_code": self.error_code,
        }


# ---------------------------------------------------------------------------
# TelemetryCollector
# ---------------------------------------------------------------------------

class TelemetryCollector:
    """
    Batches ``TelemetryEvent`` objects and flushes them asynchronously.

    - Events are added synchronously via ``record()``; the queue is
      thread-safe for single-threaded asyncio use.
    - ``start_flush_loop()`` schedules a background task that flushes every
      ``_FLUSH_INTERVAL`` seconds.  Call it once at ``startup()``.
    - ``flush()`` sends the current batch immediately (called at shutdown).
    - All network errors are silently swallowed.
    - When opt-out is active, ``record()`` is a no-op.
    """

    def __init__(
        self,
        endpoint: str = _ENDPOINT,
        flush_interval: float = _FLUSH_INTERVAL,
        *,
        _http_post: Any = None,   # injectable for tests
    ) -> None:
        self._endpoint = endpoint
        self._flush_interval = flush_interval
        self._queue: list[TelemetryEvent] = []
        self._enabled: bool = self._check_enabled()
        self._flush_task: asyncio.Task | None = None  # type: ignore[type-arg]
        self._http_post = _http_post  # async callable(url, payload) → None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, event: TelemetryEvent) -> None:
        """Enqueue an event.  No-op if telemetry is disabled."""
        if not self._enabled:
            return
        if len(self._queue) >= _MAX_QUEUE:
            self._queue.pop(0)  # drop oldest to stay bounded
        self._queue.append(event)

    def start_flush_loop(self) -> None:
        """
        Schedule the background flush loop in the running event loop.
        Idempotent — safe to call multiple times.
        """
        if not self._enabled:
            return
        if self._flush_task is not None and not self._flush_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
            self._flush_task = loop.create_task(self._flush_loop())
        except RuntimeError:
            pass  # no running loop — flush will happen at shutdown only

    async def flush(self) -> None:
        """Send all queued events now.  Silently swallows all errors."""
        if not self._enabled or not self._queue:
            return
        batch = list(self._queue)
        self._queue.clear()
        await self._send(batch)

    async def stop(self) -> None:
        """Cancel the flush loop and flush remaining events."""
        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self.flush()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_enabled(self) -> bool:
        val = os.environ.get(_ENV_OPT_OUT, "").lower()
        return val not in ("off", "0", "false", "no", "disable", "disabled")

    async def _flush_loop(self) -> None:
        """Background task: flush every _flush_interval seconds."""
        try:
            while True:
                await asyncio.sleep(self._flush_interval)
                await self.flush()
        except asyncio.CancelledError:
            pass

    async def _send(self, events: list[TelemetryEvent]) -> None:
        """
        POST the batch to the telemetry endpoint.
        All errors are silently swallowed — telemetry must never affect
        agent execution.
        """
        if not events:
            return

        payload = {"events": [e.to_dict() for e in events]}

        try:
            if self._http_post is not None:
                await self._http_post(self._endpoint, payload)
            else:
                await self._default_send(payload)
        except Exception:
            pass  # swallow everything — telemetry is best-effort

    async def _default_send(self, payload: dict[str, Any]) -> None:
        """Send via the ninetrix httpx singleton (available after startup)."""
        try:
            from ninetrix._internals.http import get_http_client
            client = get_http_client()
            await client.post(
                self._endpoint,
                json=payload,
                timeout=5.0,
            )
        except Exception:
            pass  # swallow — telemetry is best-effort


# ---------------------------------------------------------------------------
# Module-level singleton + convenience function
# ---------------------------------------------------------------------------

_telemetry: TelemetryCollector = TelemetryCollector()


def record_event(event: str, **kwargs: Any) -> None:
    """
    Convenience wrapper — create and enqueue a ``TelemetryEvent`` in one call.

    All kwargs map directly to ``TelemetryEvent`` fields.  Unknown keys are
    silently ignored so callers are forward-compatible with new fields.

    Example::

        from ninetrix.observability.telemetry import record_event

        record_event(
            "agent_run",
            provider="anthropic",
            model="claude-sonnet-4-6",
            tool_count=2,
            duration_ms=850,
            tokens_used=310,
            steps=3,
            success=True,
        )
    """
    _valid_fields = TelemetryEvent.__dataclass_fields__
    filtered = {k: v for k, v in kwargs.items() if k in _valid_fields}
    _telemetry.record(TelemetryEvent(event=event, **filtered))
