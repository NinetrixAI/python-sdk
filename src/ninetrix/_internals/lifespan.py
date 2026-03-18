"""
ninetrix._internals.lifespan
============================
L1 kernel — stdlib + sibling L1 modules only.

SDK startup / shutdown / graceful SIGTERM handling.

Without lifespan management:
- HTTP connections leak on exit
- In-flight checkpoints can corrupt on SIGTERM
- Telemetry events drop (once telemetry lands in PR 8)

Usage patterns:

    # FastAPI
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    import ninetrix

    @asynccontextmanager
    async def app_lifespan(app: FastAPI):
        async with ninetrix.lifespan():
            yield

    app = FastAPI(lifespan=app_lifespan)

    # Standalone async script
    async def main():
        async with ninetrix.lifespan():
            result = await agent.arun("...")

    # Manual (rarely needed)
    await ninetrix.startup()
    try:
        ...
    finally:
        await ninetrix.shutdown()
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

logger = logging.getLogger(__name__)

# Track whether signal handlers have been registered to avoid double-registration
_signal_handlers_registered = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def startup() -> None:
    """
    Initialize all SDK singletons. Call once before creating any Agent.

    What it does:
    - Warms up the httpx connection pool
    - Registers SIGTERM / SIGINT handlers for graceful shutdown
    - Starts telemetry flush loop (once observability/telemetry lands in PR 8)

    Idempotent — safe to call multiple times.
    """
    from ninetrix._internals.http import get_http_client

    # Warm up the connection pool (creates the singleton)
    get_http_client()
    logger.debug("ninetrix: HTTP client initialized")

    # Register signal handlers (no-op on Windows where add_signal_handler is unavailable)
    _register_signal_handlers()

    # Telemetry flush loop — guarded until PR 8 lands
    _start_telemetry_if_available()

    logger.debug("ninetrix: startup complete")


async def shutdown() -> None:
    """
    Flush all pending state and close resources. Call on application exit.

    What it does:
    - Flushes telemetry batch (once PR 8 lands)
    - Closes httpx client and drains connection pool
    - Cancels SDK background tasks cleanly

    Idempotent — safe to call multiple times.
    """
    from ninetrix._internals.http import close_http_client

    # Flush telemetry before closing HTTP (telemetry uses the HTTP client)
    await _flush_telemetry_if_available()

    await close_http_client()
    logger.debug("ninetrix: HTTP client closed")

    logger.debug("ninetrix: shutdown complete")


@asynccontextmanager
async def lifespan() -> AsyncIterator[None]:
    """
    Async context manager that calls startup() on enter and shutdown() on exit.

    The canonical way to manage SDK lifecycle. Works with FastAPI, standalone
    scripts, or any async framework.

    Example:
        async with ninetrix.lifespan():
            result = await agent.arun("Analyze AAPL")
    """
    await startup()
    try:
        yield
    finally:
        await shutdown()


# ---------------------------------------------------------------------------
# Signal handlers
# ---------------------------------------------------------------------------

def _register_signal_handlers() -> None:
    """
    Register SIGTERM and SIGINT handlers to trigger graceful_shutdown().
    No-op on Windows (add_signal_handler is not available).
    No-op if no running event loop exists (will be registered lazily).
    """
    global _signal_handlers_registered
    if _signal_handlers_registered:
        return

    if sys.platform == "win32":
        # Windows does not support add_signal_handler — skip silently
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — handlers will not be registered
        # (startup() should be called from async context)
        logger.debug("ninetrix: no running event loop — signal handlers not registered")
        return

    loop.add_signal_handler(
        signal.SIGTERM,
        lambda: asyncio.ensure_future(_graceful_shutdown(exit_code=0)),
    )
    loop.add_signal_handler(
        signal.SIGINT,
        lambda: asyncio.ensure_future(_graceful_shutdown(exit_code=130)),
    )
    _signal_handlers_registered = True
    logger.debug("ninetrix: SIGTERM/SIGINT handlers registered")


async def _graceful_shutdown(exit_code: int = 0) -> None:
    """
    Triggered by SIGTERM or SIGINT.
    Flushes state, closes resources, then exits with the given code.
    """
    logger.info("ninetrix: received shutdown signal — flushing state and exiting")
    try:
        await shutdown()
    finally:
        raise SystemExit(exit_code)


# ---------------------------------------------------------------------------
# Telemetry hooks — guarded until PR 8 (observability/telemetry) lands
# ---------------------------------------------------------------------------

def _start_telemetry_if_available() -> None:
    """Start telemetry flush loop if the module exists (PR 8+)."""
    try:
        from ninetrix.observability.telemetry import _telemetry  # noqa: F401
        _telemetry.start_flush_loop()
    except (ImportError, AttributeError):
        pass  # telemetry not yet implemented — skip silently


async def _flush_telemetry_if_available() -> None:
    """Flush pending telemetry events if the module exists (PR 8+)."""
    try:
        from ninetrix.observability.telemetry import _telemetry  # noqa: F401
        await _telemetry.flush()
    except (ImportError, AttributeError):
        pass  # telemetry not yet implemented — skip silently
