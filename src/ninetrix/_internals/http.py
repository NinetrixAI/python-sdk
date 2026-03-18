"""
ninetrix._internals.http
========================
L1 kernel — stdlib + httpx only.

Process-wide httpx.AsyncClient singleton with connection pooling.
All SDK components that make outbound HTTP calls (AgentClient, RemoteAgent,
MCPToolSource, telemetry) share this single client.

Usage:
    from ninetrix._internals.http import get_http_client, close_http_client

    # Correct — use the client directly (singleton stays open):
    client = get_http_client()
    resp = await client.post(url, json=payload)

    # WARNING: do NOT use `async with get_http_client() as client:` — httpx's
    # __aexit__ closes the client, destroying the connection pool.
    # Use http_client_lifespan() if you need a managed close-on-exit context.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx

try:
    import importlib.metadata as _meta
    _SDK_VERSION = _meta.version("ninetrix-sdk")
except Exception:
    _SDK_VERSION = "0.0.0"

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

_client: httpx.AsyncClient | None = None
_lock = asyncio.Lock()

# Connection pool defaults — matches httpx defaults but explicit for clarity
_LIMITS = httpx.Limits(
    max_connections=100,
    max_keepalive_connections=20,
    keepalive_expiry=30.0,
)

_DEFAULT_HEADERS = {
    "User-Agent": f"ninetrix-sdk/{_SDK_VERSION} python-httpx/{httpx.__version__}",
    "X-Ninetrix-SDK-Version": _SDK_VERSION,
}

_DEFAULT_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=60.0,
    write=30.0,
    pool=5.0,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_http_client() -> httpx.AsyncClient:
    """
    Return the process-wide httpx.AsyncClient singleton, creating it if needed.

    The client is created lazily on first call and reused for the process lifetime.
    Call close_http_client() (or use lifespan()) to shut it down cleanly on exit.

    Note: This function is synchronous — the client is not connected until the
    first request. httpx manages connection pooling internally.
    """
    global _client
    if _client is None or _client.is_closed:
        _client = _build_client()
    return _client


async def close_http_client() -> None:
    """
    Close the singleton client and release all pooled connections.
    Safe to call multiple times or when no client exists.
    Called automatically by lifespan.shutdown().
    """
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


@asynccontextmanager
async def http_client_lifespan() -> AsyncIterator[httpx.AsyncClient]:
    """
    Context manager that yields the singleton client and closes it on exit.
    Use when the SDK owns the lifespan (standalone scripts, tests).

    Example:
        async with http_client_lifespan() as client:
            resp = await client.get("https://example.com")
    """
    client = get_http_client()
    try:
        yield client
    finally:
        await close_http_client()


# ---------------------------------------------------------------------------
# Internal builder
# ---------------------------------------------------------------------------

def _build_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers=_DEFAULT_HEADERS,
        timeout=_DEFAULT_TIMEOUT,
        limits=_LIMITS,
        follow_redirects=True,
    )
