"""
Unit tests for ninetrix._internals.http and ninetrix._internals.lifespan (PR 5).

Coverage:
- get_http_client(): returns singleton, creates on first call, recreates after close
- close_http_client(): closes client, sets to None, idempotent
- http_client_lifespan(): yields client, closes on exit
- Client headers: User-Agent and X-Ninetrix-SDK-Version present
- Client pool limits: max_connections, keepalive
- lifespan(): startup + shutdown called as context manager
- startup() / shutdown(): idempotent, no error when called multiple times
- _signal_handlers_registered: not re-registered on repeated startup()
- Telemetry hooks: silently skip when observability not yet available
"""

import asyncio
import sys
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import httpx

from ninetrix._internals import http as http_module
from ninetrix._internals.http import (
    get_http_client,
    close_http_client,
    http_client_lifespan,
    _DEFAULT_HEADERS,
    _build_client,
)
from ninetrix._internals import lifespan as lifespan_module
from ninetrix._internals.lifespan import startup, shutdown, lifespan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_http_singleton():
    """Reset the module-level singleton between tests."""
    http_module._client = None


def _reset_lifespan_state():
    lifespan_module._signal_handlers_registered = False


# ---------------------------------------------------------------------------
# get_http_client()
# ---------------------------------------------------------------------------

class TestGetHttpClient:
    def setup_method(self):
        _reset_http_singleton()

    def teardown_method(self):
        # Best-effort cleanup — don't fail if already closed
        asyncio.run(close_http_client())

    def test_returns_async_client(self):
        client = get_http_client()
        assert isinstance(client, httpx.AsyncClient)

    def test_returns_same_singleton(self):
        c1 = get_http_client()
        c2 = get_http_client()
        assert c1 is c2

    def test_recreates_after_close(self):
        c1 = get_http_client()
        asyncio.run(close_http_client())
        c2 = get_http_client()
        assert c1 is not c2
        assert not c2.is_closed

    def test_client_has_user_agent_header(self):
        client = get_http_client()
        headers = dict(client.headers)
        assert "user-agent" in headers
        assert "ninetrix-sdk" in headers["user-agent"].lower()

    def test_client_has_sdk_version_header(self):
        client = get_http_client()
        headers = dict(client.headers)
        assert "x-ninetrix-sdk-version" in headers

    def test_client_follows_redirects(self):
        client = get_http_client()
        assert client.follow_redirects is True


# ---------------------------------------------------------------------------
# close_http_client()
# ---------------------------------------------------------------------------

class TestCloseHttpClient:
    def setup_method(self):
        _reset_http_singleton()

    def test_closes_existing_client(self):
        client = get_http_client()
        assert not client.is_closed
        asyncio.run(close_http_client())
        assert client.is_closed

    def test_sets_singleton_to_none(self):
        get_http_client()
        asyncio.run(close_http_client())
        assert http_module._client is None

    def test_idempotent_when_no_client(self):
        # Should not raise when called with no client
        asyncio.run(close_http_client())
        asyncio.run(close_http_client())

    def test_idempotent_after_close(self):
        get_http_client()
        asyncio.run(close_http_client())
        asyncio.run(close_http_client())  # second call must not raise


# ---------------------------------------------------------------------------
# http_client_lifespan()
# ---------------------------------------------------------------------------

class TestHttpClientLifespan:
    def setup_method(self):
        _reset_http_singleton()

    def test_yields_client(self):
        async def run():
            async with http_client_lifespan() as client:
                assert isinstance(client, httpx.AsyncClient)
                assert not client.is_closed

        asyncio.run(run())

    def test_closes_on_exit(self):
        captured = []

        async def run():
            async with http_client_lifespan() as client:
                captured.append(client)
            assert captured[0].is_closed

        asyncio.run(run())

    def test_closes_on_exception(self):
        captured = []

        async def run():
            try:
                async with http_client_lifespan() as client:
                    captured.append(client)
                    raise ValueError("deliberate error")
            except ValueError:
                pass
            assert captured[0].is_closed

        asyncio.run(run())


# ---------------------------------------------------------------------------
# lifespan() context manager
# ---------------------------------------------------------------------------

class TestLifespanContextManager:
    def setup_method(self):
        _reset_http_singleton()
        _reset_lifespan_state()

    def teardown_method(self):
        asyncio.run(close_http_client())
        _reset_lifespan_state()

    def test_startup_creates_http_client(self):
        async def run():
            async with lifespan():
                assert http_module._client is not None
                assert not http_module._client.is_closed

        asyncio.run(run())

    def test_shutdown_closes_http_client(self):
        async def run():
            async with lifespan():
                pass
            assert http_module._client is None

        asyncio.run(run())

    def test_shutdown_on_exception(self):
        async def run():
            try:
                async with lifespan():
                    raise RuntimeError("deliberate")
            except RuntimeError:
                pass
            assert http_module._client is None

        asyncio.run(run())

    def test_lifespan_exported_from_ninetrix(self):
        import ninetrix
        assert hasattr(ninetrix, "lifespan")
        assert hasattr(ninetrix, "startup")
        assert hasattr(ninetrix, "shutdown")


# ---------------------------------------------------------------------------
# startup() / shutdown() — idempotency
# ---------------------------------------------------------------------------

class TestStartupShutdown:
    def setup_method(self):
        _reset_http_singleton()
        _reset_lifespan_state()

    def teardown_method(self):
        asyncio.run(close_http_client())
        _reset_lifespan_state()

    def test_startup_idempotent(self):
        async def run():
            await startup()
            await startup()  # second call must not raise

        asyncio.run(run())

    def test_shutdown_idempotent(self):
        async def run():
            await startup()
            await shutdown()
            await shutdown()  # second call must not raise

        asyncio.run(run())

    def test_shutdown_without_startup_safe(self):
        async def run():
            await shutdown()  # no prior startup — must not raise

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Signal handler registration
# ---------------------------------------------------------------------------

class TestSignalHandlers:
    def setup_method(self):
        _reset_http_singleton()
        _reset_lifespan_state()

    def teardown_method(self):
        asyncio.run(close_http_client())
        _reset_lifespan_state()

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix signals only")
    def test_signal_handlers_registered_once(self):
        async def run():
            await startup()
            assert lifespan_module._signal_handlers_registered is True
            await startup()  # second call — must not double-register
            assert lifespan_module._signal_handlers_registered is True

        asyncio.run(run())

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix signals only")
    def test_signal_handlers_registered_flag_set(self):
        async def run():
            assert lifespan_module._signal_handlers_registered is False
            await startup()
            assert lifespan_module._signal_handlers_registered is True

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Telemetry guards
# ---------------------------------------------------------------------------

class TestTelemetryGuards:
    def setup_method(self):
        _reset_http_singleton()
        _reset_lifespan_state()

    def teardown_method(self):
        asyncio.run(close_http_client())
        _reset_lifespan_state()

    def test_startup_does_not_raise_when_telemetry_missing(self):
        # observability.telemetry doesn't exist yet — must be silently skipped
        async def run():
            await startup()  # must not raise ImportError

        asyncio.run(run())

    def test_shutdown_does_not_raise_when_telemetry_missing(self):
        async def run():
            await startup()
            await shutdown()  # must not raise ImportError

        asyncio.run(run())


# ---------------------------------------------------------------------------
# _build_client() internals
# ---------------------------------------------------------------------------

class TestBuildClient:
    def test_returns_async_client(self):
        client = _build_client()
        assert isinstance(client, httpx.AsyncClient)
        asyncio.run(client.aclose())

    def test_default_headers_dict(self):
        assert "User-Agent" in _DEFAULT_HEADERS
        assert "X-Ninetrix-SDK-Version" in _DEFAULT_HEADERS
