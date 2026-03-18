"""Tests for client/local.py (AgentClient) + client/remote.py (RemoteAgent) — PR 22."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ninetrix import AgentClient, RemoteAgent
from ninetrix._internals.types import AgentResult, CredentialError, ProviderError


# =============================================================================
# Helpers
# =============================================================================


def _mock_response(data: dict, status: int = 200) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json.return_value = data
    r.text = json.dumps(data)
    return r


_AGENT_RESULT_DATA = {
    "output": "answer",
    "thread_id": "t1",
    "tokens_used": 10,
    "input_tokens": 7,
    "output_tokens": 3,
    "cost_usd": 0.001,
    "steps": 1,
    "history": [],
}


# =============================================================================
# AgentClient — construction
# =============================================================================


def test_agent_client_name_defaults_to_url():
    c = AgentClient("http://agent:9000")
    assert c.name == "http://agent:9000"


def test_agent_client_name_explicit():
    c = AgentClient("http://agent:9000", name="analyst")
    assert c.name == "analyst"


def test_agent_client_strips_trailing_slash():
    c = AgentClient("http://agent:9000/")
    assert c.base_url == "http://agent:9000"


def test_agent_client_repr():
    c = AgentClient("http://agent:9000", name="analyst")
    assert "AgentClient" in repr(c)
    assert "analyst" in repr(c)


# =============================================================================
# AgentClient — arun (happy path)
# =============================================================================


@pytest.mark.asyncio
async def test_agent_client_arun_returns_agent_result():
    c = AgentClient("http://agent:9000", name="bot")
    mock_resp = _mock_response(_AGENT_RESULT_DATA)

    mock_http = MagicMock()
    mock_http.post = AsyncMock(return_value=mock_resp)

    with patch("ninetrix.client.local.get_http_client", return_value=mock_http):
        result = await c.arun("hello")

    assert isinstance(result, AgentResult)
    assert result.output == "answer"
    assert result.thread_id == "t1"
    assert result.tokens_used == 10


@pytest.mark.asyncio
async def test_agent_client_arun_posts_to_invoke():
    c = AgentClient("http://agent:9000")
    mock_resp = _mock_response(_AGENT_RESULT_DATA)
    mock_http = MagicMock()
    mock_http.post = AsyncMock(return_value=mock_resp)

    with patch("ninetrix.client.local.get_http_client", return_value=mock_http):
        await c.arun("hello", thread_id="t1")

    call_args = mock_http.post.call_args
    assert "/invoke" in call_args[0][0]
    assert call_args[1]["json"]["message"] == "hello"
    assert call_args[1]["json"]["thread_id"] == "t1"


@pytest.mark.asyncio
async def test_agent_client_arun_no_thread_id_omitted():
    c = AgentClient("http://agent:9000")
    mock_resp = _mock_response(_AGENT_RESULT_DATA)
    mock_http = MagicMock()
    mock_http.post = AsyncMock(return_value=mock_resp)

    with patch("ninetrix.client.local.get_http_client", return_value=mock_http):
        await c.arun("hello")

    payload = mock_http.post.call_args[1]["json"]
    assert "thread_id" not in payload


# =============================================================================
# AgentClient — error handling
# =============================================================================


@pytest.mark.asyncio
async def test_agent_client_http_error_raises_provider_error():
    c = AgentClient("http://agent:9000")
    mock_resp = _mock_response({"detail": "not found"}, status=404)
    mock_http = MagicMock()
    mock_http.post = AsyncMock(return_value=mock_resp)

    with patch("ninetrix.client.local.get_http_client", return_value=mock_http):
        with pytest.raises(ProviderError) as exc_info:
            await c.arun("hello")

    assert "404" in str(exc_info.value)


@pytest.mark.asyncio
async def test_agent_client_500_raises_provider_error():
    c = AgentClient("http://agent:9000")
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    mock_http = MagicMock()
    mock_http.post = AsyncMock(return_value=mock_resp)

    with patch("ninetrix.client.local.get_http_client", return_value=mock_http):
        with pytest.raises(ProviderError):
            await c.arun("hello")


@pytest.mark.asyncio
async def test_agent_client_network_error_raises_provider_error():
    c = AgentClient("http://agent:9000")
    mock_http = MagicMock()
    mock_http.post = AsyncMock(side_effect=OSError("connection refused"))

    with patch("ninetrix.client.local.get_http_client", return_value=mock_http):
        with pytest.raises(ProviderError) as exc_info:
            await c.arun("hello")

    assert "network error" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_agent_client_non_json_raises_provider_error():
    c = AgentClient("http://agent:9000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.side_effect = ValueError("not json")
    mock_resp.text = "not json"
    mock_http = MagicMock()
    mock_http.post = AsyncMock(return_value=mock_resp)

    with patch("ninetrix.client.local.get_http_client", return_value=mock_http):
        with pytest.raises(ProviderError):
            await c.arun("hello")


# =============================================================================
# AgentClient — tenant context auth
# =============================================================================


@pytest.mark.asyncio
async def test_agent_client_sends_tenant_auth_header():
    from ninetrix._internals.tenant import TenantContext, tenant_scope

    c = AgentClient("http://agent:9000")
    mock_resp = _mock_response(_AGENT_RESULT_DATA)
    mock_http = MagicMock()
    mock_http.post = AsyncMock(return_value=mock_resp)

    async with tenant_scope(TenantContext(workspace_id="ws1", api_key="nxt_test")):
        with patch("ninetrix.client.local.get_http_client", return_value=mock_http):
            await c.arun("hello")

    headers = mock_http.post.call_args[1]["headers"]
    assert headers.get("Authorization") == "Bearer nxt_test"


@pytest.mark.asyncio
async def test_agent_client_no_tenant_no_auth_header():
    c = AgentClient("http://agent:9000")
    mock_resp = _mock_response(_AGENT_RESULT_DATA)
    mock_http = MagicMock()
    mock_http.post = AsyncMock(return_value=mock_resp)

    with patch("ninetrix.client.local.get_http_client", return_value=mock_http):
        await c.arun("hello")

    headers = mock_http.post.call_args[1]["headers"]
    assert "Authorization" not in headers


# =============================================================================
# AgentClient — stream stub
# =============================================================================


@pytest.mark.asyncio
async def test_agent_client_stream_raises_provider_error():
    c = AgentClient("http://agent:9000")
    with pytest.raises(ProviderError):
        async for _ in c.stream("hello"):
            pass


# =============================================================================
# AgentClient — AgentProtocol satisfaction
# =============================================================================


def test_agent_client_satisfies_agent_protocol():
    from ninetrix._internals.types import AgentProtocol
    c = AgentClient("http://agent:9000")
    assert isinstance(c, AgentProtocol)


def test_agent_client_has_name():
    c = AgentClient("http://agent:9000", name="bot")
    assert c.name == "bot"


def test_agent_client_run_and_arun_callable():
    c = AgentClient("http://agent:9000")
    assert callable(c.run)
    assert callable(c.arun)


# =============================================================================
# RemoteAgent — construction
# =============================================================================


def test_remote_agent_name_from_slug():
    r = RemoteAgent("my-workspace/analyst")
    assert r.name == "analyst"


def test_remote_agent_name_single_segment():
    r = RemoteAgent("analyst")
    assert r.name == "analyst"


def test_remote_agent_repr():
    r = RemoteAgent("ws/analyst", api_key="nxt_x")
    assert "RemoteAgent" in repr(r)
    assert "ws/analyst" in repr(r)


# =============================================================================
# RemoteAgent — arun (happy path)
# =============================================================================


@pytest.mark.asyncio
async def test_remote_agent_arun_returns_agent_result():
    r = RemoteAgent("ws/analyst", api_key="nxt_test")
    mock_resp = _mock_response(_AGENT_RESULT_DATA)
    mock_http = MagicMock()
    mock_http.post = AsyncMock(return_value=mock_resp)

    with patch("ninetrix.client.remote.get_http_client", return_value=mock_http):
        result = await r.arun("hello")

    assert isinstance(result, AgentResult)
    assert result.output == "answer"


@pytest.mark.asyncio
async def test_remote_agent_arun_posts_correct_url():
    r = RemoteAgent("ws/analyst", api_key="nxt_test")
    mock_resp = _mock_response(_AGENT_RESULT_DATA)
    mock_http = MagicMock()
    mock_http.post = AsyncMock(return_value=mock_resp)

    with patch("ninetrix.client.remote.get_http_client", return_value=mock_http):
        await r.arun("hello")

    url = mock_http.post.call_args[0][0]
    assert "/v1/agents/ws/analyst/invoke" in url


@pytest.mark.asyncio
async def test_remote_agent_sends_bearer_token():
    r = RemoteAgent("ws/analyst", api_key="nxt_test123")
    mock_resp = _mock_response(_AGENT_RESULT_DATA)
    mock_http = MagicMock()
    mock_http.post = AsyncMock(return_value=mock_resp)

    with patch("ninetrix.client.remote.get_http_client", return_value=mock_http):
        await r.arun("hello")

    headers = mock_http.post.call_args[1]["headers"]
    assert headers["Authorization"] == "Bearer nxt_test123"


# =============================================================================
# RemoteAgent — API key resolution
# =============================================================================


def test_remote_agent_no_key_raises_credential_error():
    r = RemoteAgent("ws/analyst")  # no api_key
    with patch.dict("os.environ", {}, clear=True):
        with patch("ninetrix.client.remote.NinetrixConfig.load") as mock_load:
            mock_cfg = MagicMock()
            mock_cfg.api_key = ""
            mock_load.return_value = mock_cfg
            with patch("ninetrix.client.remote.get_tenant", return_value=None):
                with pytest.raises(CredentialError):
                    r._resolve_api_key()


def test_remote_agent_uses_tenant_api_key():
    from ninetrix._internals.tenant import TenantContext
    r = RemoteAgent("ws/analyst")

    with patch("ninetrix.client.remote.get_tenant") as mock_get:
        mock_get.return_value = TenantContext(workspace_id="ws", api_key="nxt_tenant")
        key = r._resolve_api_key()

    assert key == "nxt_tenant"


def test_remote_agent_explicit_key_takes_priority():
    r = RemoteAgent("ws/analyst", api_key="nxt_explicit")
    key = r._resolve_api_key()
    assert key == "nxt_explicit"


# =============================================================================
# RemoteAgent — error handling
# =============================================================================


@pytest.mark.asyncio
async def test_remote_agent_401_raises_credential_error():
    r = RemoteAgent("ws/analyst", api_key="nxt_bad")
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized"
    mock_http = MagicMock()
    mock_http.post = AsyncMock(return_value=mock_resp)

    with patch("ninetrix.client.remote.get_http_client", return_value=mock_http):
        with pytest.raises(CredentialError):
            await r.arun("hello")


@pytest.mark.asyncio
async def test_remote_agent_500_raises_provider_error():
    r = RemoteAgent("ws/analyst", api_key="nxt_test")
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Server Error"
    mock_http = MagicMock()
    mock_http.post = AsyncMock(return_value=mock_resp)

    with patch("ninetrix.client.remote.get_http_client", return_value=mock_http):
        with pytest.raises(ProviderError):
            await r.arun("hello")


@pytest.mark.asyncio
async def test_remote_agent_network_error_raises_provider_error():
    r = RemoteAgent("ws/analyst", api_key="nxt_test")
    mock_http = MagicMock()
    mock_http.post = AsyncMock(side_effect=OSError("timeout"))

    with patch("ninetrix.client.remote.get_http_client", return_value=mock_http):
        with pytest.raises(ProviderError):
            await r.arun("hello")


# =============================================================================
# RemoteAgent — base URL override
# =============================================================================


def test_remote_agent_base_url_override(monkeypatch):
    monkeypatch.setenv("NINETRIX_CLOUD_URL", "https://staging.ninetrix.io")
    r = RemoteAgent("ws/analyst", api_key="nxt_test")
    assert r._resolve_base_url() == "https://staging.ninetrix.io"


def test_remote_agent_default_base_url():
    r = RemoteAgent("ws/analyst", api_key="nxt_test")
    import os
    os.environ.pop("NINETRIX_CLOUD_URL", None)
    assert "ninetrix.io" in r._resolve_base_url()


# =============================================================================
# RemoteAgent — stream stub
# =============================================================================


@pytest.mark.asyncio
async def test_remote_agent_stream_raises_provider_error():
    r = RemoteAgent("ws/analyst", api_key="nxt_test")
    with pytest.raises(ProviderError):
        async for _ in r.stream("hello"):
            pass


# =============================================================================
# RemoteAgent — AgentProtocol satisfaction
# =============================================================================


def test_remote_agent_satisfies_agent_protocol():
    from ninetrix._internals.types import AgentProtocol
    r = RemoteAgent("ws/analyst")
    assert isinstance(r, AgentProtocol)


def test_remote_agent_run_arun_stream_callable():
    r = RemoteAgent("ws/analyst")
    assert callable(r.run)
    assert callable(r.arun)
    assert callable(r.stream)


# =============================================================================
# Public imports
# =============================================================================


def test_agent_client_importable_from_ninetrix():
    from ninetrix import AgentClient
    assert AgentClient is not None


def test_remote_agent_importable_from_ninetrix():
    from ninetrix import RemoteAgent
    assert RemoteAgent is not None


def test_agent_client_importable_from_client():
    from ninetrix.client import AgentClient
    assert AgentClient is not None


def test_remote_agent_importable_from_client():
    from ninetrix.client import RemoteAgent
    assert RemoteAgent is not None
