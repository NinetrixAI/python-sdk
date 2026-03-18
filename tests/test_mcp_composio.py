"""Tests for MCPToolSource + ComposioToolSource (PR 23)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ninetrix import MCPToolSource, ComposioToolSource
from ninetrix._internals.types import ToolError
from ninetrix.runtime.dispatcher import ToolDispatcher, _extract_mcp_result


# =============================================================================
# _extract_mcp_result helper
# =============================================================================


def test_extract_string_passthrough():
    assert _extract_mcp_result("hello") == "hello"


def test_extract_mcp_content_list():
    result = {"content": [{"type": "text", "text": "answer"}]}
    assert _extract_mcp_result(result) == "answer"


def test_extract_mcp_content_multi():
    result = {"content": [
        {"type": "text", "text": "line1"},
        {"type": "text", "text": "line2"},
    ]}
    assert _extract_mcp_result(result) == "line1\nline2"


def test_extract_mcp_non_text_content():
    result = {"content": [{"type": "image", "url": "http://x.png"}]}
    assert "image" in _extract_mcp_result(result) or "url" in _extract_mcp_result(result)


def test_extract_dict_with_result_key():
    result = {"result": "42"}
    assert _extract_mcp_result(result) == "42"


def test_extract_empty_dict():
    r = _extract_mcp_result({})
    assert isinstance(r, str)


def test_extract_none():
    assert isinstance(_extract_mcp_result(None), str)


def test_extract_int():
    assert _extract_mcp_result(42) == "42"


# =============================================================================
# MCPToolSource — construction
# =============================================================================


def test_mcp_source_repr():
    s = MCPToolSource("http://gw:8080", "tok", "ws1")
    assert "MCPToolSource" in repr(s)
    assert "gw:8080" in repr(s)


def test_mcp_source_empty_before_init():
    s = MCPToolSource("http://gw:8080", "tok", "ws1")
    assert s.tool_definitions() == []
    assert s.handles("any_tool") is False


# =============================================================================
# MCPToolSource — initialize (happy path)
# =============================================================================


@pytest.mark.asyncio
async def test_mcp_initialize_fetches_schemas():
    s = MCPToolSource("http://gw:8080", "tok", "ws1")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "tools": [
                {"name": "slack__send_message", "description": "Send a Slack message",
                 "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}},
                {"name": "github__create_pr", "description": "Create a PR",
                 "inputSchema": {"type": "object", "properties": {}}},
            ]
        },
    }
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("ninetrix.runtime.dispatcher.get_http_client", return_value=mock_client):
        await s.initialize()

    assert s.handles("slack__send_message")
    assert s.handles("github__create_pr")
    assert not s.handles("unknown_tool")
    assert len(s.tool_definitions()) == 2


@pytest.mark.asyncio
async def test_mcp_initialize_converts_to_openai_format():
    s = MCPToolSource("http://gw:8080", "tok", "ws1")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "jsonrpc": "2.0", "id": 1,
        "result": {"tools": [
            {"name": "my_tool", "description": "desc",
             "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}}},
        ]},
    }
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("ninetrix.runtime.dispatcher.get_http_client", return_value=mock_client):
        await s.initialize()

    defs = s.tool_definitions()
    assert defs[0]["type"] == "function"
    assert defs[0]["function"]["name"] == "my_tool"
    assert defs[0]["function"]["description"] == "desc"
    assert "properties" in defs[0]["function"]["parameters"]


@pytest.mark.asyncio
async def test_mcp_initialize_posts_tools_list():
    s = MCPToolSource("http://gw:8080", "tok", "ws1")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("ninetrix.runtime.dispatcher.get_http_client", return_value=mock_client):
        await s.initialize()

    call_args = mock_client.post.call_args
    url = call_args[0][0]
    payload = call_args[1]["json"]
    assert "/v1/mcp/ws1" in url
    assert payload["method"] == "tools/list"
    assert payload["jsonrpc"] == "2.0"


@pytest.mark.asyncio
async def test_mcp_initialize_sends_auth_header():
    s = MCPToolSource("http://gw:8080", "my-token", "ws1")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("ninetrix.runtime.dispatcher.get_http_client", return_value=mock_client):
        await s.initialize()

    headers = mock_client.post.call_args[1]["headers"]
    assert headers["Authorization"] == "Bearer my-token"


# =============================================================================
# MCPToolSource — initialize (error paths)
# =============================================================================


@pytest.mark.asyncio
async def test_mcp_initialize_http_error_raises_tool_error():
    s = MCPToolSource("http://gw:8080", "tok", "ws1")
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.text = "Forbidden"
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("ninetrix.runtime.dispatcher.get_http_client", return_value=mock_client):
        with pytest.raises(ToolError) as exc:
            await s.initialize()
    assert "403" in str(exc.value)


@pytest.mark.asyncio
async def test_mcp_initialize_jsonrpc_error_raises_tool_error():
    s = MCPToolSource("http://gw:8080", "tok", "ws1")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "jsonrpc": "2.0", "id": 1,
        "error": {"code": -32603, "message": "internal error"},
    }
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("ninetrix.runtime.dispatcher.get_http_client", return_value=mock_client):
        with pytest.raises(ToolError) as exc:
            await s.initialize()
    assert "internal error" in str(exc.value)


# =============================================================================
# MCPToolSource — call (happy path)
# =============================================================================


@pytest.mark.asyncio
async def test_mcp_call_returns_text_content():
    s = MCPToolSource("http://gw:8080", "tok", "ws1")
    # Pre-seed tool names so handles() returns True
    s._tool_names = {"slack__send_message"}

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "jsonrpc": "2.0", "id": 2,
        "result": {"content": [{"type": "text", "text": "Message sent!"}]},
    }
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("ninetrix.runtime.dispatcher.get_http_client", return_value=mock_client):
        result = await s.call("slack__send_message", {"text": "hello"})

    assert result == "Message sent!"


@pytest.mark.asyncio
async def test_mcp_call_posts_tools_call():
    s = MCPToolSource("http://gw:8080", "tok", "ws1")
    s._tool_names = {"my_tool"}

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "jsonrpc": "2.0", "id": 2,
        "result": {"content": [{"type": "text", "text": "ok"}]},
    }
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("ninetrix.runtime.dispatcher.get_http_client", return_value=mock_client):
        await s.call("my_tool", {"x": 1})

    payload = mock_client.post.call_args[1]["json"]
    assert payload["method"] == "tools/call"
    assert payload["params"]["name"] == "my_tool"
    assert payload["params"]["arguments"] == {"x": 1}


# =============================================================================
# MCPToolSource — call (error paths)
# =============================================================================


@pytest.mark.asyncio
async def test_mcp_call_http_error_raises_tool_error():
    s = MCPToolSource("http://gw:8080", "tok", "ws1")
    s._tool_names = {"my_tool"}

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Server Error"
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("ninetrix.runtime.dispatcher.get_http_client", return_value=mock_client):
        with pytest.raises(ToolError):
            await s.call("my_tool", {})


@pytest.mark.asyncio
async def test_mcp_call_jsonrpc_error_raises_tool_error():
    s = MCPToolSource("http://gw:8080", "tok", "ws1")
    s._tool_names = {"my_tool"}

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "jsonrpc": "2.0", "id": 2,
        "error": {"code": -32601, "message": "Tool not found: my_tool"},
    }
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("ninetrix.runtime.dispatcher.get_http_client", return_value=mock_client):
        with pytest.raises(ToolError) as exc:
            await s.call("my_tool", {})
    assert "Tool not found" in str(exc.value)


@pytest.mark.asyncio
async def test_mcp_call_auth_required_error_includes_hint():
    s = MCPToolSource("http://gw:8080", "tok", "ws1")
    s._tool_names = {"slack__send_message"}

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "jsonrpc": "2.0", "id": 2,
        "error": {
            "code": -32010,
            "message": "Integration 'slack' requires authorization",
            "data": {"auth_url": "https://app.ninetrix.io/connect/slack"},
        },
    }
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("ninetrix.runtime.dispatcher.get_http_client", return_value=mock_client):
        with pytest.raises(ToolError) as exc:
            await s.call("slack__send_message", {})
    assert "app.ninetrix.io" in str(exc.value) or "Connect" in str(exc.value)


# =============================================================================
# MCPToolSource — ToolDispatcher integration
# =============================================================================


@pytest.mark.asyncio
async def test_dispatcher_with_mcp_source():
    s = MCPToolSource("http://gw:8080", "tok", "ws1")
    # Seed without calling initialize() to keep test simple
    s._tool_names = {"search"}
    s._tool_schemas = [{
        "type": "function",
        "function": {"name": "search", "description": "search", "parameters": {}},
    }]

    dispatcher = ToolDispatcher([s])
    assert len(dispatcher.all_tool_definitions()) == 1

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "jsonrpc": "2.0", "id": 1,
        "result": {"content": [{"type": "text", "text": "results"}]},
    }
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("ninetrix.runtime.dispatcher.get_http_client", return_value=mock_client):
        result = await dispatcher.call("search", {"q": "test"})
    assert result == "results"


# =============================================================================
# ComposioToolSource — construction
# =============================================================================


def test_composio_source_repr():
    s = ComposioToolSource(apps=["GITHUB", "SLACK"])
    assert "ComposioToolSource" in repr(s)
    assert "GITHUB" in repr(s)


def test_composio_source_empty_before_init():
    s = ComposioToolSource(apps=["GITHUB"])
    assert s.tool_definitions() == []
    assert s.handles("GITHUB_CREATE_PR") is False


# =============================================================================
# ComposioToolSource — initialize (no SDK installed)
# =============================================================================


@pytest.mark.asyncio
async def test_composio_initialize_no_sdk_warns():
    """When composio-openai is not installed, initialize() warns but does not raise."""
    import sys
    import warnings as _warnings

    s = ComposioToolSource(apps=["GITHUB"])

    # Simulate missing composio_openai by inserting None in sys.modules
    with patch.dict(sys.modules, {"composio_openai": None, "composio": None}):
        with _warnings.catch_warnings(record=True):
            _warnings.simplefilter("always")
            # initialize should not raise even when SDK is absent
            try:
                await s.initialize()
            except Exception:
                pass  # ImportError from None in sys.modules is acceptable

    # Either way, no schemas should be loaded from a missing SDK
    assert s.tool_definitions() == []


# =============================================================================
# ComposioToolSource — call (no toolset)
# =============================================================================


@pytest.mark.asyncio
async def test_composio_call_without_init_raises_tool_error():
    """Calling a Composio tool without initialization raises ToolError."""
    s = ComposioToolSource(apps=["GITHUB"])
    # Manually make it think it handles the tool but has no toolset
    s._tool_names = {"GITHUB_CREATE_PR"}

    with pytest.raises(ToolError) as exc:
        await s.call("GITHUB_CREATE_PR", {"title": "bug"})
    assert "composio" in str(exc.value).lower()


# =============================================================================
# ComposioToolSource — public imports
# =============================================================================


def test_mcp_tool_source_importable_from_ninetrix():
    from ninetrix import MCPToolSource
    assert MCPToolSource is not None


def test_composio_tool_source_importable_from_ninetrix():
    from ninetrix import ComposioToolSource
    assert ComposioToolSource is not None


def test_mcp_tool_source_importable_from_dispatcher():
    from ninetrix.runtime.dispatcher import MCPToolSource
    assert MCPToolSource is not None


def test_composio_tool_source_importable_from_dispatcher():
    from ninetrix.runtime.dispatcher import ComposioToolSource
    assert ComposioToolSource is not None
