"""Tests for OpenAPIToolSource — turns OpenAPI specs into LLM tools."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ninetrix.tools.sources.openapi import OpenAPIToolSource
from ninetrix.tools.agent_context import AgentContext
from ninetrix.tools.auth_resolver import AuthResolver
from ninetrix._internals.types import ConfigurationError


# ── Test specs ────────────────────────────────────────────────────────────

MINIMAL_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Test", "version": "1.0"},
    "servers": [{"url": "https://api.example.com"}],
    "paths": {
        "/users/{id}": {
            "get": {
                "operationId": "getUser",
                "summary": "Get a user by ID",
                "parameters": [
                    {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
                ],
            },
        },
        "/users": {
            "get": {
                "operationId": "listUsers",
                "summary": "List all users",
                "parameters": [
                    {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 10}},
                    {"name": "status", "in": "query", "schema": {"type": "string", "enum": ["active", "inactive"]}},
                ],
            },
            "post": {
                "operationId": "createUser",
                "summary": "Create a new user",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "User name"},
                                    "email": {"type": "string", "description": "Email address"},
                                },
                                "required": ["name", "email"],
                            }
                        }
                    }
                },
            },
        },
    },
}


# ── Construction + validation ─────────────────────────────────────────────


def test_validate_config_empty_spec():
    source = OpenAPIToolSource(spec="")
    with pytest.raises(ConfigurationError, match="requires a spec"):
        source.validate_config()


def test_validate_config_ok():
    source = OpenAPIToolSource(spec=MINIMAL_SPEC)
    source.validate_config()  # should not raise


# ── initialize from dict ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_initialize_from_dict():
    source = OpenAPIToolSource(spec=MINIMAL_SPEC)
    await source.initialize()

    assert len(source.tool_definitions()) == 3
    names = {t["function"]["name"] for t in source.tool_definitions()}
    assert names == {"getUser", "listUsers", "createUser"}


@pytest.mark.asyncio
async def test_initialize_with_name_prefix():
    source = OpenAPIToolSource(spec=MINIMAL_SPEC, name_prefix="test__")
    await source.initialize()

    names = {t["function"]["name"] for t in source.tool_definitions()}
    assert "test__getUser" in names
    assert "test__listUsers" in names


@pytest.mark.asyncio
async def test_initialize_with_action_filter():
    source = OpenAPIToolSource(spec=MINIMAL_SPEC, actions=["getUser"])
    await source.initialize()

    assert len(source.tool_definitions()) == 1
    assert source.tool_definitions()[0]["function"]["name"] == "getUser"


@pytest.mark.asyncio
async def test_handles():
    source = OpenAPIToolSource(spec=MINIMAL_SPEC)
    await source.initialize()

    assert source.handles("getUser") is True
    assert source.handles("nonexistent") is False


# ── Tool definitions ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_definitions_schema():
    source = OpenAPIToolSource(spec=MINIMAL_SPEC, actions=["getUser"])
    await source.initialize()

    td = source.tool_definitions()[0]
    assert td["type"] == "function"
    assert td["function"]["name"] == "getUser"
    assert td["function"]["description"] == "Get a user by ID"
    params = td["function"]["parameters"]
    assert params["type"] == "object"
    assert "id" in params["properties"]
    assert "id" in params.get("required", [])


@pytest.mark.asyncio
async def test_tool_definitions_query_params():
    source = OpenAPIToolSource(spec=MINIMAL_SPEC, actions=["listUsers"])
    await source.initialize()

    td = source.tool_definitions()[0]
    params = td["function"]["parameters"]
    assert "limit" in params["properties"]
    assert params["properties"]["limit"]["type"] == "integer"
    assert params["properties"]["status"]["enum"] == ["active", "inactive"]


@pytest.mark.asyncio
async def test_tool_definitions_request_body():
    source = OpenAPIToolSource(spec=MINIMAL_SPEC, actions=["createUser"])
    await source.initialize()

    td = source.tool_definitions()[0]
    params = td["function"]["parameters"]
    assert "name" in params["properties"]
    assert "email" in params["properties"]
    assert "name" in params.get("required", [])


# ── Auto-generated operation IDs ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_auto_generated_operation_id():
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Test", "version": "1.0"},
        "servers": [{"url": "https://api.example.com"}],
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health check",
                    # No operationId — should auto-generate
                }
            }
        },
    }
    source = OpenAPIToolSource(spec=spec)
    await source.initialize()

    names = {t["function"]["name"] for t in source.tool_definitions()}
    assert "get_health" in names


# ── call() ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_get_with_path_param():
    source = OpenAPIToolSource(spec=MINIMAL_SPEC, actions=["getUser"])
    await source.initialize()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "application/json"}
    mock_resp.json.return_value = {"id": "123", "name": "Alice"}

    mock_http = AsyncMock()
    mock_http.request = AsyncMock(return_value=mock_resp)

    with patch.object(source, "_get_http", return_value=mock_http):
        result = await source.call("getUser", {"id": "123"})

    assert "Alice" in result
    mock_http.request.assert_called_once()
    call_kwargs = mock_http.request.call_args
    assert call_kwargs.kwargs["method"] == "GET"
    assert "/users/123" in call_kwargs.kwargs["url"]


@pytest.mark.asyncio
async def test_call_get_with_query_params():
    source = OpenAPIToolSource(spec=MINIMAL_SPEC, actions=["listUsers"])
    await source.initialize()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "application/json"}
    mock_resp.json.return_value = [{"id": "1"}]

    mock_http = AsyncMock()
    mock_http.request = AsyncMock(return_value=mock_resp)

    with patch.object(source, "_get_http", return_value=mock_http):
        result = await source.call("listUsers", {"limit": 5, "status": "active"})

    call_kwargs = mock_http.request.call_args
    assert call_kwargs.kwargs["params"] == {"limit": 5, "status": "active"}


@pytest.mark.asyncio
async def test_call_post_with_body():
    source = OpenAPIToolSource(spec=MINIMAL_SPEC, actions=["createUser"])
    await source.initialize()

    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.headers = {"content-type": "application/json"}
    mock_resp.json.return_value = {"id": "new-1", "name": "Bob"}

    mock_http = AsyncMock()
    mock_http.request = AsyncMock(return_value=mock_resp)

    with patch.object(source, "_get_http", return_value=mock_http):
        result = await source.call("createUser", {"name": "Bob", "email": "bob@example.com"})

    call_kwargs = mock_http.request.call_args
    assert call_kwargs.kwargs["method"] == "POST"
    assert call_kwargs.kwargs["json"] == {"name": "Bob", "email": "bob@example.com"}


@pytest.mark.asyncio
async def test_call_with_auth():
    auth_config = {"type": "bearer", "token": "sk-test"}
    source = OpenAPIToolSource(
        spec=MINIMAL_SPEC,
        actions=["listUsers"],
        auth=auth_config,
    )
    await source.initialize()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "application/json"}
    mock_resp.json.return_value = []

    mock_http = AsyncMock()
    mock_http.request = AsyncMock(return_value=mock_resp)

    with patch.object(source, "_get_http", return_value=mock_http):
        await source.call("listUsers", {})

    call_kwargs = mock_http.request.call_args
    assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer sk-test"


@pytest.mark.asyncio
async def test_call_http_error_returns_error_text():
    source = OpenAPIToolSource(spec=MINIMAL_SPEC, actions=["getUser"])
    await source.initialize()

    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.text = "Not Found"

    mock_http = AsyncMock()
    mock_http.request = AsyncMock(return_value=mock_resp)

    with patch.object(source, "_get_http", return_value=mock_http):
        result = await source.call("getUser", {"id": "999"})

    assert "404" in result
    assert "Not Found" in result


@pytest.mark.asyncio
async def test_call_unknown_tool_raises():
    source = OpenAPIToolSource(spec=MINIMAL_SPEC)
    await source.initialize()

    from ninetrix._internals.types import ToolError
    with pytest.raises(ToolError, match="not found"):
        await source.call("nonexistent", {})


# ── Base URL override ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_base_url_override():
    source = OpenAPIToolSource(
        spec=MINIMAL_SPEC,
        base_url="https://custom.api.com",
        actions=["getUser"],
    )
    await source.initialize()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "text/plain"}
    mock_resp.text = "ok"

    mock_http = AsyncMock()
    mock_http.request = AsyncMock(return_value=mock_resp)

    with patch.object(source, "_get_http", return_value=mock_http):
        await source.call("getUser", {"id": "1"})

    call_kwargs = mock_http.request.call_args
    assert call_kwargs.kwargs["url"].startswith("https://custom.api.com/")


# ── Load from file ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_from_json_file(tmp_path):
    spec_file = tmp_path / "api.json"
    spec_file.write_text(json.dumps(MINIMAL_SPEC))

    source = OpenAPIToolSource(spec=str(spec_file))
    await source.initialize()

    assert len(source.tool_definitions()) == 3


@pytest.mark.asyncio
async def test_load_from_missing_file():
    source = OpenAPIToolSource(spec="/nonexistent/path.json")

    with pytest.raises(ConfigurationError, match="not found"):
        await source.initialize()


# ── source_type ───────────────────────────────────────────────────────────


def test_source_type():
    assert OpenAPIToolSource.source_type == "openapi"
