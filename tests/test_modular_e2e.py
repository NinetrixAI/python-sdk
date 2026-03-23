"""
End-to-end tests for the modular tool provider system.

Tests the full flow: register sources → build dispatcher → call tools.
Covers: LocalToolSource, OpenAPIToolSource, discovery, AgentContext, AuthResolver,
ToolDispatcher lifecycle (validate → initialize → call → shutdown).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ninetrix import Tool, ToolDef, _registry
from ninetrix.tools.base import ToolSource
from ninetrix.tools.agent_context import AgentContext
from ninetrix.tools.auth_resolver import AuthResolver
from ninetrix.tools.sources.local import LocalToolSource
from ninetrix.tools.sources.openapi import OpenAPIToolSource
from ninetrix.tools.discovery import (
    auto_register_builtins,
    discover_and_register_plugins,
    get_source_class,
    register_source,
    reset_registry,
)
from ninetrix.runtime.dispatcher import ToolDispatcher


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_state():
    _registry.clear()
    reset_registry()
    yield
    _registry.clear()
    reset_registry()


# ── Helpers ───────────────────────────────────────────────────────────────

PETSTORE_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Petstore", "version": "1.0"},
    "servers": [{"url": "https://petstore.example.com/v1"}],
    "paths": {
        "/pets": {
            "get": {
                "operationId": "listPets",
                "summary": "List all pets",
                "parameters": [
                    {"name": "limit", "in": "query", "schema": {"type": "integer"}},
                    {"name": "species", "in": "query", "schema": {"type": "string", "enum": ["dog", "cat", "bird"]}},
                ],
            },
            "post": {
                "operationId": "createPet",
                "summary": "Create a pet",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "species": {"type": "string"},
                                    "age": {"type": "integer"},
                                },
                                "required": ["name", "species"],
                            }
                        }
                    }
                },
            },
        },
        "/pets/{petId}": {
            "get": {
                "operationId": "getPet",
                "summary": "Get a pet by ID",
                "parameters": [
                    {"name": "petId", "in": "path", "required": True, "schema": {"type": "string"}},
                ],
            },
            "delete": {
                "operationId": "deletePet",
                "summary": "Delete a pet",
                "parameters": [
                    {"name": "petId", "in": "path", "required": True, "schema": {"type": "string"}},
                ],
            },
        },
    },
}


def _mock_http_response(status=200, json_data=None, text="", content_type="application/json"):
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"content-type": content_type}
    resp.text = text or json.dumps(json_data or {})
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


# ══════════════════════════════════════════════════════════════════════════
# Test 1: Local + OpenAPI sources in one dispatcher
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_mixed_local_and_openapi_sources():
    """Real scenario: agent has local Python tools AND an OpenAPI integration."""

    # 1. Define a local @Tool
    @Tool
    def calculate_age_in_dog_years(human_age: int) -> str:
        """Convert human age to dog years."""
        return f"{human_age * 7} dog years"

    # 2. Create OpenAPI source (Petstore API)
    mock_http = AsyncMock()
    ctx = AgentContext(
        http=mock_http,
        auth=AuthResolver(env=lambda k, d="": {"PET_TOKEN": "bearer-123"}.get(k, d)),
        agent_name="pet-agent",
    )

    openapi_source = OpenAPIToolSource(
        spec=PETSTORE_SPEC,
        name_prefix="petstore__",
        auth={"type": "bearer", "token": "${PET_TOKEN}"},
        ctx=ctx,
    )

    # 3. Build local source
    td = _registry.get("calculate_age_in_dog_years")
    local_source = LocalToolSource([td])

    # 4. Wire up dispatcher
    dispatcher = ToolDispatcher([local_source, openapi_source])
    await dispatcher.initialize()

    # 5. Verify all tools are visible
    all_tools = dispatcher.all_tool_definitions()
    names = {t["function"]["name"] for t in all_tools}
    assert "calculate_age_in_dog_years" in names
    assert "petstore__listPets" in names
    assert "petstore__createPet" in names
    assert "petstore__getPet" in names
    assert "petstore__deletePet" in names
    assert len(names) == 5

    # 6. Call the local tool
    result = await dispatcher.call("calculate_age_in_dog_years", {"human_age": 3})
    assert result == "21 dog years"

    # 7. Call the OpenAPI tool
    mock_http.request = AsyncMock(return_value=_mock_http_response(
        json_data=[{"id": "1", "name": "Buddy", "species": "dog"}]
    ))

    result = await dispatcher.call("petstore__listPets", {"limit": 10, "species": "dog"})
    parsed = json.loads(result)
    assert parsed[0]["name"] == "Buddy"

    # Verify HTTP call was made correctly
    call_kwargs = mock_http.request.call_args.kwargs
    assert call_kwargs["method"] == "GET"
    assert "petstore.example.com/v1/pets" in call_kwargs["url"]
    assert call_kwargs["params"] == {"limit": 10, "species": "dog"}
    assert call_kwargs["headers"]["Authorization"] == "Bearer bearer-123"

    # 8. Shutdown
    await dispatcher.shutdown()


# ══════════════════════════════════════════════════════════════════════════
# Test 2: OpenAPI POST with body + path params
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_openapi_create_and_get():
    """Test POST (requestBody) and GET (path param) against Petstore."""

    mock_http = AsyncMock()
    ctx = AgentContext(http=mock_http, auth=AuthResolver())

    source = OpenAPIToolSource(
        spec=PETSTORE_SPEC,
        name_prefix="pets__",
        ctx=ctx,
    )
    await source.initialize()

    # POST /pets — create
    mock_http.request = AsyncMock(return_value=_mock_http_response(
        status=201,
        json_data={"id": "42", "name": "Rex", "species": "dog", "age": 3},
    ))

    result = await source.call("pets__createPet", {"name": "Rex", "species": "dog", "age": 3})
    parsed = json.loads(result)
    assert parsed["id"] == "42"

    call_kwargs = mock_http.request.call_args.kwargs
    assert call_kwargs["method"] == "POST"
    assert call_kwargs["json"] == {"name": "Rex", "species": "dog", "age": 3}

    # GET /pets/{petId} — fetch
    mock_http.request = AsyncMock(return_value=_mock_http_response(
        json_data={"id": "42", "name": "Rex", "species": "dog"},
    ))

    result = await source.call("pets__getPet", {"petId": "42"})
    parsed = json.loads(result)
    assert parsed["name"] == "Rex"

    call_kwargs = mock_http.request.call_args.kwargs
    assert call_kwargs["method"] == "GET"
    assert "/pets/42" in call_kwargs["url"]


# ══════════════════════════════════════════════════════════════════════════
# Test 3: Discovery system registers built-ins + community plugins
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_discovery_registers_and_resolves():
    """Test that auto_register_builtins + plugin discovery works."""

    # 1. Register builtins
    auto_register_builtins()

    assert get_source_class("mcp") is not None
    assert get_source_class("openapi") is not None
    assert get_source_class("composio") is not None

    # 2. Simulate a community plugin
    class JiraToolSource(ToolSource):
        source_type = "jira"

        def __init__(self, instance_url: str, **kwargs):
            self._url = instance_url
            self._tools = {}

        def tool_definitions(self):
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "jira__create_issue",
                        "description": "Create a Jira issue",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "project": {"type": "string"},
                                "summary": {"type": "string"},
                                "description": {"type": "string"},
                            },
                            "required": ["project", "summary"],
                        },
                    },
                }
            ]

        def handles(self, name):
            return name.startswith("jira__")

        async def call(self, name, arguments):
            return json.dumps({"key": "PROJ-123", "summary": arguments["summary"]})

    # 3. Register it
    register_source("jira", JiraToolSource)
    assert get_source_class("jira") is JiraToolSource

    # 4. Use it in a dispatcher alongside local tools
    @Tool
    def format_ticket(key: str, summary: str) -> str:
        """Format a ticket for display."""
        return f"[{key}] {summary}"

    local_source = LocalToolSource([_registry.get("format_ticket")])
    jira_source = JiraToolSource(instance_url="https://mycompany.atlassian.net")

    dispatcher = ToolDispatcher([local_source, jira_source])
    await dispatcher.initialize()

    # Verify both sources contribute tools
    all_tools = dispatcher.all_tool_definitions()
    names = {t["function"]["name"] for t in all_tools}
    assert "format_ticket" in names
    assert "jira__create_issue" in names

    # Call the community plugin
    result = await dispatcher.call("jira__create_issue", {
        "project": "PROJ",
        "summary": "Fix login bug",
    })
    parsed = json.loads(result)
    assert parsed["key"] == "PROJ-123"

    # Call the local tool
    result = await dispatcher.call("format_ticket", {"key": "PROJ-123", "summary": "Fix login bug"})
    assert result == "[PROJ-123] Fix login bug"


# ══════════════════════════════════════════════════════════════════════════
# Test 4: AuthResolver with different auth types
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_openapi_with_different_auth_types():
    """Test that different auth types produce correct HTTP headers."""

    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Test", "version": "1.0"},
        "servers": [{"url": "https://api.test.com"}],
        "paths": {
            "/data": {
                "get": {
                    "operationId": "getData",
                    "summary": "Get data",
                }
            }
        },
    }

    fake_env = lambda k, d="": {
        "MY_TOKEN": "tok-secret",
        "MY_USER": "admin",
        "MY_PASS": "p@ss",
        "MY_KEY": "key-xyz",
    }.get(k, d)

    # Bearer auth
    mock_http = AsyncMock()
    mock_http.request = AsyncMock(return_value=_mock_http_response(json_data={"ok": True}))
    ctx = AgentContext(http=mock_http, auth=AuthResolver(env=fake_env))

    source = OpenAPIToolSource(spec=spec, auth={"type": "bearer", "token": "${MY_TOKEN}"}, ctx=ctx)
    await source.initialize()
    await source.call("getData", {})
    assert mock_http.request.call_args.kwargs["headers"]["Authorization"] == "Bearer tok-secret"

    # Basic auth
    mock_http.request.reset_mock()
    source2 = OpenAPIToolSource(
        spec=spec,
        auth={"type": "basic", "username": "${MY_USER}", "password": "${MY_PASS}"},
        ctx=ctx,
    )
    await source2.initialize()
    await source2.call("getData", {})
    auth_header = mock_http.request.call_args.kwargs["headers"]["Authorization"]
    assert auth_header.startswith("Basic ")

    # Custom header
    mock_http.request.reset_mock()
    source3 = OpenAPIToolSource(
        spec=spec,
        auth={"type": "header", "header_name": "X-API-Key", "token": "${MY_KEY}"},
        ctx=ctx,
    )
    await source3.initialize()
    await source3.call("getData", {})
    assert mock_http.request.call_args.kwargs["headers"]["X-API-Key"] == "key-xyz"


# ══════════════════════════════════════════════════════════════════════════
# Test 5: Dispatcher lifecycle (validate → init → health → shutdown)
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_full_dispatcher_lifecycle():
    """Test the complete lifecycle: validate → initialize → health_check → call → shutdown."""

    lifecycle_log = []

    class TrackedSource(ToolSource):
        source_type = "tracked"

        def __init__(self, name):
            self._name = name

        def validate_config(self):
            lifecycle_log.append(f"validate:{self._name}")

        async def initialize(self):
            lifecycle_log.append(f"init:{self._name}")

        def tool_definitions(self):
            return [{
                "type": "function",
                "function": {
                    "name": f"{self._name}_tool",
                    "description": f"Tool from {self._name}",
                    "parameters": {"type": "object", "properties": {}},
                },
            }]

        def handles(self, name):
            return name == f"{self._name}_tool"

        async def call(self, name, args):
            lifecycle_log.append(f"call:{self._name}")
            return f"result from {self._name}"

        async def health_check(self):
            lifecycle_log.append(f"health:{self._name}")
            return True

        async def shutdown(self):
            lifecycle_log.append(f"shutdown:{self._name}")

    # Build dispatcher with two tracked sources
    d = ToolDispatcher([TrackedSource("alpha"), TrackedSource("beta")])

    # Initialize (validate all, then init all)
    await d.initialize()
    assert lifecycle_log == [
        "validate:alpha", "validate:beta",
        "init:alpha", "init:beta",
    ]

    # Health check
    lifecycle_log.clear()
    health = await d.health_check()
    assert health == {"TrackedSource:tracked": True, "TrackedSource:tracked": True}
    assert "health:alpha" in lifecycle_log

    # Call
    lifecycle_log.clear()
    result = await d.call("alpha_tool", {})
    assert result == "result from alpha"
    assert lifecycle_log == ["call:alpha"]

    result = await d.call("beta_tool", {})
    assert result == "result from beta"

    # Shutdown (reverse order)
    lifecycle_log.clear()
    await d.shutdown()
    assert lifecycle_log == ["shutdown:beta", "shutdown:alpha"]


# ══════════════════════════════════════════════════════════════════════════
# Test 6: OpenAPI with action filter + base_url override
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_openapi_action_filter_and_base_url():
    """Only expose specific operations, override base URL."""

    mock_http = AsyncMock()
    mock_http.request = AsyncMock(return_value=_mock_http_response(json_data=[]))
    ctx = AgentContext(http=mock_http, auth=AuthResolver())

    source = OpenAPIToolSource(
        spec=PETSTORE_SPEC,
        actions=["listPets", "getPet"],  # Only these two
        base_url="https://staging.petstore.com/v2",  # Override
        ctx=ctx,
    )
    await source.initialize()

    # Only 2 tools should be registered
    tools = source.tool_definitions()
    names = {t["function"]["name"] for t in tools}
    assert names == {"listPets", "getPet"}
    assert "createPet" not in names
    assert "deletePet" not in names

    # Call should use overridden base URL
    await source.call("listPets", {})
    url = mock_http.request.call_args.kwargs["url"]
    assert url.startswith("https://staging.petstore.com/v2/")


# ══════════════════════════════════════════════════════════════════════════
# Test 7: Load spec from JSON file
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_openapi_from_file(tmp_path):
    """Load spec from a local JSON file — simulates ./specs/api.json in agentfile.yaml."""

    spec_file = tmp_path / "petstore.json"
    spec_file.write_text(json.dumps(PETSTORE_SPEC))

    source = OpenAPIToolSource(spec=str(spec_file), name_prefix="ps__")
    await source.initialize()

    names = {t["function"]["name"] for t in source.tool_definitions()}
    assert "ps__listPets" in names
    assert len(names) == 4
