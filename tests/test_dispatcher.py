"""Tests for tools/context.py and runtime/dispatcher.py — PR 15."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ninetrix.tools.context import ToolContext, _is_context_param
from ninetrix.runtime.dispatcher import (
    LocalToolSource,
    RegistryToolSource,
    ToolDispatcher,
    ToolSource,
    _find_context_param,
)
from ninetrix.registry import ToolDef
from ninetrix._internals.types import ToolError


# =============================================================================
# ToolContext
# =============================================================================


def test_tool_context_defaults():
    ctx = ToolContext()
    assert ctx.thread_id == ""
    assert ctx.agent_name == ""
    assert ctx.turn_index == 0


def test_tool_context_set_fields():
    ctx = ToolContext(thread_id="t1", agent_name="bot", turn_index=3)
    assert ctx.thread_id == "t1"
    assert ctx.agent_name == "bot"
    assert ctx.turn_index == 3


def test_tool_context_get_found():
    ctx = ToolContext(_store={"db": "my_db"})
    assert ctx.get("db") == "my_db"


def test_tool_context_get_default():
    ctx = ToolContext()
    assert ctx.get("missing") is None
    assert ctx.get("missing", "fallback") == "fallback"


def test_tool_context_getitem_found():
    ctx = ToolContext(_store={"key": 42})
    assert ctx["key"] == 42


def test_tool_context_getitem_missing():
    ctx = ToolContext()
    with pytest.raises(KeyError, match="ToolContext has no resource 'key'"):
        _ = ctx["key"]


def test_tool_context_contains():
    ctx = ToolContext(_store={"x": 1})
    assert "x" in ctx
    assert "y" not in ctx


def test_tool_context_is_tool_context_marker():
    ctx = ToolContext()
    assert ctx._is_tool_context is True


def test_is_context_param_with_class():
    assert _is_context_param(ToolContext) is True


def test_is_context_param_with_instance_annotation():
    # Annotation typed as the class itself
    assert _is_context_param(ToolContext) is True


def test_is_context_param_with_str():
    assert _is_context_param("not_a_context") is False


def test_is_context_param_with_int():
    assert _is_context_param(int) is False


# =============================================================================
# _find_context_param helper
# =============================================================================


def test_find_context_param_present():
    def fn(sql: str, ctx: ToolContext) -> str:
        return sql

    assert _find_context_param(fn) == "ctx"


def test_find_context_param_absent():
    def fn(x: str, y: int) -> str:
        return x

    assert _find_context_param(fn) is None


def test_find_context_param_different_name():
    def fn(query: str, context: ToolContext) -> str:
        return query

    assert _find_context_param(fn) == "context"


def test_find_context_param_no_annotations():
    def fn(x, y):
        return x

    assert _find_context_param(fn) is None


# =============================================================================
# LocalToolSource — tool_definitions
# =============================================================================


def _make_tool_def(name: str, fn: Any, desc: str = "desc") -> ToolDef:
    return ToolDef(
        name=name,
        description=desc,
        parameters={"type": "object", "properties": {"x": {"type": "string"}}},
        fn=fn,
    )


def test_local_tool_definitions():
    def my_tool(x: str) -> str:
        return x

    source = LocalToolSource([_make_tool_def("my_tool", my_tool)])
    defs = source.tool_definitions()
    assert len(defs) == 1
    assert defs[0]["type"] == "function"
    assert defs[0]["function"]["name"] == "my_tool"


def test_local_handles_true():
    def t(x: str) -> str:
        return x

    source = LocalToolSource([_make_tool_def("t", t)])
    assert source.handles("t") is True


def test_local_handles_false():
    source = LocalToolSource([])
    assert source.handles("unknown") is False


# =============================================================================
# LocalToolSource — call
# =============================================================================


@pytest.mark.asyncio
async def test_local_call_sync_tool():
    def add(a: int, b: int) -> int:
        return a + b

    td = ToolDef(
        name="add",
        description="add two numbers",
        parameters={"type": "object", "properties": {}},
        fn=add,
    )
    source = LocalToolSource([td])
    result = await source.call("add", {"a": 2, "b": 3})
    assert result == "5"


@pytest.mark.asyncio
async def test_local_call_async_tool():
    async def async_add(a: int, b: int) -> int:
        return a + b

    td = ToolDef(
        name="async_add",
        description="add async",
        parameters={"type": "object", "properties": {}},
        fn=async_add,
    )
    source = LocalToolSource([td])
    result = await source.call("async_add", {"a": 10, "b": 5})
    assert result == "15"


@pytest.mark.asyncio
async def test_local_call_returns_none_gives_done():
    def no_return() -> None:
        pass

    td = ToolDef(
        name="no_return",
        description="does nothing",
        parameters={"type": "object", "properties": {}},
        fn=no_return,
    )
    source = LocalToolSource([td])
    result = await source.call("no_return", {})
    assert result == "(done)"


@pytest.mark.asyncio
async def test_local_call_injects_tool_context():
    captured: list[ToolContext] = []

    def fn_with_ctx(query: str, ctx: ToolContext) -> str:
        captured.append(ctx)
        return f"result:{query}"

    td = ToolDef(
        name="ctx_tool",
        description="test ctx injection",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        fn=fn_with_ctx,
    )
    source = LocalToolSource([td], tool_context={"db": "test_db"})
    result = await source.call(
        "ctx_tool",
        {"query": "hello"},
        thread_id="t99",
        agent_name="bot",
        turn_index=2,
    )

    assert result == "result:hello"
    assert len(captured) == 1
    ctx = captured[0]
    assert ctx.thread_id == "t99"
    assert ctx.agent_name == "bot"
    assert ctx.turn_index == 2
    assert ctx.get("db") == "test_db"


@pytest.mark.asyncio
async def test_local_call_merges_per_request_context():
    captured: list[ToolContext] = []

    def fn(x: str, ctx: ToolContext) -> str:
        captured.append(ctx)
        return x

    td = ToolDef(
        name="fn",
        description="",
        parameters={"type": "object", "properties": {}},
        fn=fn,
    )
    source = LocalToolSource([td], tool_context={"static": "A"})
    await source.call("fn", {"x": "v"}, extra_context={"dynamic": "B"})

    ctx = captured[0]
    assert ctx.get("static") == "A"
    assert ctx.get("dynamic") == "B"


@pytest.mark.asyncio
async def test_local_call_unknown_tool_raises():
    source = LocalToolSource([])
    with pytest.raises(ToolError, match="not registered"):
        await source.call("ghost", {})


@pytest.mark.asyncio
async def test_local_call_exception_wraps_as_tool_error():
    def bad() -> str:
        raise ValueError("boom")

    td = ToolDef(name="bad", description="", parameters={"type": "object", "properties": {}}, fn=bad)
    source = LocalToolSource([td])
    with pytest.raises(ToolError, match="boom"):
        await source.call("bad", {})


# =============================================================================
# ToolContext excluded from schema via schema.py
# =============================================================================


def test_tool_context_excluded_from_schema():
    """Verify @Tool with ctx: ToolContext excludes ctx from JSON schema."""
    from ninetrix import Tool, _registry

    _registry.clear()

    @Tool
    def search(query: str, ctx: ToolContext) -> str:
        """Search the web."""
        return "result"

    td = _registry.get("search")
    assert td is not None
    props = td.parameters.get("properties", {})
    assert "query" in props
    assert "ctx" not in props
    assert "context" not in props

    _registry.clear()


# =============================================================================
# ToolDispatcher
# =============================================================================


@pytest.mark.asyncio
async def test_dispatcher_aggregates_schemas():
    def t1(x: str) -> str:
        return x

    def t2(y: str) -> str:
        return y

    s1 = LocalToolSource([_make_tool_def("t1", t1)])
    s2 = LocalToolSource([_make_tool_def("t2", t2)])
    dispatcher = ToolDispatcher([s1, s2])
    defs = dispatcher.all_tool_definitions()
    names = [d["function"]["name"] for d in defs]
    assert "t1" in names
    assert "t2" in names


@pytest.mark.asyncio
async def test_dispatcher_call_routes_correctly():
    def tool_a(x: str) -> str:
        return f"A:{x}"

    def tool_b(y: str) -> str:
        return f"B:{y}"

    dispatcher = ToolDispatcher([
        LocalToolSource([_make_tool_def("tool_a", tool_a)]),
        LocalToolSource([_make_tool_def("tool_b", tool_b)]),
    ])
    r1 = await dispatcher.call("tool_a", {"x": "hi"})
    r2 = await dispatcher.call("tool_b", {"y": "bye"})
    assert r1 == "A:hi"
    assert r2 == "B:bye"


@pytest.mark.asyncio
async def test_dispatcher_unknown_tool_returns_error_string():
    dispatcher = ToolDispatcher([])
    result = await dispatcher.call("ghost", {})
    assert "not available" in result


@pytest.mark.asyncio
async def test_dispatcher_initialize_calls_sources():
    mock_source = MagicMock(spec=ToolSource)
    mock_source.initialize = AsyncMock()
    dispatcher = ToolDispatcher([mock_source])
    await dispatcher.initialize()
    mock_source.initialize.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatcher_initialize_skips_sources_without_initialize():
    # LocalToolSource has no initialize() — should not raise
    def t(x: str) -> str:
        return x

    source = LocalToolSource([_make_tool_def("t", t)])
    dispatcher = ToolDispatcher([source])
    await dispatcher.initialize()  # should complete without error


def test_dispatcher_repr():
    source = LocalToolSource([])
    dispatcher = ToolDispatcher([source])
    r = repr(dispatcher)
    assert "LocalToolSource" in r


# =============================================================================
# RegistryToolSource
# =============================================================================


@pytest.mark.asyncio
async def test_registry_source_initialize():
    source = RegistryToolSource(
        skills=["web_search"],
        registry_url="https://registry.example.com",
        api_key="test_key",
    )

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "schemas": [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_ctx_manager = AsyncMock()
    mock_ctx_manager.__aenter__ = AsyncMock(return_value=mock_client)
    mock_ctx_manager.__aexit__ = AsyncMock(return_value=None)

    with patch("ninetrix.runtime.dispatcher.get_http_client", return_value=mock_ctx_manager):
        await source.initialize()

    assert source.handles("web_search") is True
    assert len(source.tool_definitions()) == 1


@pytest.mark.asyncio
async def test_registry_source_handles_false_before_init():
    source = RegistryToolSource(
        skills=["web_search"],
        registry_url="https://registry.example.com",
        api_key="key",
    )
    assert source.handles("web_search") is False


@pytest.mark.asyncio
async def test_registry_source_call():
    source = RegistryToolSource(
        skills=["web_search"],
        registry_url="https://registry.example.com",
        api_key="key",
    )
    source._skill_names = {"web_search"}

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"result": "Paris is the capital of France"}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_ctx_manager = AsyncMock()
    mock_ctx_manager.__aenter__ = AsyncMock(return_value=mock_client)
    mock_ctx_manager.__aexit__ = AsyncMock(return_value=None)

    with patch("ninetrix.runtime.dispatcher.get_http_client", return_value=mock_ctx_manager):
        result = await source.call("web_search", {"query": "capital of France"})

    assert result == "Paris is the capital of France"


@pytest.mark.asyncio
async def test_registry_source_call_http_error_raises_tool_error():
    source = RegistryToolSource(
        skills=["web_search"],
        registry_url="https://registry.example.com",
        api_key="key",
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_ctx_manager = AsyncMock()
    mock_ctx_manager.__aenter__ = AsyncMock(return_value=mock_client)
    mock_ctx_manager.__aexit__ = AsyncMock(return_value=None)

    with patch("ninetrix.runtime.dispatcher.get_http_client", return_value=mock_ctx_manager):
        with pytest.raises(ToolError, match="HTTP 500"):
            await source.call("web_search", {"query": "test"})
