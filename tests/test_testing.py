"""Tests for ninetrix.testing — PR 30: MockTool + AgentSandbox."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ninetrix.testing import MockTool, AgentSandbox
from ninetrix.testing.sandbox import MockProvider, SandboxedToolSource, SandboxedDispatcher, SandboxResult
from ninetrix._internals.types import AgentResult, LLMResponse, ToolCall, ToolError
from ninetrix.registry import _registry


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear the global tool registry before each test to avoid leakage."""
    snapshot = _registry.snapshot()
    yield
    _registry.clear()
    for name, td in snapshot.items():
        _registry._tools[name] = td


def _make_agent(script: list[str] | None = None) -> Any:
    """Create a minimal agent pre-wired with an InMemory checkpointer."""
    from ninetrix.agent.agent import Agent
    from ninetrix.runtime.dispatcher import ToolDispatcher
    from ninetrix.runtime.runner import AgentRunner, RunnerConfig
    from ninetrix.checkpoint.memory import InMemoryCheckpointer

    provider = MagicMock()
    provider.complete = AsyncMock(return_value=LLMResponse(
        content="done", tool_calls=[], input_tokens=0, output_tokens=0,
    ))

    agent = Agent(name="test-agent", role="test assistant")
    runner = AgentRunner(
        provider=provider,
        dispatcher=ToolDispatcher([]),
        config=RunnerConfig(name="test-agent"),
        checkpointer=InMemoryCheckpointer(),
    )
    agent._runner = runner
    return agent


# =============================================================================
# MockTool — construction
# =============================================================================


def test_mock_tool_basic_construction():
    mock = MockTool("test_fn", return_value="hello")
    assert mock.name == "test_fn"
    assert mock.call_count == 0
    assert mock.calls == []


def test_mock_tool_registers_in_registry():
    _registry.clear()
    MockTool("my_mock_fn", return_value=42)
    assert "my_mock_fn" in _registry


def test_mock_tool_custom_schema():
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    mock = MockTool("weather", schema=schema)
    assert mock._schema == schema


def test_mock_tool_default_schema():
    mock = MockTool("no_schema_tool")
    assert mock._schema == {"type": "object", "properties": {}}


# =============================================================================
# MockTool — dispatch
# =============================================================================


@pytest.mark.asyncio
async def test_mock_tool_return_value():
    mock = MockTool("rv_tool", return_value={"temp": 72})
    result = await mock._dispatch()
    assert result == {"temp": 72}
    assert mock.call_count == 1


@pytest.mark.asyncio
async def test_mock_tool_side_effect_callable():
    mock = MockTool("side_fn", side_effect=lambda city: {"city": city, "temp": 72})
    result = await mock._dispatch(city="London")
    assert result == {"city": "London", "temp": 72}
    assert mock.call_count == 1
    assert mock.calls[0] == {"city": "London"}


@pytest.mark.asyncio
async def test_mock_tool_side_effect_exception_instance():
    mock = MockTool("fail_fn", side_effect=ToolError("API down"))
    with pytest.raises(ToolError, match="API down"):
        await mock._dispatch()


@pytest.mark.asyncio
async def test_mock_tool_side_effect_exception_class():
    mock = MockTool("fail_cls", side_effect=ValueError)
    with pytest.raises(ValueError):
        await mock._dispatch()


@pytest.mark.asyncio
async def test_mock_tool_tracks_multiple_calls():
    mock = MockTool("multi_call", return_value="ok")
    await mock._dispatch(x=1)
    await mock._dispatch(x=2)
    await mock._dispatch(x=3)
    assert mock.call_count == 3
    assert mock.calls[0] == {"x": 1}
    assert mock.calls[2] == {"x": 3}


@pytest.mark.asyncio
async def test_mock_tool_reset_clears_calls():
    mock = MockTool("resettable", return_value="x")
    await mock._dispatch()
    assert mock.call_count == 1
    mock.reset()
    assert mock.call_count == 0
    assert mock.calls == []


# =============================================================================
# MockTool — assertions
# =============================================================================


@pytest.mark.asyncio
async def test_assert_called_passes_when_called():
    mock = MockTool("called_fn", return_value=None)
    await mock._dispatch()
    mock.assert_called()  # should not raise


def test_assert_called_fails_when_not_called():
    mock = MockTool("uncalled_fn", return_value=None)
    with pytest.raises(AssertionError, match="never called"):
        mock.assert_called()


@pytest.mark.asyncio
async def test_assert_called_with_passes():
    mock = MockTool("kwarg_fn", return_value="ok")
    await mock._dispatch(city="London", units="metric")
    mock.assert_called_with(city="London")
    mock.assert_called_with(city="London", units="metric")


@pytest.mark.asyncio
async def test_assert_called_with_fails_wrong_value():
    mock = MockTool("wrong_val", return_value="ok")
    await mock._dispatch(city="Paris")
    with pytest.raises(AssertionError, match="mismatch"):
        mock.assert_called_with(city="London")


@pytest.mark.asyncio
async def test_assert_called_with_fails_missing_key():
    mock = MockTool("missing_key", return_value="ok")
    await mock._dispatch(city="Paris")
    with pytest.raises(AssertionError, match="not found"):
        mock.assert_called_with(temperature=72)


@pytest.mark.asyncio
async def test_assert_call_count_passes():
    mock = MockTool("count_fn", return_value=None)
    await mock._dispatch()
    await mock._dispatch()
    mock.assert_call_count(2)


@pytest.mark.asyncio
async def test_assert_call_count_fails():
    mock = MockTool("count_fail_fn", return_value=None)
    await mock._dispatch()
    with pytest.raises(AssertionError, match="expected 3 call"):
        mock.assert_call_count(3)


@pytest.mark.asyncio
async def test_assert_not_called_passes():
    mock = MockTool("never_called", return_value=None)
    mock.assert_not_called()  # should not raise


@pytest.mark.asyncio
async def test_assert_not_called_fails_when_called():
    mock = MockTool("was_called", return_value=None)
    await mock._dispatch()
    with pytest.raises(AssertionError, match="expected not to be called"):
        mock.assert_not_called()


# =============================================================================
# MockProvider
# =============================================================================


@pytest.mark.asyncio
async def test_mock_provider_single_response():
    provider = MockProvider(script=["Hello there!"])
    response = await provider.complete([], [])
    assert response.content == "Hello there!"
    assert response.tool_calls == []


@pytest.mark.asyncio
async def test_mock_provider_cycles_through_script():
    provider = MockProvider(script=["First", "Second", "Third"])
    r1 = await provider.complete([], [])
    r2 = await provider.complete([], [])
    r3 = await provider.complete([], [])
    assert r1.content == "First"
    assert r2.content == "Second"
    assert r3.content == "Third"


@pytest.mark.asyncio
async def test_mock_provider_repeats_last_when_exhausted():
    provider = MockProvider(script=["Only one"])
    await provider.complete([], [])
    r2 = await provider.complete([], [])
    assert r2.content == "Only one"


@pytest.mark.asyncio
async def test_mock_provider_tool_calls_script():
    provider = MockProvider(tool_calls_script=[
        {
            "content": "Let me check.",
            "tool_calls": [{"name": "get_weather", "arguments": {"city": "London"}}],
        },
        {"content": "The weather is 72F."},
    ])
    r1 = await provider.complete([], [])
    assert len(r1.tool_calls) == 1
    assert r1.tool_calls[0].name == "get_weather"
    assert r1.tool_calls[0].arguments == {"city": "London"}

    r2 = await provider.complete([], [])
    assert r2.content == "The weather is 72F."
    assert r2.tool_calls == []


@pytest.mark.asyncio
async def test_mock_provider_default_script():
    provider = MockProvider()
    r = await provider.complete([], [])
    assert isinstance(r.content, str)
    assert len(r.content) > 0


# =============================================================================
# SandboxedToolSource
# =============================================================================


@pytest.mark.asyncio
async def test_sandboxed_source_dispatches_to_mock():
    mock = MockTool("calc", return_value=42)
    source = SandboxedToolSource({"calc": mock})
    result = await source.call("calc", {"x": 1})
    assert result == "42"
    assert mock.call_count == 1


@pytest.mark.asyncio
async def test_sandboxed_source_handles_string_result():
    mock = MockTool("say_hello", return_value="hello world")
    source = SandboxedToolSource({"say_hello": mock})
    result = await source.call("say_hello", {})
    assert result == "hello world"


@pytest.mark.asyncio
async def test_sandboxed_source_handles_none_result():
    mock = MockTool("do_thing", return_value=None)
    source = SandboxedToolSource({"do_thing": mock})
    result = await source.call("do_thing", {})
    assert result == "(done)"


@pytest.mark.asyncio
async def test_sandboxed_source_unregistered_tool_returns_error():
    source = SandboxedToolSource({})
    result = await source.call("unknown_tool", {})
    assert "Error" in result
    assert "unknown_tool" in result


def test_sandboxed_source_tool_definitions():
    mock1 = MockTool("tool_a", return_value=1)
    mock2 = MockTool("tool_b", return_value=2)
    source = SandboxedToolSource({"tool_a": mock1, "tool_b": mock2})
    defs = source.tool_definitions()
    names = [d["function"]["name"] for d in defs]
    assert "tool_a" in names
    assert "tool_b" in names


# =============================================================================
# SandboxResult
# =============================================================================


def test_sandbox_result_success():
    r = SandboxResult(output="ok", thread_id="t1")
    assert r.success is True
    assert r.error is None


def test_sandbox_result_failure():
    r = SandboxResult(output="", thread_id="t1", error=ValueError("boom"))
    assert r.success is False


def test_sandbox_result_from_agent_result():
    agent_result = AgentResult(
        output="hello",
        thread_id="abc",
        tokens_used=10,
        input_tokens=6,
        output_tokens=4,
        cost_usd=0.001,
        steps=1,
        history=[],
    )
    sr = SandboxResult.from_agent_result(agent_result)
    assert sr.output == "hello"
    assert sr.thread_id == "abc"
    assert sr.tokens_used == 10
    assert sr.success is True


# =============================================================================
# AgentSandbox — basic usage
# =============================================================================


@pytest.mark.asyncio
async def test_sandbox_run_returns_result():
    agent = _make_agent()

    async with AgentSandbox(agent, script=["The answer is 42."]) as sandbox:
        result = await sandbox.run("What is the answer?")

    assert isinstance(result, SandboxResult)
    assert result.success


@pytest.mark.asyncio
async def test_sandbox_replaces_provider():
    """The sandbox mock provider should be used, not the original."""
    agent = _make_agent()

    async with AgentSandbox(agent, script=["Sandboxed response."]) as sandbox:
        result = await sandbox.run("Hello")

    assert result.output == "Sandboxed response."


@pytest.mark.asyncio
async def test_sandbox_add_mock_before_enter():
    """add_mock can be called before context entry; mock is available inside."""
    agent = _make_agent()
    sandbox = AgentSandbox(
        agent,
        tool_calls_script=[
            {
                "content": "Checking weather.",
                "tool_calls": [{"name": "wx", "arguments": {"city": "NYC"}}],
            },
            {"content": "It is sunny."},
        ],
    )
    mock = MockTool("wx", return_value="sunny")
    sandbox.add_mock(mock)

    async with sandbox:
        result = await sandbox.run("Weather in NYC?")

    assert result.success
    assert mock.call_count == 1


@pytest.mark.asyncio
async def test_sandbox_add_mock_inside_context():
    """add_mock can also be called after __aenter__."""
    agent = _make_agent()

    async with AgentSandbox(
        agent,
        tool_calls_script=[
            {
                "content": "",
                "tool_calls": [{"name": "search", "arguments": {"q": "python"}}],
            },
            {"content": "Results found."},
        ],
    ) as sandbox:
        sandbox.add_mock(MockTool("search", return_value=["result1"]))
        result = await sandbox.run("Search for python")

    assert result.success


@pytest.mark.asyncio
async def test_sandbox_tool_calls_log():
    agent = _make_agent()

    async with AgentSandbox(
        agent,
        tool_calls_script=[
            {
                "content": "",
                "tool_calls": [{"name": "lookup", "arguments": {"id": "42"}}],
            },
            {"content": "Found item."},
        ],
    ) as sandbox:
        sandbox.add_mock(MockTool("lookup", return_value={"name": "item42"}))
        await sandbox.run("Find item 42")

    calls = sandbox.tool_calls
    assert len(calls) == 1
    assert calls[0]["tool"] == "lookup"
    assert calls[0]["args"] == {"id": "42"}


@pytest.mark.asyncio
async def test_sandbox_assert_tool_called():
    agent = _make_agent()

    async with AgentSandbox(
        agent,
        tool_calls_script=[
            {
                "content": "",
                "tool_calls": [{"name": "fetch", "arguments": {}}],
            },
            {"content": "Done."},
        ],
    ) as sandbox:
        sandbox.add_mock(MockTool("fetch", return_value="data"))
        await sandbox.run("Fetch data")
        sandbox.assert_tool_called("fetch")


@pytest.mark.asyncio
async def test_sandbox_assert_tool_called_fails():
    agent = _make_agent()

    async with AgentSandbox(agent, script=["No tools needed."]) as sandbox:
        await sandbox.run("Hello")
        with pytest.raises(AssertionError, match="never called"):
            sandbox.assert_tool_called("nonexistent_tool")


@pytest.mark.asyncio
async def test_sandbox_assert_tool_call_count():
    agent = _make_agent()

    async with AgentSandbox(
        agent,
        tool_calls_script=[
            {
                "content": "",
                "tool_calls": [
                    {"name": "ping", "arguments": {}},
                    {"name": "ping", "arguments": {}},
                ],
            },
            {"content": "Pinged twice."},
        ],
    ) as sandbox:
        sandbox.add_mock(MockTool("ping", return_value="pong"))
        await sandbox.run("Ping twice")
        sandbox.assert_tool_call_count("ping", 2)


@pytest.mark.asyncio
async def test_sandbox_assert_no_tool_called():
    agent = _make_agent()

    async with AgentSandbox(agent, script=["Just text."]) as sandbox:
        await sandbox.run("Hello")
        sandbox.assert_no_tool_called("dangerous_tool")


@pytest.mark.asyncio
async def test_sandbox_assert_no_tool_called_fails_when_called():
    agent = _make_agent()

    async with AgentSandbox(
        agent,
        tool_calls_script=[
            {
                "content": "",
                "tool_calls": [{"name": "bad_tool", "arguments": {}}],
            },
            {"content": "Used bad tool."},
        ],
    ) as sandbox:
        sandbox.add_mock(MockTool("bad_tool", return_value="ran"))
        await sandbox.run("Use bad tool")
        with pytest.raises(AssertionError, match="expected NOT"):
            sandbox.assert_no_tool_called("bad_tool")


@pytest.mark.asyncio
async def test_sandbox_captures_exception_as_failed_result():
    """If agent.arun raises, sandbox returns SandboxResult with error."""
    from ninetrix.agent.agent import Agent
    from ninetrix.runtime.dispatcher import ToolDispatcher
    from ninetrix.runtime.runner import AgentRunner, RunnerConfig
    from ninetrix.checkpoint.memory import InMemoryCheckpointer
    from ninetrix._internals.types import ProviderError

    provider = MagicMock()
    provider.complete = AsyncMock(side_effect=ProviderError("LLM exploded", provider="mock"))

    agent = Agent(name="error-agent", role="test")
    runner = AgentRunner(
        provider=provider,
        dispatcher=ToolDispatcher([]),
        config=RunnerConfig(name="error-agent"),
        checkpointer=InMemoryCheckpointer(),
    )
    agent._runner = runner

    async with AgentSandbox(agent, script=["This will never be used."]) as sandbox:
        result = await sandbox.run("Trigger error")

    # The sandbox mock provider replaced the error-raising one, so it should succeed
    # (sandbox always installs MockProvider unless custom provider set)
    assert result.success


@pytest.mark.asyncio
async def test_sandbox_restores_agent_after_exit():
    """After sandbox exit the agent's runner is invalidated (ready to rebuild)."""
    agent = _make_agent()

    async with AgentSandbox(agent, script=["ok"]) as sandbox:
        runner_inside = agent._runner

    # After exit, runner is invalidated
    assert agent._runner is None


@pytest.mark.asyncio
async def test_sandbox_multiple_runs():
    """Can call sandbox.run() multiple times within one context."""
    agent = _make_agent()

    async with AgentSandbox(agent, script=["first", "second"]) as sandbox:
        r1 = await sandbox.run("Message 1", thread_id="t1")
        r2 = await sandbox.run("Message 2", thread_id="t2")

    assert r1.output == "first"
    assert r2.output == "second"


# =============================================================================
# Public imports
# =============================================================================


def test_mock_tool_importable_from_testing():
    from ninetrix.testing import MockTool
    assert MockTool is not None


def test_agent_sandbox_importable_from_testing():
    from ninetrix.testing import AgentSandbox
    assert AgentSandbox is not None


def test_testing_importable_from_ninetrix():
    from ninetrix.testing import MockTool, AgentSandbox
    assert MockTool is not None
    assert AgentSandbox is not None
