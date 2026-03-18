"""Tests for runtime/streaming.py + agent.stream() — PR 21."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ninetrix._internals.types import LLMResponse, StreamEvent, ToolCall
from ninetrix.runtime.streaming import StreamingRunner
from ninetrix.runtime.runner import RunnerConfig
from ninetrix.runtime.dispatcher import ToolDispatcher


# =============================================================================
# Helpers
# =============================================================================


def _provider(content: str = "answer", tool_calls: list | None = None) -> MagicMock:
    p = MagicMock()
    p.complete = AsyncMock(return_value=LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        input_tokens=5,
        output_tokens=3,
    ))
    return p


def _runner(provider, tools=None) -> StreamingRunner:
    return StreamingRunner(
        provider=provider,
        dispatcher=ToolDispatcher(tools or []),
        config=RunnerConfig(name="bot"),
    )


async def _collect(gen) -> list[StreamEvent]:
    events = []
    async for event in gen:
        events.append(event)
    return events


# =============================================================================
# Basic event sequence — no tool calls
# =============================================================================


@pytest.mark.asyncio
async def test_stream_yields_token_turn_end_done():
    r = _runner(_provider("hello"))
    events = await _collect(r.stream("hi"))
    types = [e.type for e in events]
    assert "token" in types
    assert "turn_end" in types
    assert "done" in types


@pytest.mark.asyncio
async def test_stream_token_content():
    r = _runner(_provider("the answer"))
    events = await _collect(r.stream("hi"))
    tokens = [e for e in events if e.type == "token"]
    assert len(tokens) == 1
    assert tokens[0].content == "the answer"


@pytest.mark.asyncio
async def test_stream_done_is_last():
    r = _runner(_provider("ok"))
    events = await _collect(r.stream("hi"))
    assert events[-1].type == "done"


@pytest.mark.asyncio
async def test_stream_done_has_cost():
    r = _runner(_provider("ok"))
    events = await _collect(r.stream("hi"))
    done = events[-1]
    assert done.cost_usd >= 0
    assert done.tokens_used >= 0


@pytest.mark.asyncio
async def test_stream_done_has_content():
    r = _runner(_provider("final answer"))
    events = await _collect(r.stream("hi"))
    done = events[-1]
    assert done.content == "final answer"


@pytest.mark.asyncio
async def test_stream_empty_response_no_token_event():
    """Provider returns empty content — no token event yielded."""
    r = _runner(_provider(""))
    events = await _collect(r.stream("hi"))
    token_events = [e for e in events if e.type == "token"]
    assert len(token_events) == 0
    # done should still be yielded
    assert events[-1].type == "done"


# =============================================================================
# Tool call events
# =============================================================================


@pytest.mark.asyncio
async def test_stream_tool_start_and_end():
    from ninetrix.registry import ToolDef
    from ninetrix.runtime.dispatcher import LocalToolSource

    def my_tool(x: str) -> str:
        return "result"

    td = ToolDef(name="my_tool", description="", parameters={}, fn=my_tool)

    responses = [
        LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="my_tool", arguments={"x": "a"})],
            input_tokens=5, output_tokens=3,
        ),
        LLMResponse(content="done", tool_calls=[], input_tokens=5, output_tokens=3),
    ]
    call_count = [0]

    async def fake_complete(*args, **kwargs):
        r = responses[call_count[0]]
        call_count[0] += 1
        return r

    provider = MagicMock()
    provider.complete = fake_complete

    runner = StreamingRunner(
        provider=provider,
        dispatcher=ToolDispatcher([LocalToolSource([td])]),
        config=RunnerConfig(name="bot"),
    )
    events = await _collect(runner.stream("use it"))
    types = [e.type for e in events]
    assert "tool_start" in types
    assert "tool_end" in types


@pytest.mark.asyncio
async def test_stream_tool_start_carries_name_and_args():
    from ninetrix.registry import ToolDef
    from ninetrix.runtime.dispatcher import LocalToolSource

    def calc(n: int) -> str:
        return str(n * 2)

    td = ToolDef(name="calc", description="", parameters={}, fn=calc)

    responses = [
        LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="calc", arguments={"n": 7})],
            input_tokens=5, output_tokens=3,
        ),
        LLMResponse(content="14", tool_calls=[], input_tokens=5, output_tokens=3),
    ]
    idx = [0]

    async def fake_complete(*args, **kwargs):
        r = responses[idx[0]]
        idx[0] += 1
        return r

    provider = MagicMock()
    provider.complete = fake_complete

    runner = StreamingRunner(
        provider=provider,
        dispatcher=ToolDispatcher([LocalToolSource([td])]),
        config=RunnerConfig(name="bot"),
    )
    events = await _collect(runner.stream("calc"))
    start = next(e for e in events if e.type == "tool_start")
    assert start.tool_name == "calc"
    assert start.tool_args == {"n": 7}


@pytest.mark.asyncio
async def test_stream_tool_end_carries_result():
    from ninetrix.registry import ToolDef
    from ninetrix.runtime.dispatcher import LocalToolSource

    def greet(name: str) -> str:
        return f"hello {name}"

    td = ToolDef(name="greet", description="", parameters={}, fn=greet)

    responses = [
        LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="greet", arguments={"name": "world"})],
            input_tokens=5, output_tokens=3,
        ),
        LLMResponse(content="ok", tool_calls=[], input_tokens=5, output_tokens=3),
    ]
    idx = [0]

    async def fake_complete(*args, **kwargs):
        r = responses[idx[0]]
        idx[0] += 1
        return r

    provider = MagicMock()
    provider.complete = fake_complete

    runner = StreamingRunner(
        provider=provider,
        dispatcher=ToolDispatcher([LocalToolSource([td])]),
        config=RunnerConfig(name="bot"),
    )
    events = await _collect(runner.stream("greet"))
    end = next(e for e in events if e.type == "tool_end")
    assert end.tool_name == "greet"
    assert "hello world" in end.tool_result


# =============================================================================
# turn_end event
# =============================================================================


@pytest.mark.asyncio
async def test_stream_turn_end_has_tokens_and_cost():
    r = _runner(_provider("answer"))
    events = await _collect(r.stream("hi"))
    te = next(e for e in events if e.type == "turn_end")
    assert te.tokens_used >= 0
    assert te.cost_usd >= 0


# =============================================================================
# Error event
# =============================================================================


@pytest.mark.asyncio
async def test_stream_error_event_on_provider_failure():
    provider = MagicMock()
    provider.complete = AsyncMock(side_effect=RuntimeError("LLM down"))
    runner = StreamingRunner(
        provider=provider,
        dispatcher=ToolDispatcher([]),
        config=RunnerConfig(name="bot"),
    )
    events = await _collect(runner.stream("hi"))
    assert events[-1].type == "error"
    assert isinstance(events[-1].error, RuntimeError)


@pytest.mark.asyncio
async def test_stream_error_does_not_raise():
    """Generator must terminate normally even on errors."""
    provider = MagicMock()
    provider.complete = AsyncMock(side_effect=ValueError("bad"))
    runner = StreamingRunner(
        provider=provider,
        dispatcher=ToolDispatcher([]),
        config=RunnerConfig(name="bot"),
    )
    # Should NOT raise
    events = await _collect(runner.stream("hi"))
    assert any(e.type == "error" for e in events)


@pytest.mark.asyncio
async def test_stream_no_done_event_on_error():
    """If error is raised before done, no done event is yielded."""
    provider = MagicMock()
    provider.complete = AsyncMock(side_effect=RuntimeError("boom"))
    runner = StreamingRunner(
        provider=provider,
        dispatcher=ToolDispatcher([]),
        config=RunnerConfig(name="bot"),
    )
    events = await _collect(runner.stream("hi"))
    assert not any(e.type == "done" for e in events)


# =============================================================================
# Structured output
# =============================================================================


@pytest.mark.asyncio
async def test_stream_done_structured_output():
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: int

    provider = MagicMock()
    provider.complete = AsyncMock(return_value=LLMResponse(
        content='{"value": 42}',
        tool_calls=[],
        input_tokens=5, output_tokens=3,
    ))

    runner = StreamingRunner(
        provider=provider,
        dispatcher=ToolDispatcher([]),
        config=RunnerConfig(name="bot", output_type=Answer),
    )
    events = await _collect(runner.stream("hi"))
    done = events[-1]
    assert done.type == "done"
    assert done.structured_output is not None
    assert done.structured_output.value == 42


@pytest.mark.asyncio
async def test_stream_done_structured_output_none_on_parse_failure():
    """Invalid JSON → structured_output stays None, no exception."""
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: int

    provider = MagicMock()
    provider.complete = AsyncMock(return_value=LLMResponse(
        content="not json at all",
        tool_calls=[],
        input_tokens=5, output_tokens=3,
    ))

    runner = StreamingRunner(
        provider=provider,
        dispatcher=ToolDispatcher([]),
        config=RunnerConfig(name="bot", output_type=Answer),
    )
    events = await _collect(runner.stream("hi"))
    done = events[-1]
    assert done.type == "done"
    assert done.structured_output is None


# =============================================================================
# Max turns
# =============================================================================


@pytest.mark.asyncio
async def test_stream_max_turns_yields_done_with_limit_message():
    """When max_turns hit, done event is still yielded."""
    # Provider always returns tool calls → infinite loop capped by max_turns
    from ninetrix.registry import ToolDef
    from ninetrix.runtime.dispatcher import LocalToolSource

    def noop(x: str) -> str:
        return "ok"

    td = ToolDef(name="noop", description="", parameters={}, fn=noop)

    provider = MagicMock()
    provider.complete = AsyncMock(return_value=LLMResponse(
        content="",
        tool_calls=[ToolCall(id="tc1", name="noop", arguments={"x": "y"})],
        input_tokens=5, output_tokens=3,
    ))

    runner = StreamingRunner(
        provider=provider,
        dispatcher=ToolDispatcher([LocalToolSource([td])]),
        config=RunnerConfig(name="bot", max_turns=2),
    )
    events = await _collect(runner.stream("go"))
    done = events[-1]
    assert done.type == "done"
    assert "maximum turn limit" in done.content


# =============================================================================
# Agent.stream() — integration
# =============================================================================


@pytest.mark.asyncio
async def test_agent_stream_returns_async_generator():
    from ninetrix.agent.agent import Agent
    a = Agent(provider="anthropic")
    import inspect
    # stream() is an async generator function
    assert inspect.isasyncgenfunction(a.stream)


@pytest.mark.asyncio
async def test_agent_has_stream_method():
    from ninetrix import Agent
    assert callable(Agent(provider="anthropic").stream)


# =============================================================================
# StreamEvent dataclass
# =============================================================================


def test_stream_event_defaults():
    e = StreamEvent(type="token")
    assert e.content == ""
    assert e.tool_name is None
    assert e.tool_result is None
    assert e.tokens_used == 0
    assert e.cost_usd == 0.0
    assert e.error is None
    assert e.structured_output is None


def test_stream_event_tool_start():
    e = StreamEvent(type="tool_start", tool_name="search", tool_args={"q": "test"})
    assert e.tool_name == "search"
    assert e.tool_args == {"q": "test"}


def test_stream_event_done_with_structured():
    from pydantic import BaseModel

    class Out(BaseModel):
        score: float

    out = Out(score=0.9)
    e = StreamEvent(type="done", content="", structured_output=out, cost_usd=0.001)
    assert e.structured_output.score == 0.9
    assert e.cost_usd == 0.001


# =============================================================================
# Public imports
# =============================================================================


def test_streaming_runner_importable_from_ninetrix():
    from ninetrix import StreamingRunner
    assert StreamingRunner is not None


def test_stream_event_importable_from_ninetrix():
    from ninetrix import StreamEvent
    assert StreamEvent is not None


def test_streaming_runner_importable_from_runtime():
    from ninetrix.runtime.streaming import StreamingRunner
    assert StreamingRunner is not None


def test_stream_event_importable_from_observability():
    from ninetrix.observability.streaming import StreamEvent
    assert StreamEvent is not None
