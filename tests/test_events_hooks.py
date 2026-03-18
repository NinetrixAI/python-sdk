"""Tests for observability/events.py + observability/hooks.py — PR 20."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ninetrix.observability.events import AgentEvent, EventBus
from ninetrix.observability.hooks import HooksMixin
from ninetrix.agent.agent import Agent


# =============================================================================
# AgentEvent
# =============================================================================


def test_agent_event_defaults():
    e = AgentEvent(type="tool.call")
    assert e.type == "tool.call"
    assert e.thread_id == ""
    assert e.agent_name == ""
    assert e.data == {}


def test_agent_event_fields():
    e = AgentEvent(type="run.end", thread_id="t1", agent_name="bot", data={"cost": 0.01})
    assert e.thread_id == "t1"
    assert e.agent_name == "bot"
    assert e.data["cost"] == 0.01


# =============================================================================
# EventBus — subscribe + emit
# =============================================================================


@pytest.mark.asyncio
async def test_subscribe_and_emit_sync_handler():
    bus = EventBus()
    received = []

    def handler(event: AgentEvent):
        received.append(event.type)

    bus.subscribe("tool.call", handler)
    await bus.emit(AgentEvent(type="tool.call"))
    assert received == ["tool.call"]


@pytest.mark.asyncio
async def test_subscribe_and_emit_async_handler():
    bus = EventBus()
    received = []

    async def handler(event: AgentEvent):
        received.append(event.type)

    bus.subscribe("tool.call", handler)
    await bus.emit(AgentEvent(type="tool.call"))
    assert received == ["tool.call"]


@pytest.mark.asyncio
async def test_emit_no_subscribers_is_noop():
    bus = EventBus()
    await bus.emit(AgentEvent(type="run.start"))  # should not raise


@pytest.mark.asyncio
async def test_multiple_handlers_for_same_event():
    bus = EventBus()
    log = []
    bus.subscribe("run.start", lambda e: log.append("a"))
    bus.subscribe("run.start", lambda e: log.append("b"))
    await bus.emit(AgentEvent(type="run.start"))
    assert log == ["a", "b"]


@pytest.mark.asyncio
async def test_handler_receives_event_data():
    bus = EventBus()
    captured: list[AgentEvent] = []
    bus.subscribe("tool.call", lambda e: captured.append(e))
    ev = AgentEvent(type="tool.call", thread_id="t1", data={"tool_name": "search"})
    await bus.emit(ev)
    assert captured[0].data["tool_name"] == "search"
    assert captured[0].thread_id == "t1"


# =============================================================================
# EventBus — wildcard matching
# =============================================================================


@pytest.mark.asyncio
async def test_global_wildcard_catches_all():
    bus = EventBus()
    log = []
    bus.subscribe("*", lambda e: log.append(e.type))
    await bus.emit(AgentEvent(type="run.start"))
    await bus.emit(AgentEvent(type="tool.call"))
    await bus.emit(AgentEvent(type="run.end"))
    assert log == ["run.start", "tool.call", "run.end"]


@pytest.mark.asyncio
async def test_prefix_wildcard_matches_namespace():
    bus = EventBus()
    log = []
    bus.subscribe("tool.*", lambda e: log.append(e.type))
    await bus.emit(AgentEvent(type="tool.call"))
    await bus.emit(AgentEvent(type="tool.result"))
    await bus.emit(AgentEvent(type="run.start"))   # should NOT be captured
    assert "tool.call" in log
    assert "tool.result" in log
    assert "run.start" not in log


@pytest.mark.asyncio
async def test_prefix_wildcard_does_not_match_unrelated():
    bus = EventBus()
    log = []
    bus.subscribe("run.*", lambda e: log.append(e.type))
    await bus.emit(AgentEvent(type="tool.call"))
    assert log == []


@pytest.mark.asyncio
async def test_exact_and_wildcard_both_fire():
    bus = EventBus()
    log = []
    bus.subscribe("tool.call", lambda e: log.append("exact"))
    bus.subscribe("tool.*", lambda e: log.append("prefix"))
    bus.subscribe("*", lambda e: log.append("global"))
    await bus.emit(AgentEvent(type="tool.call"))
    assert "exact" in log
    assert "prefix" in log
    assert "global" in log


# =============================================================================
# EventBus — unsubscribe
# =============================================================================


@pytest.mark.asyncio
async def test_unsubscribe_removes_handler():
    bus = EventBus()
    log = []

    def handler(e: AgentEvent):
        log.append(e.type)

    bus.subscribe("run.start", handler)
    bus.unsubscribe("run.start", handler)
    await bus.emit(AgentEvent(type="run.start"))
    assert log == []


@pytest.mark.asyncio
async def test_unsubscribe_nonexistent_is_noop():
    bus = EventBus()
    bus.unsubscribe("run.start", lambda e: None)  # should not raise


@pytest.mark.asyncio
async def test_unsubscribe_one_of_two_handlers():
    bus = EventBus()
    log = []
    h1 = lambda e: log.append("h1")
    h2 = lambda e: log.append("h2")
    bus.subscribe("run.start", h1)
    bus.subscribe("run.start", h2)
    bus.unsubscribe("run.start", h1)
    await bus.emit(AgentEvent(type="run.start"))
    assert log == ["h2"]


# =============================================================================
# EventBus — subscriber_count + clear
# =============================================================================


def test_subscriber_count_total():
    bus = EventBus()
    bus.subscribe("tool.call", lambda e: None)
    bus.subscribe("run.start", lambda e: None)
    bus.subscribe("run.start", lambda e: None)
    assert bus.subscriber_count() == 3


def test_subscriber_count_by_type():
    bus = EventBus()
    bus.subscribe("tool.call", lambda e: None)
    bus.subscribe("tool.call", lambda e: None)
    assert bus.subscriber_count("tool.call") == 2
    assert bus.subscriber_count("run.start") == 0


def test_clear_removes_all():
    bus = EventBus()
    bus.subscribe("tool.call", lambda e: None)
    bus.subscribe("*", lambda e: None)
    bus.clear()
    assert bus.subscriber_count() == 0


def test_repr():
    bus = EventBus()
    bus.subscribe("tool.call", lambda e: None)
    r = repr(bus)
    assert "EventBus" in r


# =============================================================================
# HooksMixin
# =============================================================================


class _FakeAgent(HooksMixin):
    """Minimal Agent-like class for testing HooksMixin in isolation."""
    def __init__(self):
        super().__init__()


def test_hooksmixin_initialises_event_bus():
    a = _FakeAgent()
    assert a._event_bus is not None
    assert isinstance(a._event_bus, EventBus)


@pytest.mark.asyncio
async def test_on_registers_sync_handler():
    a = _FakeAgent()
    received = []

    @a.on("tool.call")
    def handler(event):
        received.append(event.type)

    await a._event_bus.emit(AgentEvent(type="tool.call"))
    assert received == ["tool.call"]


@pytest.mark.asyncio
async def test_on_registers_async_handler():
    a = _FakeAgent()
    received = []

    @a.on("run.end")
    async def handler(event):
        received.append(event.type)

    await a._event_bus.emit(AgentEvent(type="run.end"))
    assert received == ["run.end"]


@pytest.mark.asyncio
async def test_on_returns_original_function():
    a = _FakeAgent()

    def my_handler(event):
        pass

    result = a.on("tool.call")(my_handler)
    assert result is my_handler


@pytest.mark.asyncio
async def test_off_removes_handler():
    a = _FakeAgent()
    log = []

    def handler(event):
        log.append(1)

    a.on("tool.call")(handler)
    a.off("tool.call", handler)
    await a._event_bus.emit(AgentEvent(type="tool.call"))
    assert log == []


@pytest.mark.asyncio
async def test_once_fires_exactly_once():
    a = _FakeAgent()
    count = [0]

    @a.once("run.end")
    def handler(event):
        count[0] += 1

    await a._event_bus.emit(AgentEvent(type="run.end"))
    await a._event_bus.emit(AgentEvent(type="run.end"))
    await a._event_bus.emit(AgentEvent(type="run.end"))
    assert count[0] == 1


@pytest.mark.asyncio
async def test_once_returns_original_function():
    a = _FakeAgent()

    def my_handler(event):
        pass

    result = a.once("run.end")(my_handler)
    assert result is my_handler


@pytest.mark.asyncio
async def test_on_wildcard_via_hooksmixin():
    a = _FakeAgent()
    log = []

    @a.on("*")
    def handler(event):
        log.append(event.type)

    await a._event_bus.emit(AgentEvent(type="tool.call"))
    await a._event_bus.emit(AgentEvent(type="run.end"))
    assert log == ["tool.call", "run.end"]


@pytest.mark.asyncio
async def test_on_prefix_wildcard_via_hooksmixin():
    a = _FakeAgent()
    log = []

    @a.on("tool.*")
    def handler(event):
        log.append(event.type)

    await a._event_bus.emit(AgentEvent(type="tool.call"))
    await a._event_bus.emit(AgentEvent(type="tool.result"))
    await a._event_bus.emit(AgentEvent(type="run.start"))
    assert log == ["tool.call", "tool.result"]


# =============================================================================
# Agent inherits HooksMixin
# =============================================================================


def test_agent_has_on_method():
    a = Agent(provider="anthropic")
    assert callable(a.on)


def test_agent_has_off_method():
    a = Agent(provider="anthropic")
    assert callable(a.off)


def test_agent_has_once_method():
    a = Agent(provider="anthropic")
    assert callable(a.once)


def test_agent_has_event_bus():
    a = Agent(provider="anthropic")
    assert isinstance(a._event_bus, EventBus)


@pytest.mark.asyncio
async def test_agent_on_decorator_registers_handler():
    a = Agent(provider="anthropic")
    log = []

    @a.on("run.start")
    def handler(event):
        log.append(event.type)

    await a._event_bus.emit(AgentEvent(type="run.start"))
    assert "run.start" in log


# =============================================================================
# Runner emits events through the bus
# =============================================================================


@pytest.mark.asyncio
async def test_runner_emits_run_start_and_run_end():
    from unittest.mock import AsyncMock, MagicMock
    from ninetrix._internals.types import LLMResponse
    from ninetrix.runtime.runner import AgentRunner, RunnerConfig
    from ninetrix.runtime.dispatcher import ToolDispatcher

    provider = MagicMock()
    provider.complete = AsyncMock(return_value=LLMResponse(
        content="done", tool_calls=[], input_tokens=5, output_tokens=3,
    ))

    bus = EventBus()
    log = []
    bus.subscribe("run.start", lambda e: log.append("run.start"))
    bus.subscribe("run.end", lambda e: log.append("run.end"))

    runner = AgentRunner(
        provider=provider,
        dispatcher=ToolDispatcher([]),
        config=RunnerConfig(name="bot"),
        event_bus=bus,
    )
    await runner.run("hello")

    assert "run.start" in log
    assert "run.end" in log


@pytest.mark.asyncio
async def test_runner_emits_turn_start_and_turn_end():
    from unittest.mock import AsyncMock, MagicMock
    from ninetrix._internals.types import LLMResponse
    from ninetrix.runtime.runner import AgentRunner, RunnerConfig
    from ninetrix.runtime.dispatcher import ToolDispatcher

    provider = MagicMock()
    provider.complete = AsyncMock(return_value=LLMResponse(
        content="answer", tool_calls=[], input_tokens=5, output_tokens=3,
    ))

    bus = EventBus()
    log = []
    bus.subscribe("turn.start", lambda e: log.append("turn.start"))
    bus.subscribe("turn.end", lambda e: log.append("turn.end"))

    runner = AgentRunner(
        provider=provider,
        dispatcher=ToolDispatcher([]),
        config=RunnerConfig(name="bot"),
        event_bus=bus,
    )
    await runner.run("hi")

    assert "turn.start" in log
    assert "turn.end" in log


@pytest.mark.asyncio
async def test_runner_emits_tool_call_and_tool_result():
    from unittest.mock import AsyncMock, MagicMock
    from ninetrix._internals.types import LLMResponse, ToolCall
    from ninetrix.runtime.runner import AgentRunner, RunnerConfig
    from ninetrix.runtime.dispatcher import ToolDispatcher
    from ninetrix.registry import ToolDef

    # Tool that returns immediately
    def my_tool(query: str) -> str:
        return "result"

    td = ToolDef(name="my_tool", description="A tool", parameters={}, fn=my_tool)

    responses = [
        LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="my_tool", arguments={"query": "q"})],
            input_tokens=5, output_tokens=3,
        ),
        LLMResponse(content="final", tool_calls=[], input_tokens=5, output_tokens=3),
    ]
    call_count = [0]

    async def fake_complete(*args, **kwargs):
        r = responses[call_count[0]]
        call_count[0] += 1
        return r

    provider = MagicMock()
    provider.complete = fake_complete

    from ninetrix.runtime.dispatcher import LocalToolSource
    bus = EventBus()
    tool_events = []
    bus.subscribe("tool.call", lambda e: tool_events.append(("call", e.data["tool_name"])))
    bus.subscribe("tool.result", lambda e: tool_events.append(("result", e.data["tool_name"])))

    runner = AgentRunner(
        provider=provider,
        dispatcher=ToolDispatcher([LocalToolSource([td])]),
        config=RunnerConfig(name="bot"),
        event_bus=bus,
    )
    await runner.run("use the tool")

    assert ("call", "my_tool") in tool_events
    assert ("result", "my_tool") in tool_events


@pytest.mark.asyncio
async def test_runner_without_event_bus_still_works():
    from unittest.mock import AsyncMock, MagicMock
    from ninetrix._internals.types import LLMResponse
    from ninetrix.runtime.runner import AgentRunner, RunnerConfig
    from ninetrix.runtime.dispatcher import ToolDispatcher

    provider = MagicMock()
    provider.complete = AsyncMock(return_value=LLMResponse(
        content="hi", tool_calls=[], input_tokens=2, output_tokens=1,
    ))

    runner = AgentRunner(
        provider=provider,
        dispatcher=ToolDispatcher([]),
        config=RunnerConfig(name="bot"),
        event_bus=None,   # no bus
    )
    result = await runner.run("hello")
    assert result.output == "hi"


@pytest.mark.asyncio
async def test_handler_error_does_not_crash_runner():
    """Errors in event handlers must not propagate to the agent run."""
    from unittest.mock import AsyncMock, MagicMock
    from ninetrix._internals.types import LLMResponse
    from ninetrix.runtime.runner import AgentRunner, RunnerConfig
    from ninetrix.runtime.dispatcher import ToolDispatcher

    provider = MagicMock()
    provider.complete = AsyncMock(return_value=LLMResponse(
        content="ok", tool_calls=[], input_tokens=2, output_tokens=1,
    ))

    bus = EventBus()
    bus.subscribe("run.start", lambda e: (_ for _ in ()).throw(RuntimeError("boom")))

    runner = AgentRunner(
        provider=provider,
        dispatcher=ToolDispatcher([]),
        config=RunnerConfig(name="bot"),
        event_bus=bus,
    )
    result = await runner.run("hello")   # must NOT raise
    assert result.output == "ok"


# =============================================================================
# Public imports
# =============================================================================


def test_agent_event_importable_from_ninetrix():
    from ninetrix import AgentEvent
    assert AgentEvent is not None


def test_event_bus_importable_from_ninetrix():
    from ninetrix import EventBus
    assert EventBus is not None


def test_hooks_mixin_importable_from_ninetrix():
    from ninetrix import HooksMixin
    assert HooksMixin is not None
