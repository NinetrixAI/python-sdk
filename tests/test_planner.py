"""Tests for runtime/planner.py — PR 24."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ninetrix import Planner
from ninetrix._internals.types import LLMResponse
from ninetrix.runtime.planner import _parse_plan
from ninetrix.runtime.runner import RunnerConfig


# =============================================================================
# _parse_plan helper
# =============================================================================


def test_parse_plan_bare_json():
    raw = '{"goal": "search the web", "steps": ["open browser", "search"]}'
    plan = _parse_plan(raw)
    assert plan["goal"] == "search the web"
    assert plan["steps"] == ["open browser", "search"]


def test_parse_plan_code_fence():
    raw = '```json\n{"goal": "do stuff", "steps": ["a", "b"]}\n```'
    plan = _parse_plan(raw)
    assert plan["goal"] == "do stuff"
    assert plan["steps"] == ["a", "b"]


def test_parse_plan_code_fence_no_lang():
    raw = '```\n{"goal": "run", "steps": ["step1"]}\n```'
    plan = _parse_plan(raw)
    assert plan["goal"] == "run"


def test_parse_plan_json_in_prose():
    raw = 'Here is the plan: {"goal": "g", "steps": ["s1"]} done.'
    plan = _parse_plan(raw)
    assert plan["goal"] == "g"
    assert "s1" in plan["steps"]


def test_parse_plan_empty_string():
    assert _parse_plan("") == {}


def test_parse_plan_invalid_json():
    assert _parse_plan("not json at all") == {}


def test_parse_plan_wrong_type():
    assert _parse_plan("[1, 2, 3]") == {}


def test_parse_plan_normalises_steps_to_strings():
    raw = '{"goal": "g", "steps": [1, 2, "three"]}'
    plan = _parse_plan(raw)
    assert all(isinstance(s, str) for s in plan["steps"])


def test_parse_plan_missing_fields_returns_empty_lists():
    raw = '{"goal": "something"}'
    plan = _parse_plan(raw)
    assert plan["goal"] == "something"
    assert plan["steps"] == []


def test_parse_plan_steps_not_list():
    raw = '{"goal": "g", "steps": "not a list"}'
    plan = _parse_plan(raw)
    assert plan["steps"] == []


# =============================================================================
# Planner — construction
# =============================================================================


def test_planner_constructs():
    p = Planner(RunnerConfig(name="bot"))
    assert p is not None


# =============================================================================
# Planner.plan — happy path
# =============================================================================


@pytest.mark.asyncio
async def test_planner_plan_returns_dict():
    config = RunnerConfig(name="bot")
    planner = Planner(config)

    provider = MagicMock()
    provider.complete = AsyncMock(return_value=LLMResponse(
        content='{"goal": "search for news", "steps": ["call search tool", "summarise"]}',
        tool_calls=[],
        input_tokens=10,
        output_tokens=20,
    ))

    plan = await planner.plan("Get latest news", [], provider)
    assert plan["goal"] == "search for news"
    assert len(plan["steps"]) == 2


@pytest.mark.asyncio
async def test_planner_plan_includes_tool_names():
    config = RunnerConfig(name="bot")
    planner = Planner(config)

    tool_defs = [
        {"type": "function", "function": {"name": "web_search", "description": "search"}},
        {"type": "function", "function": {"name": "summarise", "description": "sum"}},
    ]

    provider = MagicMock()
    provider.complete = AsyncMock(return_value=LLMResponse(
        content='{"goal": "find info", "steps": ["use web_search"]}',
        tool_calls=[],
        input_tokens=10,
        output_tokens=10,
    ))

    await planner.plan("Find info", tool_defs, provider)

    call_args = provider.complete.call_args
    messages = call_args[0][0]
    user_msg = next(m["content"] for m in messages if m["role"] == "user")
    assert "web_search" in user_msg
    assert "summarise" in user_msg


@pytest.mark.asyncio
async def test_planner_plan_uses_low_temperature():
    config = RunnerConfig(name="bot")
    planner = Planner(config)

    provider = MagicMock()
    provider.complete = AsyncMock(return_value=LLMResponse(
        content='{"goal": "g", "steps": []}',
        tool_calls=[],
        input_tokens=5,
        output_tokens=5,
    ))

    await planner.plan("hi", [], provider)

    call_kwargs = provider.complete.call_args[1]
    pconfig = call_kwargs.get("config")
    assert pconfig is not None
    assert pconfig.temperature == 0.0


@pytest.mark.asyncio
async def test_planner_plan_returns_empty_on_provider_error():
    config = RunnerConfig(name="bot")
    planner = Planner(config)

    provider = MagicMock()
    provider.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

    plan = await planner.plan("hello", [], provider)
    assert plan == {}


@pytest.mark.asyncio
async def test_planner_plan_returns_empty_on_bad_json():
    config = RunnerConfig(name="bot")
    planner = Planner(config)

    provider = MagicMock()
    provider.complete = AsyncMock(return_value=LLMResponse(
        content="I cannot create a plan right now.",
        tool_calls=[],
        input_tokens=5,
        output_tokens=5,
    ))

    plan = await planner.plan("hello", [], provider)
    assert plan == {}


# =============================================================================
# Planner.build_execution_prompt
# =============================================================================


def test_build_prompt_with_plan():
    planner = Planner(RunnerConfig(name="bot"))
    plan = {"goal": "find the answer", "steps": ["step1", "step2"]}
    result = planner.build_execution_prompt("What is X?", plan)
    assert "What is X?" in result
    assert "step1" in result
    assert "step2" in result
    assert "find the answer" in result


def test_build_prompt_empty_plan_returns_message():
    planner = Planner(RunnerConfig(name="bot"))
    result = planner.build_execution_prompt("original", {})
    assert result == "original"


def test_build_prompt_no_steps_returns_message():
    planner = Planner(RunnerConfig(name="bot"))
    result = planner.build_execution_prompt("original", {"goal": "g", "steps": []})
    assert result == "original"


def test_build_prompt_numbers_steps():
    planner = Planner(RunnerConfig(name="bot"))
    plan = {"goal": "g", "steps": ["a", "b", "c"]}
    result = planner.build_execution_prompt("msg", plan)
    assert "1." in result
    assert "2." in result
    assert "3." in result


def test_build_prompt_includes_plan_markers():
    planner = Planner(RunnerConfig(name="bot"))
    plan = {"goal": "g", "steps": ["do something"]}
    result = planner.build_execution_prompt("msg", plan)
    assert "Execution Plan" in result or "Plan" in result


# =============================================================================
# AgentRunner — planned execution mode
# =============================================================================


@pytest.mark.asyncio
async def test_runner_planned_mode_creates_planner():
    from ninetrix.runtime.runner import AgentRunner, RunnerConfig
    from ninetrix.runtime.dispatcher import ToolDispatcher

    provider = MagicMock()
    provider.complete = AsyncMock(return_value=LLMResponse(
        content="done", tool_calls=[], input_tokens=5, output_tokens=3,
    ))

    runner = AgentRunner(
        provider=provider,
        dispatcher=ToolDispatcher([]),
        config=RunnerConfig(name="bot", execution_mode="planned"),
    )
    assert runner._planner is not None
    assert isinstance(runner._planner, Planner)


def test_runner_direct_mode_no_planner():
    from ninetrix.runtime.runner import AgentRunner, RunnerConfig
    from ninetrix.runtime.dispatcher import ToolDispatcher

    provider = MagicMock()
    runner = AgentRunner(
        provider=provider,
        dispatcher=ToolDispatcher([]),
        config=RunnerConfig(name="bot", execution_mode="direct"),
    )
    assert runner._planner is None


@pytest.mark.asyncio
async def test_runner_planned_mode_calls_planner():
    from ninetrix.runtime.runner import AgentRunner, RunnerConfig
    from ninetrix.runtime.dispatcher import ToolDispatcher

    plan_response = '{"goal": "find answer", "steps": ["search", "summarise"]}'
    final_response = LLMResponse(
        content="Here is the answer", tool_calls=[], input_tokens=5, output_tokens=5,
    )

    call_count = [0]

    async def fake_complete(messages, tools, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call = planner
            return LLMResponse(
                content=plan_response, tool_calls=[], input_tokens=3, output_tokens=10,
            )
        return final_response

    provider = MagicMock()
    provider.complete = fake_complete

    runner = AgentRunner(
        provider=provider,
        dispatcher=ToolDispatcher([]),
        config=RunnerConfig(name="bot", execution_mode="planned"),
    )
    result = await runner.run("What is X?")
    # Planner + execution = at least 2 LLM calls
    assert call_count[0] >= 2
    assert result.output == "Here is the answer"


@pytest.mark.asyncio
async def test_runner_planned_mode_injects_plan_into_message():
    """Planned mode should inject step text into the user message sent to LLM."""
    from ninetrix.runtime.runner import AgentRunner, RunnerConfig
    from ninetrix.runtime.dispatcher import ToolDispatcher

    plan_json = '{"goal": "do the work", "steps": ["special_step_abc"]}'
    call_messages = []

    async def fake_complete(messages, tools, **kwargs):
        call_messages.append(messages)
        if len(call_messages) == 1:
            return LLMResponse(
                content=plan_json, tool_calls=[], input_tokens=3, output_tokens=10,
            )
        return LLMResponse(
            content="done", tool_calls=[], input_tokens=3, output_tokens=3,
        )

    provider = MagicMock()
    provider.complete = fake_complete

    runner = AgentRunner(
        provider=provider,
        dispatcher=ToolDispatcher([]),
        config=RunnerConfig(name="bot", execution_mode="planned"),
    )
    await runner.run("Do something")

    # Second call should contain the plan step in the user message
    exec_messages = call_messages[1]
    user_content = next(
        (m["content"] for m in exec_messages if m["role"] == "user"), ""
    )
    assert "special_step_abc" in user_content


@pytest.mark.asyncio
async def test_runner_planned_mode_planner_failure_falls_back():
    """If planner returns empty plan, run proceeds with original message."""
    from ninetrix.runtime.runner import AgentRunner, RunnerConfig
    from ninetrix.runtime.dispatcher import ToolDispatcher

    call_count = [0]

    async def fake_complete(messages, tools, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # Planner call returns garbage
            return LLMResponse(
                content="not json", tool_calls=[], input_tokens=3, output_tokens=3,
            )
        return LLMResponse(
            content="fine", tool_calls=[], input_tokens=3, output_tokens=3,
        )

    provider = MagicMock()
    provider.complete = fake_complete

    runner = AgentRunner(
        provider=provider,
        dispatcher=ToolDispatcher([]),
        config=RunnerConfig(name="bot", execution_mode="planned"),
    )
    # Should not raise
    result = await runner.run("hello")
    assert result.output == "fine"


# =============================================================================
# RunnerConfig — execution_mode field
# =============================================================================


def test_runner_config_default_mode():
    cfg = RunnerConfig()
    assert cfg.execution_mode == "direct"


def test_runner_config_planned_mode():
    cfg = RunnerConfig(execution_mode="planned")
    assert cfg.execution_mode == "planned"


# =============================================================================
# Agent — execution_mode wired through
# =============================================================================


def test_agent_execution_mode_default():
    from ninetrix import Agent
    a = Agent(provider="anthropic")
    assert a.config.execution_mode == "direct"


def test_agent_execution_mode_planned():
    from ninetrix import Agent
    a = Agent(provider="anthropic", execution_mode="planned")
    assert a.config.execution_mode == "planned"


# =============================================================================
# Public imports
# =============================================================================


def test_planner_importable_from_ninetrix():
    from ninetrix import Planner
    assert Planner is not None


def test_planner_importable_from_runtime():
    from ninetrix.runtime.planner import Planner
    assert Planner is not None
