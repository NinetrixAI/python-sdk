"""Tests for runtime/runner.py — PR 16."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ninetrix.runtime.runner import (
    AgentRunner,
    RunnerConfig,
    _build_assistant_message,
    _dispatch_tool,
    _extract_validation_errors,
    _get_output_schema,
)
from ninetrix.runtime.dispatcher import ToolDispatcher, LocalToolSource
from ninetrix._internals.types import (
    AgentResult,
    CheckpointerProtocol,
    LLMResponse,
    OutputParseError,
    ProviderConfig,
    ToolCall,
)
from ninetrix.registry import ToolDef


# =============================================================================
# Test fixtures / helpers
# =============================================================================


def _make_response(
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    input_tokens: int = 10,
    output_tokens: int = 20,
    stop_reason: str = "end_turn",
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        stop_reason=stop_reason,
    )


def _make_provider(responses: list[LLMResponse]) -> MagicMock:
    """Create a mock LLMProviderAdapter that returns responses in sequence."""
    provider = MagicMock()
    provider.complete = AsyncMock(side_effect=responses)
    return provider


def _make_dispatcher(tools: dict[str, str] | None = None) -> ToolDispatcher:
    """Create a dispatcher with simple sync tools."""
    if not tools:
        return ToolDispatcher([])

    tool_defs = []
    for name, return_val in tools.items():
        rv = return_val  # capture

        def make_fn(v: str):
            def fn(**kwargs: Any) -> str:
                return v

            fn.__name__ = name
            return fn

        td = ToolDef(
            name=name,
            description=f"Tool {name}",
            parameters={"type": "object", "properties": {}},
            fn=make_fn(rv),
        )
        tool_defs.append(td)

    return ToolDispatcher([LocalToolSource(tool_defs)])


def _make_runner(
    responses: list[LLMResponse],
    tools: dict[str, str] | None = None,
    config: RunnerConfig | None = None,
    checkpointer: Any = None,
) -> AgentRunner:
    provider = _make_provider(responses)
    dispatcher = _make_dispatcher(tools)
    cfg = config or RunnerConfig(name="test-agent", system_prompt="You are helpful.")
    return AgentRunner(
        provider=provider,
        dispatcher=dispatcher,
        config=cfg,
        checkpointer=checkpointer,
    )


# =============================================================================
# RunnerConfig
# =============================================================================


def test_runner_config_defaults():
    cfg = RunnerConfig()
    assert cfg.name == "agent"
    assert cfg.max_turns == 20
    assert cfg.max_tokens == 8192
    assert cfg.output_type is None
    assert cfg.output_retries == 1


def test_runner_config_effective_provider_config_default():
    cfg = RunnerConfig(temperature=0.5, max_tokens=4096)
    pc = cfg.effective_provider_config()
    assert pc.temperature == 0.5
    assert pc.max_tokens == 4096


def test_runner_config_effective_provider_config_explicit():
    explicit = ProviderConfig(temperature=0.1, max_tokens=1000)
    cfg = RunnerConfig(provider_config=explicit)
    assert cfg.effective_provider_config() is explicit


# =============================================================================
# AgentRunner.run() — basic cases
# =============================================================================


@pytest.mark.asyncio
async def test_run_simple_response():
    runner = _make_runner([_make_response(content="Paris is the capital of France.")])
    result = await runner.run("What is the capital of France?")

    assert isinstance(result, AgentResult)
    assert result.output == "Paris is the capital of France."
    assert result.steps == 1
    assert result.tokens_used == 30  # 10 in + 20 out


@pytest.mark.asyncio
async def test_run_generates_thread_id():
    runner = _make_runner([_make_response(content="hello")])
    result = await runner.run("hi")
    assert len(result.thread_id) == 16
    assert result.thread_id.isalnum()


@pytest.mark.asyncio
async def test_run_uses_provided_thread_id():
    runner = _make_runner([_make_response(content="hi")])
    result = await runner.run("hi", thread_id="my-thread")
    assert result.thread_id == "my-thread"


@pytest.mark.asyncio
async def test_run_history_excludes_system():
    runner = _make_runner([_make_response(content="answer")])
    result = await runner.run("question")
    roles = [m["role"] for m in result.history]
    assert "system" not in roles
    assert "user" in roles
    assert "assistant" in roles


@pytest.mark.asyncio
async def test_run_accumulates_tokens():
    runner = _make_runner([
        _make_response(content="answer", input_tokens=100, output_tokens=50),
    ])
    result = await runner.run("question")
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.tokens_used == 150


@pytest.mark.asyncio
async def test_run_cost_usd_positive():
    runner = _make_runner([
        _make_response(content="ok", input_tokens=1000, output_tokens=500),
    ])
    result = await runner.run("hello")
    assert result.cost_usd > 0


# =============================================================================
# AgentRunner.run() — tool calls
# =============================================================================


@pytest.mark.asyncio
async def test_run_with_tool_call():
    tool_response = _make_response(
        content="",
        tool_calls=[ToolCall(id="tc1", name="get_weather", arguments={"city": "Paris"})],
        stop_reason="tool_use",
    )
    final_response = _make_response(content="Paris is sunny with 22°C.")

    runner = _make_runner(
        responses=[tool_response, final_response],
        tools={"get_weather": "Sunny, 22°C"},
    )
    result = await runner.run("What is the weather in Paris?")
    assert "sunny" in result.output.lower() or result.steps == 2
    assert result.steps == 2


@pytest.mark.asyncio
async def test_run_tool_result_in_history():
    tool_response = _make_response(
        content="",
        tool_calls=[ToolCall(id="tc2", name="lookup", arguments={})],
    )
    final = _make_response(content="Done.")

    runner = _make_runner(
        responses=[tool_response, final],
        tools={"lookup": "found it"},
    )
    result = await runner.run("look something up")
    roles = [m["role"] for m in result.history]
    assert "tool" in roles


@pytest.mark.asyncio
async def test_run_max_turns_reached():
    # Always returns a tool call — never terminates
    endless_response = _make_response(
        content="",
        tool_calls=[ToolCall(id="tc", name="tool", arguments={})],
    )
    runner = _make_runner(
        responses=[endless_response] * 5,  # provide enough for 5 turns
        tools={"tool": "ok"},
        config=RunnerConfig(max_turns=3),
    )
    result = await runner.run("do the thing")
    assert "maximum turn limit" in result.output.lower()
    assert result.steps == 3


@pytest.mark.asyncio
async def test_run_tool_timeout_returns_error_string():
    async def slow_tool(**kwargs: Any) -> str:
        await asyncio.sleep(60)
        return "done"

    from ninetrix.registry import ToolDef
    td = ToolDef(name="slow", description="slow", parameters={"type": "object", "properties": {}}, fn=slow_tool)
    dispatcher = ToolDispatcher([LocalToolSource([td])])

    tool_response = _make_response(
        content="",
        tool_calls=[ToolCall(id="tc", name="slow", arguments={})],
    )
    final = _make_response(content="timed out handled")

    provider = _make_provider([tool_response, final])
    runner = AgentRunner(
        provider=provider,
        dispatcher=dispatcher,
        config=RunnerConfig(tool_timeout=0.01, max_turns=2),  # 10ms timeout
    )
    result = await runner.run("run slow tool")
    # The tool_result message should contain the timeout message
    tool_results = [m for m in result.history if m["role"] == "tool"]
    assert any("timed out" in m["content"] for m in tool_results)


# =============================================================================
# Structured output
# =============================================================================


def test_runner_config_output_type_pydantic():
    """Verify RunnerConfig accepts a Pydantic model type."""
    try:
        from pydantic import BaseModel

        class MyModel(BaseModel):
            name: str
            age: int

        cfg = RunnerConfig(output_type=MyModel)
        schema = _get_output_schema(cfg.output_type)
        assert schema is not None
        assert "properties" in schema
    except ImportError:
        pytest.skip("pydantic not installed")


@pytest.mark.asyncio
async def test_run_structured_output_success():
    try:
        from pydantic import BaseModel

        class CityInfo(BaseModel):
            city: str
            country: str

        json_str = '{"city": "Paris", "country": "France"}'
        runner = _make_runner(
            responses=[_make_response(content=json_str)],
            config=RunnerConfig(output_type=CityInfo),
        )
        result = await runner.run("Tell me about Paris")
        assert hasattr(result.output, "city")
        assert result.output.city == "Paris"
        assert result.output.country == "France"
    except ImportError:
        pytest.skip("pydantic not installed")


@pytest.mark.asyncio
async def test_run_structured_output_retry_succeeds():
    try:
        from pydantic import BaseModel

        class Simple(BaseModel):
            value: int

        bad_json = "not json at all"
        good_json = '{"value": 42}'
        runner = _make_runner(
            responses=[
                _make_response(content=bad_json),   # first LLM call returns bad JSON
                _make_response(content=good_json),   # retry returns good JSON
            ],
            config=RunnerConfig(output_type=Simple, output_retries=1),
        )
        result = await runner.run("give me a value")
        assert result.output.value == 42
    except ImportError:
        pytest.skip("pydantic not installed")


@pytest.mark.asyncio
async def test_run_structured_output_retry_exhausted_raises():
    try:
        from pydantic import BaseModel

        class Simple(BaseModel):
            value: int

        runner = _make_runner(
            responses=[
                _make_response(content="not json"),  # first call
                _make_response(content="still not json"),  # retry call
            ],
            config=RunnerConfig(output_type=Simple, output_retries=1),
        )
        with pytest.raises(OutputParseError, match="Failed to parse"):
            await runner.run("give me a value")
    except ImportError:
        pytest.skip("pydantic not installed")


@pytest.mark.asyncio
async def test_run_structured_output_no_retry_raises_immediately():
    try:
        from pydantic import BaseModel

        class Simple(BaseModel):
            value: int

        runner = _make_runner(
            responses=[_make_response(content="bad json")],
            config=RunnerConfig(output_type=Simple, output_retries=0),
        )
        with pytest.raises(OutputParseError):
            await runner.run("give me a value")
    except ImportError:
        pytest.skip("pydantic not installed")


# =============================================================================
# Checkpointer integration
# =============================================================================


@pytest.mark.asyncio
async def test_run_saves_to_checkpointer():
    checkpointer = MagicMock()
    checkpointer.save = AsyncMock()
    checkpointer.load = AsyncMock(return_value=None)

    runner = _make_runner(
        responses=[_make_response(content="done")],
        checkpointer=checkpointer,
    )
    result = await runner.run("hello", thread_id="t1")
    checkpointer.save.assert_awaited_once()
    call_kwargs = checkpointer.save.call_args.kwargs
    assert call_kwargs["thread_id"] == "t1"
    assert call_kwargs["agent_id"] == "test-agent"
    assert isinstance(call_kwargs["history"], list)


@pytest.mark.asyncio
async def test_run_restores_from_checkpointer():
    prior = [
        {"role": "user", "content": "first message"},
        {"role": "assistant", "content": "first response"},
    ]
    checkpointer = MagicMock()
    checkpointer.load = AsyncMock(return_value={"history": prior})
    checkpointer.save = AsyncMock()

    runner = _make_runner(
        responses=[_make_response(content="continued")],
        checkpointer=checkpointer,
    )
    result = await runner.run("follow-up", thread_id="t2")
    # History should include the restored messages
    contents = [m.get("content") for m in result.history]
    assert "first message" in contents


@pytest.mark.asyncio
async def test_run_prior_history_overrides_checkpointer():
    checkpointer = MagicMock()
    checkpointer.load = AsyncMock(return_value={"history": [
        {"role": "user", "content": "from_checkpointer"}
    ]})
    checkpointer.save = AsyncMock()

    runner = _make_runner(
        responses=[_make_response(content="ok")],
        checkpointer=checkpointer,
    )
    prior = [{"role": "user", "content": "from_prior"}]
    result = await runner.run("message", prior_history=prior)

    contents = [m.get("content") for m in result.history]
    assert "from_prior" in contents
    assert "from_checkpointer" not in contents
    # Checkpointer.load should NOT be called when prior_history is supplied
    checkpointer.load.assert_not_awaited()


# =============================================================================
# Budget enforcement
# =============================================================================


@pytest.mark.asyncio
async def test_run_budget_exceeded_raises():
    from ninetrix._internals.types import BudgetExceededError

    runner = _make_runner(
        responses=[_make_response(content="hi", input_tokens=1_000_000, output_tokens=0)],
        config=RunnerConfig(max_budget_usd=0.001),
    )
    with pytest.raises(BudgetExceededError):
        await runner.run("hello")


# =============================================================================
# _build_assistant_message helper
# =============================================================================


def test_build_assistant_message_no_tools():
    response = LLMResponse(content="Hello", tool_calls=[], stop_reason="end_turn")
    msg = _build_assistant_message(response)
    assert msg["role"] == "assistant"
    assert msg["content"] == "Hello"
    assert msg["tool_calls"] is None


def test_build_assistant_message_with_tools():
    tc = ToolCall(id="id1", name="search", arguments={"q": "python"})
    response = LLMResponse(content="", tool_calls=[tc], stop_reason="tool_use")
    msg = _build_assistant_message(response)
    assert msg["tool_calls"] is not None
    assert len(msg["tool_calls"]) == 1
    tc_payload = msg["tool_calls"][0]
    assert tc_payload["id"] == "id1"
    assert tc_payload["function"]["name"] == "search"
    parsed = json.loads(tc_payload["function"]["arguments"])
    assert parsed["q"] == "python"


# =============================================================================
# _get_output_schema helper
# =============================================================================


def test_get_output_schema_none():
    assert _get_output_schema(None) is None


def test_get_output_schema_no_pydantic():
    assert _get_output_schema(str) is None


def test_get_output_schema_pydantic():
    try:
        from pydantic import BaseModel

        class M(BaseModel):
            x: int

        schema = _get_output_schema(M)
        assert schema is not None
        assert "properties" in schema
    except ImportError:
        pytest.skip("pydantic not installed")


# =============================================================================
# _extract_validation_errors helper
# =============================================================================


def test_extract_validation_errors_generic():
    errors = _extract_validation_errors(ValueError("bad"))
    assert len(errors) == 1
    assert "bad" in errors[0]["msg"]


def test_extract_validation_errors_pydantic():
    try:
        from pydantic import BaseModel, ValidationError

        class M(BaseModel):
            x: int

        try:
            M.model_validate({"x": "not_an_int"})
        except ValidationError as e:
            errors = _extract_validation_errors(e)
            assert len(errors) >= 1
    except ImportError:
        pytest.skip("pydantic not installed")


# =============================================================================
# Budget reset between runs
# =============================================================================


@pytest.mark.asyncio
async def test_budget_resets_between_runs():
    """Each run() call starts with a fresh budget."""
    runner = _make_runner(
        responses=[
            _make_response(content="run1", input_tokens=100, output_tokens=50),
            _make_response(content="run2", input_tokens=100, output_tokens=50),
        ],
    )
    r1 = await runner.run("first")
    r2 = await runner.run("second")
    # Each run should report only its own usage
    assert r1.tokens_used == 150
    assert r2.tokens_used == 150
