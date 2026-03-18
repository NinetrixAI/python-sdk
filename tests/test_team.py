"""Tests for workflow/team.py — PR 26."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ninetrix import Team, TeamResult
from ninetrix._internals.types import AgentResult, LLMResponse
from ninetrix.workflow.team import _get_agent_description, _get_agent_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_result(output: str = "ok") -> AgentResult:
    return AgentResult(
        output=output,
        tokens_used=10,
        cost_usd=0.01,
        thread_id="t",
        input_tokens=5,
        output_tokens=5,
        steps=1,
    )


def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=[],
        input_tokens=5,
        output_tokens=3,
    )


def _mock_agent(name: str, description: str = "", output: str = "handled") -> MagicMock:
    agent = MagicMock()
    agent.config = MagicMock()
    agent.config.name = name
    agent.config.description = description
    agent.arun = AsyncMock(return_value=_make_agent_result(output))
    return agent


def _mock_protocol_agent(name: str, description: str = "", output: str = "handled") -> MagicMock:
    """Agent that uses .name attribute directly (AgentClient / RemoteAgent style)."""
    agent = MagicMock()
    agent.name = name
    agent.description = description
    del agent.config   # ensure .config doesn't exist
    agent.arun = AsyncMock(return_value=_make_agent_result(output))
    return agent


def _inject_router(team: Team, route_to: str) -> None:
    """Inject a mock router provider that always returns route_to."""
    provider = MagicMock()
    provider.complete = AsyncMock(return_value=_make_llm_response(route_to))
    team.inject_router_provider(provider)


# =============================================================================
# Construction
# =============================================================================


def test_team_constructs():
    a = _mock_agent("a")
    team = Team(agents=[a])
    assert team is not None


def test_team_requires_agents():
    with pytest.raises(ValueError, match="at least one"):
        Team(agents=[])


def test_team_name_default():
    team = Team(agents=[_mock_agent("a")])
    assert team.name == "team"


def test_team_name_custom():
    team = Team(agents=[_mock_agent("a")], name="support")
    assert team.name == "support"


def test_team_agents_dict_keyed_by_name():
    a = _mock_agent("billing")
    b = _mock_agent("support")
    team = Team(agents=[a, b])
    assert "billing" in team._agents
    assert "support" in team._agents


def test_team_protocol_agent_name():
    """Agents with .name attribute (not .config.name) should work."""
    a = _mock_protocol_agent("protocol_agent")
    team = Team(agents=[a])
    assert "protocol_agent" in team._agents


# =============================================================================
# _get_agent_name helper
# =============================================================================


def test_get_agent_name_config():
    agent = MagicMock()
    agent.config = MagicMock()
    agent.config.name = "my_agent"
    assert _get_agent_name(agent) == "my_agent"


def test_get_agent_name_direct():
    agent = MagicMock(spec=[])
    agent.name = "direct_name"
    assert _get_agent_name(agent) == "direct_name"


def test_get_agent_name_fallback():
    agent = object()
    result = _get_agent_name(agent)
    assert isinstance(result, str)
    assert len(result) > 0


# =============================================================================
# _get_agent_description helper
# =============================================================================


def test_get_description_from_config():
    agent = MagicMock()
    agent.config.description = "billing specialist"
    assert _get_agent_description(agent) == "billing specialist"


def test_get_description_from_direct():
    agent = MagicMock(spec=[])
    agent.description = "support agent"
    assert _get_agent_description(agent) == "support agent"


def test_get_description_empty_returns_placeholder():
    agent = MagicMock()
    agent.config.description = ""
    assert _get_agent_description(agent) == "(no description)"


# =============================================================================
# _parse_route
# =============================================================================


def test_parse_route_exact_match():
    team = Team(agents=[_mock_agent("billing"), _mock_agent("support")])
    assert team._parse_route("billing") == "billing"
    assert team._parse_route("support") == "support"


def test_parse_route_case_insensitive():
    team = Team(agents=[_mock_agent("Billing"), _mock_agent("Support")])
    assert team._parse_route("billing") == "Billing"


def test_parse_route_substring():
    team = Team(agents=[_mock_agent("billing"), _mock_agent("support")])
    assert team._parse_route("The billing agent should handle this.") == "billing"


def test_parse_route_fallback_to_first():
    team = Team(agents=[_mock_agent("a"), _mock_agent("b")])
    assert team._parse_route("xyzzy_no_match") == "a"


# =============================================================================
# Team.arun — routing
# =============================================================================


@pytest.mark.asyncio
async def test_team_arun_routes_to_correct_agent():
    billing = _mock_agent("billing", output="refund processed")
    support = _mock_agent("support", output="ticket created")

    team = Team(agents=[billing, support])
    _inject_router(team, "billing")

    result = await team.arun("I was charged twice")
    assert result.routed_to == "billing"
    assert result.output == "refund processed"
    billing.arun.assert_called_once()
    support.arun.assert_not_called()


@pytest.mark.asyncio
async def test_team_arun_routes_support():
    billing = _mock_agent("billing", output="billing handled")
    support = _mock_agent("support", output="support handled")

    team = Team(agents=[billing, support])
    _inject_router(team, "support")

    result = await team.arun("My account is locked")
    assert result.routed_to == "support"
    assert result.output == "support handled"


@pytest.mark.asyncio
async def test_team_arun_returns_team_result():
    a = _mock_agent("agent_a")
    team = Team(agents=[a])
    _inject_router(team, "agent_a")

    result = await team.arun("hello")
    assert isinstance(result, TeamResult)
    assert result.routed_to == "agent_a"
    assert result.agent_result is not None


@pytest.mark.asyncio
async def test_team_arun_assigns_thread_id():
    a = _mock_agent("a")
    team = Team(agents=[a])
    _inject_router(team, "a")

    result = await team.arun("hello")
    assert result.thread_id is not None and len(result.thread_id) > 0


@pytest.mark.asyncio
async def test_team_arun_stable_thread_id():
    a = _mock_agent("a")
    team = Team(agents=[a])
    _inject_router(team, "a")

    result = await team.arun("hello", thread_id="fixed-123")
    assert result.thread_id == "fixed-123"


@pytest.mark.asyncio
async def test_team_arun_passes_thread_id_to_agent():
    a = _mock_agent("a")
    team = Team(agents=[a])
    _inject_router(team, "a")

    await team.arun("hello", thread_id="thread-abc")
    a.arun.assert_called_once_with("hello", thread_id="thread-abc")


# =============================================================================
# Team.run — sync wrapper
# =============================================================================


def test_team_run_sync():
    a = _mock_agent("a", output="sync_ok")
    team = Team(agents=[a])
    _inject_router(team, "a")

    result = team.run("hello")
    assert result.output == "sync_ok"
    assert result.routed_to == "a"


# =============================================================================
# Router fallback on provider error
# =============================================================================


@pytest.mark.asyncio
async def test_team_router_error_falls_back_to_first_agent():
    a = _mock_agent("a", output="fallback result")
    b = _mock_agent("b")
    team = Team(agents=[a, b])

    # Provider raises — should fall back to first agent silently
    provider = MagicMock()
    provider.complete = AsyncMock(side_effect=RuntimeError("LLM down"))
    team.inject_router_provider(provider)

    result = await team.arun("hello")
    assert result.routed_to == "a"
    assert result.output == "fallback result"


# =============================================================================
# Router uses low temperature
# =============================================================================


@pytest.mark.asyncio
async def test_team_router_uses_low_temperature():
    a = _mock_agent("a")
    team = Team(agents=[a])

    provider = MagicMock()
    provider.complete = AsyncMock(return_value=_make_llm_response("a"))
    team.inject_router_provider(provider)

    await team.arun("hello")
    call_kwargs = provider.complete.call_args[1]
    pconfig = call_kwargs.get("config")
    assert pconfig is not None
    assert pconfig.temperature == 0.0


# =============================================================================
# Router prompt includes agent names and descriptions
# =============================================================================


@pytest.mark.asyncio
async def test_team_router_prompt_includes_agent_names():
    billing = _mock_agent("billing", description="handles invoices")
    support = _mock_agent("support", description="handles tickets")
    team = Team(agents=[billing, support])

    captured: list[list[dict]] = []

    async def fake_complete(messages, tools, **kwargs):
        captured.append(messages)
        return _make_llm_response("billing")

    provider = MagicMock()
    provider.complete = fake_complete
    team.inject_router_provider(provider)

    await team.arun("I have a question about my invoice")
    assert captured
    prompt_text = captured[0][0]["content"]
    assert "billing" in prompt_text
    assert "support" in prompt_text


@pytest.mark.asyncio
async def test_team_router_prompt_includes_descriptions():
    billing = _mock_agent("billing", description="handles invoices and refunds")
    team = Team(agents=[billing])

    captured: list[list[dict]] = []

    async def fake_complete(messages, tools, **kwargs):
        captured.append(messages)
        return _make_llm_response("billing")

    provider = MagicMock()
    provider.complete = fake_complete
    team.inject_router_provider(provider)

    await team.arun("test")
    prompt_text = captured[0][0]["content"]
    assert "handles invoices and refunds" in prompt_text


# =============================================================================
# Protocol agent compatibility (AgentClient / RemoteAgent style)
# =============================================================================


@pytest.mark.asyncio
async def test_team_accepts_protocol_agents():
    """Agents with .name (not .config.name) should work."""
    a = _mock_protocol_agent("remote_a", output="remote result")
    team = Team(agents=[a])
    _inject_router(team, "remote_a")

    result = await team.arun("test")
    assert result.routed_to == "remote_a"
    assert result.output == "remote result"


@pytest.mark.asyncio
async def test_team_mixed_agent_types():
    """Mix of Agent-style and AgentClient-style objects."""
    local = _mock_agent("local", output="local response")
    remote = _mock_protocol_agent("remote", output="remote response")

    team = Team(agents=[local, remote])
    _inject_router(team, "remote")

    result = await team.arun("hello")
    assert result.routed_to == "remote"
    local.arun.assert_not_called()
    remote.arun.assert_called_once()


# =============================================================================
# Public imports
# =============================================================================


def test_team_importable_from_ninetrix():
    from ninetrix import Team
    assert Team is not None


def test_team_result_importable():
    from ninetrix import TeamResult
    assert TeamResult is not None


def test_team_result_fields():
    r = TeamResult(
        output="hello",
        routed_to="agent_a",
        agent_result=_make_agent_result("hello"),
        thread_id="t1",
    )
    assert r.output == "hello"
    assert r.routed_to == "agent_a"
    assert r.thread_id == "t1"
