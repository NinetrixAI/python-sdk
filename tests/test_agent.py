"""Tests for agent/config.py + agent/agent.py + agent/introspection.py — PR 18."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ninetrix.agent.config import AgentConfig, MCPConfig, ThinkingConfig, VerifierConfig
from ninetrix.agent.agent import Agent, _run_in_thread
from ninetrix.agent.introspection import AgentInfo, DryRunResult, ValidationIssue
from ninetrix._internals.types import AgentResult


# =============================================================================
# AgentConfig
# =============================================================================


def test_agentconfig_defaults():
    cfg = AgentConfig()
    assert cfg.name == "agent"
    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-sonnet-4-6"
    assert cfg.temperature == 0.0
    assert cfg.max_turns == 20
    assert cfg.max_budget_usd == 0.0
    assert cfg.local_tools == []
    assert cfg.mcp_tools == []
    assert cfg.composio_tools == []
    assert cfg.output_type is None


def test_agentconfig_system_prompt_empty_by_default():
    cfg = AgentConfig()
    assert cfg.system_prompt == ""


def test_agentconfig_system_prompt_role_only():
    cfg = AgentConfig(role="data analyst")
    assert "You are a data analyst." in cfg.system_prompt


def test_agentconfig_system_prompt_goal_only():
    cfg = AgentConfig(goal="Answer metric questions.")
    assert "Goal: Answer metric questions." in cfg.system_prompt


def test_agentconfig_system_prompt_all_parts():
    cfg = AgentConfig(
        role="analyst",
        goal="answer questions",
        instructions="Be concise.",
        constraints=["No hallucination", "Cite sources"],
    )
    prompt = cfg.system_prompt
    assert "You are a analyst." in prompt
    assert "Goal: answer questions" in prompt
    assert "Instructions:" in prompt
    assert "Be concise." in prompt
    assert "Constraints:" in prompt
    assert "- No hallucination" in prompt
    assert "- Cite sources" in prompt


def test_agentconfig_sub_configs_have_defaults():
    cfg = AgentConfig()
    assert isinstance(cfg.mcp, MCPConfig)
    assert isinstance(cfg.thinking, ThinkingConfig)
    assert isinstance(cfg.verifier, VerifierConfig)


# =============================================================================
# Introspection dataclasses
# =============================================================================


def test_agentinfo_fields():
    info = AgentInfo(name="bot", provider="anthropic", model="claude-sonnet-4-6", tool_count=3)
    assert info.name == "bot"
    assert info.tool_count == 3
    assert info.local_tools == []


def test_validationissue_fields():
    issue = ValidationIssue(
        level="error", code="AUTH_MISSING",
        message="No key", fix="Set ANTHROPIC_API_KEY"
    )
    assert issue.level == "error"
    assert issue.code == "AUTH_MISSING"


def test_dryrunresult_has_errors_false():
    result = DryRunResult(warnings=[
        ValidationIssue(level="warning", code="W1", message="m", fix="f")
    ])
    assert not result.has_errors


def test_dryrunresult_has_errors_true():
    result = DryRunResult(warnings=[
        ValidationIssue(level="error", code="AUTH_MISSING", message="m", fix="f")
    ])
    assert result.has_errors


# =============================================================================
# Agent construction
# =============================================================================


def test_agent_default_construction():
    a = Agent()
    assert a.name == "agent"
    assert a.config.provider == "anthropic"
    assert a.config.model == "claude-sonnet-4-6"


def test_agent_kwargs():
    a = Agent(
        name="my-bot",
        provider="openai",
        model="gpt-4o",
        role="researcher",
        max_turns=5,
        max_budget_usd=1.0,
    )
    assert a.name == "my-bot"
    assert a.config.provider == "openai"
    assert a.config.max_turns == 5
    assert a.config.max_budget_usd == 1.0
    assert "You are a researcher." in a.config.system_prompt


def test_agent_repr():
    a = Agent(name="bot", provider="anthropic", model="claude-haiku-4-5-20251001")
    r = repr(a)
    assert "Agent" in r
    assert "bot" in r
    assert "anthropic" in r


def test_agent_is_generic_subclass():
    from ninetrix.agent.agent import Agent
    assert issubclass(Agent, Agent)


# =============================================================================
# Agent.info()
# =============================================================================


def test_agent_info_no_tools():
    a = Agent(name="bot", provider="anthropic", model="claude-sonnet-4-6")
    info = a.info()
    assert info.name == "bot"
    assert info.provider == "anthropic"
    assert info.tool_count == 0
    assert not info.has_persistence


def test_agent_info_with_tools():
    from ninetrix.tool import Tool

    @Tool
    def my_tool(x: int) -> int:
        """A tool."""
        return x

    a = Agent(tools=[my_tool])
    info = a.info()
    assert info.tool_count == 1


def test_agent_info_with_mcp_tools():
    a = Agent(mcp_tools=["github", "filesystem"])
    info = a.info()
    assert info.tool_count == 2
    assert "github" in info.mcp_tools


def test_agent_info_has_persistence_db_url():
    a = Agent(db_url="postgresql://localhost/test")
    info = a.info()
    assert info.has_persistence


def test_agent_info_system_prompt_chars():
    a = Agent(role="analyst", goal="answer questions")
    info = a.info()
    assert info.system_prompt_chars > 0


# =============================================================================
# Agent.validate()
# =============================================================================


@pytest.mark.asyncio
async def test_validate_returns_error_when_no_api_key():
    a = Agent(provider="anthropic")
    # Patch CredentialStore.resolve to return None
    with patch("ninetrix._internals.auth.CredentialStore") as MockStore:
        instance = MockStore.return_value
        instance.resolve.return_value = None
        issues = await a.validate()

    errors = [i for i in issues if i.level == "error"]
    assert any(i.code == "AUTH_MISSING" for i in errors)


@pytest.mark.asyncio
async def test_validate_returns_empty_when_key_present():
    a = Agent(provider="anthropic")
    with patch("ninetrix._internals.auth.CredentialStore") as MockStore:
        instance = MockStore.return_value
        instance.resolve.return_value = "sk-ant-test"
        issues = await a.validate()

    # Only possible remaining issue is NO_TOOLS_OR_INSTRUCTIONS (warning)
    errors = [i for i in issues if i.level == "error"]
    assert errors == []


@pytest.mark.asyncio
async def test_validate_warns_no_tools_or_instructions():
    a = Agent()  # no tools, no role/goal/instructions
    with patch("ninetrix._internals.auth.CredentialStore") as MockStore:
        instance = MockStore.return_value
        instance.resolve.return_value = "sk-ant-test"
        issues = await a.validate()

    codes = [i.code for i in issues]
    assert "NO_TOOLS_OR_INSTRUCTIONS" in codes


@pytest.mark.asyncio
async def test_validate_no_warning_when_role_set():
    a = Agent(role="analyst")
    with patch("ninetrix._internals.auth.CredentialStore") as MockStore:
        instance = MockStore.return_value
        instance.resolve.return_value = "sk-ant-test"
        issues = await a.validate()

    codes = [i.code for i in issues]
    assert "NO_TOOLS_OR_INSTRUCTIONS" not in codes


@pytest.mark.asyncio
async def test_validate_error_on_negative_budget():
    a = Agent(max_budget_usd=-1.0)
    with patch("ninetrix._internals.auth.CredentialStore") as MockStore:
        instance = MockStore.return_value
        instance.resolve.return_value = "sk-ant-test"
        issues = await a.validate()

    codes = [i.code for i in issues]
    assert "INVALID_BUDGET" in codes


# =============================================================================
# Agent.dry_run()
# =============================================================================


@pytest.mark.asyncio
async def test_dry_run_returns_dryrunresult():
    a = Agent(role="bot")
    with patch("ninetrix._internals.auth.CredentialStore") as MockStore:
        instance = MockStore.return_value
        instance.resolve.return_value = "sk-ant-test"
        result = await a.dry_run("hello")

    assert isinstance(result, DryRunResult)
    assert result.system_prompt_chars > 0


@pytest.mark.asyncio
async def test_dry_run_tools_available():
    from ninetrix.tool import Tool

    @Tool
    def search(query: str) -> str:
        """Search the web."""
        return ""

    a = Agent(tools=[search], role="bot")
    with patch("ninetrix._internals.auth.CredentialStore") as MockStore:
        instance = MockStore.return_value
        instance.resolve.return_value = "sk-ant-test"
        result = await a.dry_run("find info")

    assert "search" in result.tools_available


@pytest.mark.asyncio
async def test_dry_run_estimated_cost_positive():
    a = Agent(role="researcher")
    with patch("ninetrix._internals.auth.CredentialStore") as MockStore:
        instance = MockStore.return_value
        instance.resolve.return_value = "sk-ant-test"
        result = await a.dry_run("a" * 4000)  # 4000 chars → ~1000 tokens

    assert result.estimated_cost_usd > 0.0


@pytest.mark.asyncio
async def test_dry_run_provider_not_reachable_when_no_key():
    a = Agent()
    with patch("ninetrix._internals.auth.CredentialStore") as MockStore:
        instance = MockStore.return_value
        instance.resolve.return_value = None
        result = await a.dry_run("hello")

    assert not result.provider_reachable


# =============================================================================
# Agent._build_runner() + invalidate_runner()
# =============================================================================


@pytest.mark.asyncio
async def test_build_runner_caches_result():
    """_build_runner called twice returns same runner instance."""
    a = Agent(provider="anthropic", role="bot")

    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock()

    with patch("ninetrix._internals.auth.CredentialStore") as MockStore, \
         patch("ninetrix.agent.agent.Agent._build_provider", return_value=mock_provider):
        instance = MockStore.return_value
        instance.resolve.return_value = "sk-ant-test"
        instance.resolve_workspace_token.return_value = None

        r1 = await a._get_runner()
        r2 = await a._get_runner()

    assert r1 is r2


@pytest.mark.asyncio
async def test_invalidate_runner_clears_cache():
    a = Agent(provider="anthropic", role="bot")

    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock()

    with patch("ninetrix._internals.auth.CredentialStore") as MockStore, \
         patch("ninetrix.agent.agent.Agent._build_provider", return_value=mock_provider):
        instance = MockStore.return_value
        instance.resolve.return_value = "sk-ant-test"
        instance.resolve_workspace_token.return_value = None

        r1 = await a._get_runner()
        a.invalidate_runner()
        r2 = await a._get_runner()

    assert r1 is not r2


@pytest.mark.asyncio
async def test_build_runner_raises_credential_error_when_no_key():
    from ninetrix._internals.types import CredentialError

    a = Agent(provider="anthropic")
    with patch("ninetrix._internals.auth.CredentialStore") as MockStore:
        instance = MockStore.return_value
        instance.resolve.return_value = None

        with pytest.raises(CredentialError):
            await a._build_runner()


# =============================================================================
# Agent.run() + Agent.arun() — integration with mocked provider
# =============================================================================


def _make_mock_agent(*, system_prompt: str = "You are helpful.") -> Agent:
    """Create an Agent pre-wired with a fake provider."""
    a = Agent(provider="anthropic", role="helpful assistant")
    return a


@pytest.mark.asyncio
async def test_arun_returns_agent_result():
    from ninetrix._internals.types import LLMResponse
    from ninetrix.runtime.dispatcher import ToolDispatcher
    from ninetrix.runtime.runner import AgentRunner, RunnerConfig
    from ninetrix.checkpoint.memory import InMemoryCheckpointer

    provider = MagicMock()
    provider.complete = AsyncMock(return_value=LLMResponse(
        content="42", tool_calls=[], input_tokens=5, output_tokens=3,
    ))

    a = Agent(provider="anthropic", role="bot")
    runner = AgentRunner(
        provider=provider,
        dispatcher=ToolDispatcher([]),
        config=RunnerConfig(name="bot"),
        checkpointer=InMemoryCheckpointer(),
    )
    # Inject runner directly (bypass _build_runner)
    a._runner = runner

    result = await a.arun("What is 6 × 7?")
    assert isinstance(result, AgentResult)
    assert result.output == "42"


def test_run_sync_returns_agent_result():
    from ninetrix._internals.types import LLMResponse
    from ninetrix.runtime.dispatcher import ToolDispatcher
    from ninetrix.runtime.runner import AgentRunner, RunnerConfig
    from ninetrix.checkpoint.memory import InMemoryCheckpointer

    provider = MagicMock()
    provider.complete = AsyncMock(return_value=LLMResponse(
        content="hello", tool_calls=[], input_tokens=3, output_tokens=2,
    ))

    a = Agent(provider="anthropic", role="bot")
    runner = AgentRunner(
        provider=provider,
        dispatcher=ToolDispatcher([]),
        config=RunnerConfig(name="bot"),
        checkpointer=InMemoryCheckpointer(),
    )
    a._runner = runner

    result = a.run("say hello")
    assert isinstance(result, AgentResult)
    assert result.output == "hello"


# =============================================================================
# _run_in_thread helper
# =============================================================================


def test_run_in_thread_returns_value():
    async def coro():
        return 99

    assert _run_in_thread(coro()) == 99


def test_run_in_thread_propagates_exception():
    async def coro():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        _run_in_thread(coro())


# =============================================================================
# Public imports
# =============================================================================


def test_agent_importable_from_ninetrix():
    from ninetrix import Agent
    assert Agent is not None


def test_agentconfig_importable_from_ninetrix():
    from ninetrix import AgentConfig
    assert AgentConfig is not None


def test_introspection_types_importable_from_ninetrix():
    from ninetrix import AgentInfo, ValidationIssue, DryRunResult
    assert AgentInfo is not None
    assert ValidationIssue is not None
    assert DryRunResult is not None
