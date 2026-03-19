"""
workflow/team.py — LLM-based dynamic routing Team.

Layer: L8 (workflow) — may import L1 (_internals) + L4 (providers) + stdlib.

Unlike ``@Workflow`` (hard-wired transitions), ``Team`` uses a cheap LLM call to
decide which agent handles each incoming message at runtime.  The router reads
all agent names + descriptions and picks the best match.

Usage::

    from ninetrix import Agent, Team

    billing = Agent(name="billing", provider="anthropic",
                    description="Handles billing, charges, invoices, refunds.")
    support = Agent(name="support", provider="anthropic",
                    description="Handles technical issues and account problems.")

    team = Team(agents=[billing, support],
                router_model="claude-haiku-4-5-20251001")

    result = team.run("I was charged twice for my subscription")
    print(result.output)
    print(result.routed_to)   # "billing"
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from typing import Any

from ninetrix._internals.types import AgentResult, ProviderConfig


# ---------------------------------------------------------------------------
# TeamResult
# ---------------------------------------------------------------------------


@dataclass
class TeamResult:
    """Result of a :class:`Team` run.

    Attributes:
        output:       The agent's final response text.
        routed_to:    Name of the agent that handled the request.
        agent_name:   Alias for ``routed_to`` — preferred in v2 code.
        agent_result: Full :class:`~ninetrix.AgentResult` from the chosen agent.
        thread_id:    The thread identifier used for this run.
    """
    output: str
    routed_to: str
    agent_result: AgentResult
    thread_id: str = ""

    @property
    def agent_name(self) -> str:
        """Which agent handled this request (alias for ``routed_to``)."""
        return self.routed_to


# ---------------------------------------------------------------------------
# Team
# ---------------------------------------------------------------------------


class Team:
    """Dynamic LLM-based routing across a list of agents.

    ``Team`` accepts any object that satisfies :class:`~ninetrix.AgentProtocol`
    — local ``Agent``, ``AgentClient``, or ``RemoteAgent``.

    The router makes a single, cheap LLM call to classify the incoming message
    and pick the most suitable agent.  Routing temperature is always ``0.0``
    for determinism.

    Args:
        agents:          List of agents.  All must expose a ``name`` attribute
                         and optionally a ``description`` attribute.
        router_model:    Model used for the routing call
                         (default: ``"claude-haiku-4-5-20251001"``).
        router_provider: Provider for the routing call (default: ``"anthropic"``).
        name:            Human-readable label for this team.
        description:     Optional description of the team's purpose.
        router_api_key:  API key override for the router.  Falls back to the
                         standard env var for *router_provider*.

    Example::

        team = Team(
            agents=[billing_agent, support_agent],
            router_model="claude-haiku-4-5-20251001",
        )
        result = team.run("Why was I charged $30 last week?")
        assert result.routed_to == "billing"
    """

    def __init__(
        self,
        *,
        agents: list[Any],
        router_model: str = "claude-haiku-4-5-20251001",
        router_provider: str = "anthropic",
        name: str = "team",
        description: str = "",
        router_api_key: str = "",
    ) -> None:
        if not agents:
            raise ValueError(
                "Team requires at least one agent.\n"
                "  Fix: pass agents=[agent1, agent2, ...]"
            )

        self.name = name
        self.description = description
        self._agents: dict[str, Any] = {}
        for agent in agents:
            agent_name = _get_agent_name(agent)
            self._agents[agent_name] = agent

        self._router_model = router_model
        self._router_provider = router_provider
        self._router_api_key = router_api_key
        self._router_provider_instance: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self, message: str, *, thread_id: str | None = None
    ) -> TeamResult:
        """Route *message* synchronously and return the result.

        Spawns a new event loop via :func:`asyncio.run`.  Not safe to call
        from inside an async context — use :meth:`arun` instead.
        """
        return asyncio.run(self.arun(message, thread_id=thread_id))

    async def arun(
        self, message: str, *, thread_id: str | None = None
    ) -> TeamResult:
        """Route *message* and delegate to the chosen agent.

        Args:
            message:   The user message to route.
            thread_id: Optional stable identifier forwarded to the chosen agent.

        Returns:
            :class:`TeamResult` with ``output``, ``routed_to``, and
            ``agent_result``.
        """
        thread_id = thread_id or uuid.uuid4().hex[:16]

        routed_to = await self._route(message)
        agent = self._agents[routed_to]
        agent_result = await agent.arun(message, thread_id=thread_id)

        return TeamResult(
            output=agent_result.output,
            routed_to=routed_to,
            agent_result=agent_result,
            thread_id=thread_id,
        )

    # ------------------------------------------------------------------
    # Routing internals
    # ------------------------------------------------------------------

    async def _route(self, message: str) -> str:
        """Classify *message* and return the name of the best-matching agent."""
        agent_list = "\n".join(
            f"- {name}: {_get_agent_description(self._agents[name])}"
            for name in self._agents
        )
        agent_names = ", ".join(self._agents.keys())

        prompt = (
            f"You are a routing assistant.  "
            f"Given the following agents and a user message, "
            f"reply with ONLY the name of the most suitable agent.  "
            f"The name must be one of: {agent_names}.\n\n"
            f"Agents:\n{agent_list}\n\n"
            f"User message: {message}\n\n"
            f"Agent name:"
        )

        provider = self._get_router_provider()
        pconfig = ProviderConfig(temperature=0.0, max_tokens=32)
        try:
            response = await provider.complete(
                [{"role": "user", "content": prompt}],
                [],
                config=pconfig,
            )
            raw = (response.content or "").strip()
            return self._parse_route(raw)
        except Exception:
            # Fallback: first agent
            return next(iter(self._agents))

    def _parse_route(self, raw: str) -> str:
        """Extract a valid agent name from the router's text response."""
        # Exact match
        if raw in self._agents:
            return raw
        # Case-insensitive match
        raw_lower = raw.lower()
        for name in self._agents:
            if name.lower() == raw_lower:
                return name
        # Substring match (model may add punctuation or extra words)
        for name in self._agents:
            if name.lower() in raw_lower:
                return name
        # Fallback: first agent
        return next(iter(self._agents))

    def _get_router_provider(self) -> Any:
        """Lazily build and cache the routing provider."""
        if self._router_provider_instance is not None:
            return self._router_provider_instance

        api_key = self._router_api_key or ""
        provider = self._router_provider
        model = self._router_model

        if provider == "anthropic":
            from ninetrix.providers.anthropic import AnthropicAdapter
            self._router_provider_instance = AnthropicAdapter(
                api_key=api_key, model=model
            )
        elif provider == "openai":
            from ninetrix.providers.openai import OpenAIAdapter
            self._router_provider_instance = OpenAIAdapter(
                api_key=api_key, model=model
            )
        elif provider == "google":
            from ninetrix.providers.google import GoogleAdapter
            self._router_provider_instance = GoogleAdapter(
                api_key=api_key, model=model
            )
        elif provider == "litellm":
            from ninetrix.providers.litellm import LiteLLMAdapter
            self._router_provider_instance = LiteLLMAdapter(
                api_key=api_key, model=model
            )
        else:
            from ninetrix._internals.types import ConfigurationError
            raise ConfigurationError(
                f"Unknown router_provider '{provider}'.\n"
                f"  Fix: use one of: 'anthropic', 'openai', 'google', 'litellm'."
            )

        return self._router_provider_instance

    def inject_router_provider(self, provider: Any) -> None:
        """Inject a pre-built provider (used in tests to avoid real API calls)."""
        self._router_provider_instance = provider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_agent_name(agent: Any) -> str:
    """Extract agent name using duck typing."""
    # Agent has config.name; AgentClient / RemoteAgent have .name
    if hasattr(agent, "config") and hasattr(agent.config, "name"):
        return agent.config.name
    if hasattr(agent, "name") and isinstance(agent.name, str):
        return agent.name
    return repr(agent)


def _get_agent_description(agent: Any) -> str:
    """Extract agent description using duck typing."""
    if hasattr(agent, "config") and hasattr(agent.config, "description"):
        return agent.config.description or "(no description)"
    if hasattr(agent, "description") and isinstance(agent.description, str):
        return agent.description or "(no description)"
    return "(no description)"
