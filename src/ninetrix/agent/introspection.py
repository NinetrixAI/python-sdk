"""
agent/introspection.py — read-only views of agent configuration and health.

Layer: L8 (agent) — may import L1 (_internals) + stdlib only.

These dataclasses are returned by Agent.info(), Agent.validate(), and
Agent.dry_run(). They are pure data — no network calls, no LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AgentInfo:
    """Static summary of an agent's current configuration.

    Returned by :meth:`~ninetrix.agent.agent.Agent.info`. No network calls.

    Example::

        info = agent.info()
        print(info.name, info.model, info.tool_count)
    """

    name: str
    provider: str
    model: str
    tool_count: int
    local_tools: list[str] = field(default_factory=list)
    mcp_tools: list[str] = field(default_factory=list)
    composio_tools: list[str] = field(default_factory=list)
    has_persistence: bool = False
    execution_mode: str = "direct"
    max_budget_usd: float = 0.0
    max_turns: int = 20
    system_prompt_chars: int = 0


@dataclass
class ValidationIssue:
    """A single problem or warning found by :meth:`~ninetrix.agent.agent.Agent.validate`.

    Attributes:
        level:    ``"error"`` blocks runs; ``"warning"`` is advisory only.
        code:     Machine-readable identifier (e.g. ``"AUTH_MISSING"``).
        message:  Human-readable description of the problem.
        fix:      Actionable suggestion for resolving the issue.
    """

    level: Literal["error", "warning"]
    code: str
    message: str
    fix: str


@dataclass
class DryRunResult:
    """Result of :meth:`~ninetrix.agent.agent.Agent.dry_run`.

    Contains a pre-flight summary without making any LLM calls. Useful for
    CI checks, cost estimation, and debugging configuration problems.

    Attributes:
        tools_available:       Names of all tools the dispatcher found.
        estimated_turns:       Rough upper bound on turns (min of max_turns, tool_count+1).
        estimated_cost_usd:    Rough cost estimate based on prompt size and model pricing.
        system_prompt_chars:   Character count of the resolved system prompt.
        warnings:              All :class:`ValidationIssue` items found (errors + warnings).
        provider_reachable:    False if any AUTH_* error was found.
        mcp_gateway_reachable: False if MCP_UNREACHABLE error was found.
        db_reachable:          False if DB_UNREACHABLE error was found.
    """

    tools_available: list[str] = field(default_factory=list)
    estimated_turns: int = 1
    estimated_cost_usd: float = 0.0
    system_prompt_chars: int = 0
    warnings: list[ValidationIssue] = field(default_factory=list)
    provider_reachable: bool = True
    mcp_gateway_reachable: bool = True
    db_reachable: bool = True

    @property
    def has_errors(self) -> bool:
        """True if any issue with level == ``"error"`` is present."""
        return any(w.level == "error" for w in self.warnings)
