"""
agent/config.py — AgentConfig dataclass (pure data, no behavior).

Layer: L8 (agent) — may import L1 (_internals) + stdlib only.

AgentConfig is the single source of truth for all agent parameters.
It is constructed by Agent.__init__() and consumed by Agent._build_runner().
No LLM calls, no network calls, no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ninetrix._internals.types import ExecutionMode, OnStepFailure, Provider


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class MCPConfig:
    """MCP gateway connection settings."""

    gateway_url: str = ""
    token: str = ""
    workspace_id: str = ""


@dataclass
class ThinkingConfig:
    """Extended thinking / chain-of-thought settings."""

    enabled: bool = False
    budget_tokens: int = 10_000


@dataclass
class VerifierConfig:
    """Step-level answer verification settings."""

    enabled: bool = False
    model: str = ""          # defaults to main model if empty


# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Pure data: everything needed to run an agent.

    Constructed by :class:`~ninetrix.agent.agent.Agent`, consumed by
    :class:`~ninetrix.runtime.runner.AgentRunner` via ``_build_runner()``.
    No methods that make network calls or have side effects.

    Example::

        cfg = AgentConfig(
            name="analyst",
            provider="anthropic",
            model="claude-sonnet-4-6",
            role="Senior data analyst",
            goal="Answer questions about company metrics.",
        )
    """

    # Identity
    name: str = "agent"
    role: str = ""
    goal: str = ""
    instructions: str = ""
    constraints: list[str] = field(default_factory=list)
    description: str = ""

    # Runtime
    provider: Provider = "anthropic"
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.0
    max_tokens: int = 8192

    # Tools — three sources
    local_tools: list[Any] = field(default_factory=list)      # @Tool callables / ToolDef
    mcp_tools: list[str] = field(default_factory=list)         # tool names via mcp://
    composio_tools: list[str] = field(default_factory=list)    # app names via composio://

    # Structured output
    output_type: Any = None          # Pydantic BaseModel subclass or None
    output_retries: int = 1

    # Feature flags
    execution_mode: ExecutionMode = "direct"
    max_turns: int = 20
    tool_timeout: float = 30.0
    on_step_failure: OnStepFailure = "continue"

    # Budget
    max_budget_usd: float = 0.0      # 0.0 → unlimited

    # Sub-configs
    thinking: ThinkingConfig = field(default_factory=ThinkingConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)

    # Persistence
    db_url: str = ""

    # Observability / telemetry
    api_url: str = ""
    runner_token: str = ""

    # Credential override (highest priority, skips all resolution)
    api_key: str = ""

    # History sliding window
    history_max_tokens: int = 128_000

    @property
    def system_prompt(self) -> str:
        """Assemble the system prompt from role/goal/instructions/constraints.

        When ``role`` is absent but ``description`` is set, a minimal system
        prompt is derived automatically so routing-only agents still have
        sensible behaviour::

            nx.agent("billing", description="Handles invoices and payments")
            # system_prompt → "You are a helpful assistant. Handles invoices and payments."
        """
        parts: list[str] = []
        if self.role:
            parts.append(f"You are a {self.role}.")
        elif self.description:
            parts.append(f"You are a helpful assistant. {self.description}.")
        if self.goal:
            parts.append(f"Goal: {self.goal}")
        if self.instructions:
            parts.append(f"Instructions:\n{self.instructions.strip()}")
        if self.constraints:
            lines = "\n".join(f"- {c}" for c in self.constraints)
            parts.append(f"Constraints:\n{lines}")
        return "\n\n".join(parts)
