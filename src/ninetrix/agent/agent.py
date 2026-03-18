"""
agent/agent.py — the Agent class: the primary user-facing entry point.

Layer: L8 (agent) — may import all lower layers (L1–L7) + stdlib.

Agent is a thin orchestration shell.  It:
  1. Stores configuration in an AgentConfig.
  2. Lazily assembles the full runtime stack via _build_runner().
  3. Exposes run() (sync, event-loop-safe) and arun() (native async).
  4. Exposes info(), validate(), dry_run() for pre-flight inspection.

The runner is cached after first build.  Call invalidate_runner() if you
need to swap configuration (e.g., rotate credentials, add tools).
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import Any, Generic, TypeVar

from ninetrix._internals.types import (
    AgentResult,
    CredentialError,
    Provider,
    T_Output,
)
from ninetrix.agent.config import AgentConfig, MCPConfig, ThinkingConfig, VerifierConfig
from ninetrix.agent.introspection import AgentInfo, DryRunResult, ValidationIssue
from ninetrix.observability.hooks import HooksMixin


# ---------------------------------------------------------------------------
# Event-loop helpers
# ---------------------------------------------------------------------------


def _run_in_thread(coro: Any) -> Any:
    """Run *coro* in a new daemon thread with its own event loop.

    Safe in ALL contexts — plain scripts, Jupyter, pytest-asyncio — because
    it never touches the calling thread's event loop.

    Returns the coroutine result or re-raises its exception.
    """
    result_future: Future[Any] = Future()

    def _target() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_future.set_result(loop.run_until_complete(coro))
        except Exception as exc:
            result_future.set_exception(exc)
        finally:
            loop.close()

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join()
    return result_future.result()  # re-raises stored exceptions


# ---------------------------------------------------------------------------
# Rough token-price table for dry_run cost estimates
# ---------------------------------------------------------------------------

_ESTIMATE_PRICE_PER_1K: dict[str, float] = {
    "claude-opus-4-6":              0.015,
    "claude-sonnet-4-6":            0.003,
    "claude-haiku-4-5-20251001":    0.00025,
    "gpt-4o":                       0.005,
    "gpt-4o-mini":                  0.00015,
    "gemini-1.5-pro":               0.0035,
    "gemini-1.5-flash":             0.000075,
}


def _estimate_price(model: str) -> float:
    """Return price-per-1k-tokens for *model* (best-effort prefix match)."""
    for key, price in _ESTIMATE_PRICE_PER_1K.items():
        if model.startswith(key) or key.startswith(model):
            return price
    return 0.003  # safe default


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent(HooksMixin, Generic[T_Output]):
    """The primary user-facing entry point for running AI agents.

    ``Agent`` is a thin shell that holds :class:`~ninetrix.agent.config.AgentConfig`
    and lazily assembles the full runtime stack on first ``run()`` / ``arun()``
    call via :meth:`_build_runner`.

    Args:
        name:          Agent name (used in checkpointer keys + tool context).
        provider:      LLM provider (``"anthropic"``, ``"openai"``, ``"google"``…).
        model:         Model identifier (e.g. ``"claude-sonnet-4-6"``).
        api_key:       Provider API key. If omitted, resolved from environment.
        role:          Short role description injected into system prompt.
        goal:          Goal injected into system prompt.
        instructions:  Detailed instructions injected into system prompt.
        constraints:   Bullet-point constraints injected into system prompt.
        tools:         Local ``@Tool`` callables (or :class:`~ninetrix.registry.ToolDef`
                       instances) registered with the dispatcher.
        mcp_tools:     Tool names to load from the MCP gateway.
        composio_tools: Composio app names to load.
        output_type:   Pydantic ``BaseModel`` subclass. When set, the runner
                       parses the LLM response into a validated model and
                       ``result.output`` is that model instead of a plain string.
        output_retries: Number of correction attempts for structured output.
        max_turns:     Maximum LLM turns before aborting (default 20).
        max_budget_usd: Hard USD ceiling; 0.0 → unlimited.
        temperature:   Sampling temperature (default 0.0).
        max_tokens:    Maximum completion tokens per LLM call.
        tool_timeout:  Seconds before a tool call is cancelled (default 30.0).
        db_url:        PostgreSQL URL for persistent checkpointing.
        api_url:       Ninetrix API URL for telemetry checkpointing.
        runner_token:  Auth token for *api_url*.
        history_max_tokens: Token budget for the sliding message window.

    Example::

        from ninetrix import Agent

        agent = Agent(
            provider="anthropic",
            model="claude-sonnet-4-6",
            role="Senior data analyst",
            goal="Answer questions about company metrics.",
        )
        result = agent.run("What were our Q4 revenue figures?")
        print(result.output)
    """

    def __init__(
        self,
        *,
        name: str = "agent",
        provider: Provider = "anthropic",
        model: str = "claude-sonnet-4-6",
        api_key: str = "",
        role: str = "",
        goal: str = "",
        instructions: str = "",
        constraints: list[str] | None = None,
        description: str = "",
        tools: list[Any] | None = None,
        mcp_tools: list[str] | None = None,
        composio_tools: list[str] | None = None,
        output_type: Any = None,
        output_retries: int = 1,
        max_turns: int = 20,
        execution_mode: str = "direct",
        max_budget_usd: float = 0.0,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        tool_timeout: float = 30.0,
        db_url: str = "",
        api_url: str = "",
        runner_token: str = "",
        history_max_tokens: int = 128_000,
    ) -> None:
        super().__init__()  # initialises HooksMixin._event_bus
        self.config = AgentConfig(
            name=name,
            provider=provider,
            model=model,
            api_key=api_key,
            role=role,
            goal=goal,
            instructions=instructions,
            constraints=constraints or [],
            description=description,
            local_tools=tools or [],
            mcp_tools=mcp_tools or [],
            composio_tools=composio_tools or [],
            output_type=output_type,
            output_retries=output_retries,
            max_turns=max_turns,
            execution_mode=execution_mode,
            max_budget_usd=max_budget_usd,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_timeout=tool_timeout,
            db_url=db_url,
            api_url=api_url,
            runner_token=runner_token,
            history_max_tokens=history_max_tokens,
        )
        self._runner: Any = None  # AgentRunner | None — lazily built

    # ------------------------------------------------------------------
    # Public API — name property (satisfies AgentProtocol)
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Agent name (from config)."""
        return self.config.name

    # ------------------------------------------------------------------
    # Sync + async run
    # ------------------------------------------------------------------

    def run(
        self,
        message: str,
        *,
        thread_id: str | None = None,
    ) -> AgentResult:
        """Run synchronously.  Safe in **any** context.

        Uses a background thread + dedicated event loop so it never
        deadlocks in Jupyter notebooks or pytest-asyncio tests.

        Note:
            From async code, prefer ``await agent.arun(message)`` — it avoids
            a thread hop and is slightly faster.

        Args:
            message:   User message to send.
            thread_id: Conversation ID for multi-turn resumption.  Auto-generated
                       if not supplied.

        Returns:
            :class:`~ninetrix._internals.types.AgentResult`
        """
        coro = self.arun(message, thread_id=thread_id)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Already inside an event loop — background thread to avoid deadlock.
            return _run_in_thread(coro)
        return asyncio.run(coro)

    async def arun(
        self,
        message: str,
        *,
        thread_id: str | None = None,
    ) -> AgentResult:
        """Run asynchronously (native coroutine).

        Preferred from async code (FastAPI handlers, notebooks, workflows).

        Args:
            message:   User message to send.
            thread_id: Conversation ID for multi-turn resumption.

        Returns:
            :class:`~ninetrix._internals.types.AgentResult`
        """
        runner = await self._get_runner()
        return await runner.run(message, thread_id=thread_id)

    async def stream(
        self,
        message: str,
        *,
        thread_id: Optional[str] = None,
        attachments: Optional[list] = None,
        tool_context: Optional[dict[str, Any]] = None,
    ):
        """Run the agent and yield :class:`~ninetrix.StreamEvent` objects.

        Returns an async generator that yields events at each lifecycle point:
        ``token``, ``tool_start``, ``tool_end``, ``turn_end``, ``done``, and
        ``error``.  The generator always terminates normally — exceptions are
        delivered as ``error`` events.

        Usage::

            async for event in agent.stream("Summarise these docs"):
                if event.type == "token":
                    print(event.content, end="", flush=True)
                elif event.type == "tool_start":
                    print(f"\\nCalling tool: {event.tool_name}")
                elif event.type == "done":
                    print(f"\\nCost: ${event.cost_usd:.5f}")

        Args:
            message:      The user's input message.
            thread_id:    Conversation ID for multi-turn resumption.
            attachments:  Image / document attachments for the first turn.
            tool_context: Per-request context dict injected into tool calls.

        Yields:
            :class:`~ninetrix._internals.types.StreamEvent` instances.
        """
        from ninetrix.runtime.streaming import StreamingRunner
        runner = await self._get_runner()
        streaming = StreamingRunner(
            provider=runner.provider,
            dispatcher=runner.dispatcher,
            config=runner.config,
            event_bus=runner.event_bus,
        )
        async for event in streaming.stream(
            message,
            thread_id=thread_id,
            attachments=attachments,
            tool_context=tool_context,
        ):
            yield event

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def info(self) -> AgentInfo:
        """Return a read-only summary of the agent's configuration.

        No network calls.  Safe to call before any run.

        Returns:
            :class:`~ninetrix.agent.introspection.AgentInfo`
        """
        cfg = self.config
        local_names = [
            getattr(t, "name", getattr(t, "__name__", str(t)))
            for t in cfg.local_tools
        ]
        return AgentInfo(
            name=cfg.name,
            provider=cfg.provider,
            model=cfg.model,
            tool_count=len(cfg.local_tools) + len(cfg.mcp_tools) + len(cfg.composio_tools),
            local_tools=local_names,
            mcp_tools=list(cfg.mcp_tools),
            composio_tools=list(cfg.composio_tools),
            has_persistence=bool(cfg.db_url or cfg.api_url),
            execution_mode=cfg.execution_mode,
            max_budget_usd=cfg.max_budget_usd,
            max_turns=cfg.max_turns,
            system_prompt_chars=len(cfg.system_prompt),
        )

    async def validate(self) -> list[ValidationIssue]:
        """Check configuration and credentials.  No LLM calls.

        Verifies:
        - Provider API key is present
        - Budget is non-negative

        Returns:
            List of :class:`~ninetrix.agent.introspection.ValidationIssue`.
            Empty list means everything looks good.
        """
        issues: list[ValidationIssue] = []
        cfg = self.config

        # Check API key
        from ninetrix._internals.auth import CredentialStore
        from ninetrix._internals.config import NinetrixConfig

        nxt_config = NinetrixConfig.load()
        creds = CredentialStore(nxt_config)
        key = creds.resolve(cfg.provider, explicit_key=cfg.api_key or None)
        if not key:
            env_hint = f"{cfg.provider.upper()}_API_KEY"
            issues.append(ValidationIssue(
                level="error",
                code="AUTH_MISSING",
                message=f"No API key found for provider '{cfg.provider}'.",
                fix=f"Set {env_hint} environment variable or pass api_key= to Agent().",
            ))

        # Check budget
        if cfg.max_budget_usd < 0:
            issues.append(ValidationIssue(
                level="error",
                code="INVALID_BUDGET",
                message=f"max_budget_usd must be >= 0, got {cfg.max_budget_usd}.",
                fix="Pass max_budget_usd=0.0 for unlimited, or a positive float.",
            ))

        # Warn if no tools and no instructions
        if not cfg.local_tools and not cfg.mcp_tools and not cfg.composio_tools:
            if not cfg.instructions and not cfg.role and not cfg.goal:
                issues.append(ValidationIssue(
                    level="warning",
                    code="NO_TOOLS_OR_INSTRUCTIONS",
                    message="Agent has no tools and no system prompt content.",
                    fix="Add tools=, role=, goal=, or instructions= to give the agent purpose.",
                ))

        return issues

    async def dry_run(self, message: str) -> DryRunResult:
        """Estimate cost and list available tools without making LLM calls.

        Runs :meth:`validate` and resolves tool definitions from the dispatcher.
        Useful for CI pre-flight checks and cost estimation.

        Args:
            message: The user message you plan to send (used for prompt-size estimate).

        Returns:
            :class:`~ninetrix.agent.introspection.DryRunResult`
        """
        cfg = self.config
        issues = await self.validate()

        # Resolve tool names from local tools (don't call initialize on remote sources)
        tool_names: list[str] = []
        for t in cfg.local_tools:
            tool_names.append(
                getattr(t, "name", getattr(t, "__name__", str(t)))
            )
        tool_names.extend(cfg.mcp_tools)
        tool_names.extend(cfg.composio_tools)

        system_chars = len(cfg.system_prompt)
        prompt_chars = system_chars + len(message)
        estimated_tokens = prompt_chars // 4
        price_per_1k = _estimate_price(cfg.model)
        estimated_cost = (estimated_tokens / 1000) * price_per_1k

        estimated_turns = min(cfg.max_turns, max(1, len(tool_names) + 1))

        provider_reachable = not any(
            i.level == "error" and "AUTH" in i.code for i in issues
        )
        mcp_reachable = not any(i.code == "MCP_UNREACHABLE" for i in issues)
        db_reachable = not any(i.code == "DB_UNREACHABLE" for i in issues)

        return DryRunResult(
            tools_available=tool_names,
            estimated_turns=estimated_turns,
            estimated_cost_usd=estimated_cost,
            system_prompt_chars=system_chars,
            warnings=issues,
            provider_reachable=provider_reachable,
            mcp_gateway_reachable=mcp_reachable,
            db_reachable=db_reachable,
        )

    # ------------------------------------------------------------------
    # Runner assembly
    # ------------------------------------------------------------------

    def invalidate_runner(self) -> None:
        """Discard the cached runner, forcing re-assembly on next run.

        Call this after changing config, rotating credentials, or adding tools.
        """
        self._runner = None

    async def _get_runner(self) -> Any:
        """Return the cached runner, building it on first call."""
        if self._runner is None:
            self._runner = await self._build_runner()
        return self._runner

    async def _build_runner(self) -> Any:
        """Assemble the full runtime stack.

        Single assembly point — all runtime objects (dispatcher, checkpointer,
        provider, runner) are created here.

        Layer imports are deferred to this method so that importing
        ``ninetrix.agent`` at module level does not eagerly import every
        provider SDK.
        """
        from ninetrix._internals.auth import CredentialStore
        from ninetrix._internals.config import NinetrixConfig
        from ninetrix._internals.types import CredentialError
        from ninetrix.checkpoint.memory import InMemoryCheckpointer
        from ninetrix.runtime.dispatcher import LocalToolSource, ToolDispatcher
        from ninetrix.runtime.runner import AgentRunner, RunnerConfig

        cfg = self.config
        nxt_config = NinetrixConfig.load()
        creds = CredentialStore(nxt_config)

        # 1. Resolve API key
        api_key = creds.resolve(cfg.provider, explicit_key=cfg.api_key or None)
        if not api_key:
            env_hint = f"{cfg.provider.upper()}_API_KEY"
            raise CredentialError(
                f"No API key found for provider '{cfg.provider}'.\n"
                f"  Why: {env_hint} is not set and no credentials file was found.\n"
                f"  Fix: export {env_hint}=<your-key>, or pass api_key= to Agent()."
            )

        # 2. Build provider adapter
        provider = self._build_provider(cfg.provider, api_key, cfg.model)

        # 3. Build tool sources
        sources = []
        if cfg.local_tools:
            from ninetrix.registry import _registry
            from ninetrix.runtime.dispatcher import LocalToolSource
            # Accept both ToolDef instances and raw @Tool-decorated callables
            tool_defs = []
            for t in cfg.local_tools:
                # If it's already a ToolDef, use directly
                from ninetrix.registry import ToolDef
                if isinstance(t, ToolDef):
                    tool_defs.append(t)
                else:
                    # Look up in global registry by name
                    name = getattr(t, "__name__", None) or getattr(t, "name", None)
                    if name:
                        td = _registry.get(name)
                        if td is not None:
                            tool_defs.append(td)
            sources.append(LocalToolSource(tool_defs))

        # MCP tools — wire MCPToolSource when gateway_url is configured
        if cfg.mcp_tools and cfg.mcp.gateway_url:
            from ninetrix.runtime.dispatcher import MCPToolSource
            mcp_token = cfg.mcp.token or nxt_config.mcp_gateway_token or ""
            workspace = cfg.mcp.workspace_id or nxt_config.workspace_id or "default"
            sources.append(MCPToolSource(
                gateway_url=cfg.mcp.gateway_url,
                token=mcp_token,
                workspace_id=workspace,
            ))

        # Composio tools — wire ComposioToolSource when apps are listed
        if cfg.composio_tools:
            from ninetrix.runtime.dispatcher import ComposioToolSource
            composio_key = creds.resolve("composio", explicit_key=None) or ""
            sources.append(ComposioToolSource(
                apps=list(cfg.composio_tools),
                api_key=composio_key,
            ))

        dispatcher = ToolDispatcher(sources)
        await dispatcher.initialize()

        # 4. Build checkpointer
        checkpointer = self._build_checkpointer(nxt_config, creds)

        # 5. Build runner config
        runner_config = RunnerConfig(
            name=cfg.name,
            system_prompt=cfg.system_prompt,
            model=cfg.model,
            provider_name=cfg.provider,
            max_turns=cfg.max_turns,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            tool_timeout=cfg.tool_timeout,
            max_budget_usd=cfg.max_budget_usd if cfg.max_budget_usd > 0 else None,
            output_type=cfg.output_type,
            output_retries=cfg.output_retries,
            history_max_tokens=cfg.history_max_tokens,
        )

        return AgentRunner(
            provider=provider,
            dispatcher=dispatcher,
            config=runner_config,
            checkpointer=checkpointer,
            event_bus=self._event_bus,
        )

    def _build_provider(self, provider: str, api_key: str, model: str) -> Any:
        """Instantiate the correct LLM provider adapter."""
        if provider == "anthropic":
            from ninetrix.providers.anthropic import AnthropicAdapter
            return AnthropicAdapter(api_key=api_key, model=model)
        if provider == "openai":
            from ninetrix.providers.openai import OpenAIAdapter
            return OpenAIAdapter(api_key=api_key, model=model)
        if provider == "google":
            from ninetrix.providers.google import GoogleAdapter
            return GoogleAdapter(api_key=api_key, model=model)
        if provider == "litellm":
            from ninetrix.providers.litellm import LiteLLMAdapter
            return LiteLLMAdapter(api_key=api_key, model=model)

        from ninetrix._internals.types import ConfigurationError
        raise ConfigurationError(
            f"Unknown provider '{provider}'.\n"
            f"  Fix: use one of: 'anthropic', 'openai', 'google', 'litellm'."
        )

    def _build_checkpointer(self, nxt_config: Any, creds: Any) -> Any:
        """Select a checkpointer based on available config.

        Priority: explicit db_url → api_url+token → InMemory (dev fallback).
        """
        from ninetrix.checkpoint.memory import InMemoryCheckpointer

        cfg = self.config
        db_url = cfg.db_url
        if db_url:
            try:
                from ninetrix.checkpoint.postgres import PostgresCheckpointer  # type: ignore[import]
                return PostgresCheckpointer(db_url)
            except ImportError:
                pass  # asyncpg not installed — fall through to InMemory

        api_url = cfg.api_url or getattr(nxt_config, "api_url", "")
        runner_token = cfg.runner_token or creds.resolve_workspace_token()
        if api_url and runner_token:
            try:
                from ninetrix.checkpoint.api import ApiCheckpointer  # type: ignore[import]
                return ApiCheckpointer(api_url, runner_token)
            except ImportError:
                pass  # not yet implemented — fall through

        return InMemoryCheckpointer()

    # ------------------------------------------------------------------
    # YAML round-trip
    # ------------------------------------------------------------------

    def to_yaml(self) -> str:
        """Serialise the agent configuration to an agentfile.yaml string.

        The returned string is valid agentfile.yaml that can be:
        - Round-tripped through :meth:`from_yaml`
        - Passed to ``ninetrix build`` (CLI)

        Returns:
            YAML string with a single ``agents:`` entry.

        Example::

            agent = Agent(name="analyst", provider="anthropic", role="Data analyst")
            print(agent.to_yaml())
            Path("agentfile.yaml").write_text(agent.to_yaml())
        """
        from ninetrix.export.writer import agent_to_yaml
        return agent_to_yaml(self.config)

    @classmethod
    def from_yaml(cls, path: str) -> "Agent":
        """Load an Agent from an agentfile.yaml file.

        Parses the first (or only) agent entry in the ``agents:`` block.
        ``${ENV_VAR}`` placeholders in string values are expanded.

        Args:
            path: Filesystem path to an agentfile.yaml file.

        Returns:
            A configured :class:`Agent` instance.

        Example::

            agent = Agent.from_yaml("agentfile.yaml")
            result = agent.run("Summarise Q1 results")
        """
        from ninetrix.export.loader import load_agent_from_yaml
        return load_agent_from_yaml(path)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"Agent(name={cfg.name!r}, provider={cfg.provider!r}, "
            f"model={cfg.model!r}, tools={len(cfg.local_tools)})"
        )
