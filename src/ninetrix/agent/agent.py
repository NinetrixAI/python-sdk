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
from typing import Any, Generic, Optional, TypeVar

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

        # Auto-wire reporter when no parent scope (workflow/team) set one.
        from ninetrix._internals.trace import get_reporter, reporter_scope
        if get_reporter() is None:
            try:
                from ninetrix.observability.reporter import RunnerReporter
                _reporter = RunnerReporter.resolve()
                if _reporter is not None:
                    async with reporter_scope(_reporter):
                        return await runner.run(message, thread_id=thread_id)
            except Exception:
                pass  # reporter setup must never block the agent

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
        from ninetrix.tools.toolkit import Toolkit
        local_names: list[str] = []
        for t in cfg.local_tools:
            if isinstance(t, Toolkit):
                local_names.extend(td.name for td in t.tools())
            else:
                local_names.append(
                    getattr(t, "name", getattr(t, "__name__", str(t)))
                )
        return AgentInfo(
            name=cfg.name,
            provider=cfg.provider,
            model=cfg.model,
            tool_count=len(local_names) + len(cfg.mcp_tools) + len(cfg.composio_tools),
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
        from ninetrix._internals.http import get_http_client
        from ninetrix._internals.types import CredentialError
        from ninetrix.checkpoint.memory import InMemoryCheckpointer
        from ninetrix.runtime.dispatcher import LocalToolSource, ToolDispatcher
        from ninetrix.runtime.runner import AgentRunner, RunnerConfig
        from ninetrix.tools.agent_context import AgentContext
        from ninetrix.tools.auth_resolver import AuthResolver

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

        # 2b. Build shared AgentContext for all tool sources
        agent_ctx = AgentContext(
            http=get_http_client(),
            auth=AuthResolver(),
            agent_name=cfg.name,
            org_id=nxt_config.org_id or "default",
        )

        # 3. Build tool sources
        sources = []
        if cfg.local_tools:
            from ninetrix.registry import _registry, ToolDef
            from ninetrix.runtime.dispatcher import LocalToolSource
            # Accept ToolDef, @Tool-decorated callables, and Toolkit objects
            tool_defs = []
            for t in cfg.local_tools:
                # Toolkit — unwrap all its ToolDefs
                from ninetrix.tools.toolkit import Toolkit
                if isinstance(t, Toolkit):
                    tool_defs.extend(t.tools())
                    continue
                # Already a ToolDef — use directly
                if isinstance(t, ToolDef):
                    tool_defs.append(t)
                    continue
                # @Tool-decorated callable — look up in global registry by name
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
            org_id = cfg.mcp.org_id or nxt_config.org_id or "default"
            sources.append(MCPToolSource(
                gateway_url=cfg.mcp.gateway_url,
                token=mcp_token,
                org_id=org_id,
                ctx=agent_ctx,
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
            execution_mode=cfg.execution_mode,
        )

        runner = AgentRunner(
            provider=provider,
            dispatcher=dispatcher,
            config=runner_config,
            checkpointer=checkpointer,
            event_bus=self._event_bus,
        )

        # Auto-attach OTEL if configure_otel() was called anywhere in the process
        from ninetrix.observability.otel import _otel_configured, attach_otel_to_bus
        if _otel_configured:
            attach_otel_to_bus(self._event_bus)

        return runner

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
        runner_token = cfg.runner_token or creds.resolve_org_token()
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
    # Lifecycle: serve / build / deploy
    # ------------------------------------------------------------------

    def serve(
        self,
        host: str = "0.0.0.0",
        port: int = 9000,
        reload: bool = False,
        *,
        app_only: bool = False,
    ) -> Any:
        """Serve the agent as an HTTP API using FastAPI + uvicorn.

        Starts a blocking web server that exposes the agent on:

        - ``POST /invoke``  — run the agent, return :class:`~ninetrix.AgentResult` JSON
        - ``GET  /info``    — return :class:`~ninetrix.agent.introspection.AgentInfo` JSON
        - ``GET  /health``  — return ``{"status": "ok", "agent": name}``
        - ``POST /stream``  — SSE endpoint streaming :class:`~ninetrix.StreamEvent` objects

        Args:
            host:      Bind address (default: ``"0.0.0.0"``).
            port:      TCP port (default: ``9000``).
            reload:    Enable hot-reload (for development).
            app_only:  If ``True``, return the FastAPI app without starting
                       the server.  Useful for ASGI embedding or testing.

        Returns:
            When ``app_only=True``: the FastAPI application instance.
            When ``app_only=False``: does not return (blocking call).

        Raises:
            ConfigurationError: If FastAPI or uvicorn is not installed.

        Example::

            agent.serve(host="0.0.0.0", port=9000)
            # Or for testing:
            app = agent.serve(app_only=True)
        """
        from ninetrix.agent.server import create_agent_app, serve_agent

        if app_only:
            return create_agent_app(self)

        serve_agent(self, host=host, port=port, reload=reload)
        return None

    def build(
        self,
        tag: Optional[str] = None,
        push: bool = False,
    ) -> dict[str, Any]:
        """Build a Docker image for this agent using the Ninetrix CLI.

        Serialises the agent to a temporary ``agentfile.yaml``, then invokes
        ``ninetrix build`` as a subprocess.

        Args:
            tag:  Docker image tag (default: ``"agentfile/{name}:latest"``).
            push: If ``True``, also run ``docker push {tag}`` after a successful build.

        Returns:
            ``{"image": tag, "yaml_path": str, "success": bool}``

        Raises:
            ConfigurationError: If the ``ninetrix`` CLI is not found in PATH.

        Example::

            result = agent.build(tag="myorg/my-agent:v1", push=True)
            print(result["image"])   # "myorg/my-agent:v1"
        """
        import shutil
        import subprocess
        import tempfile
        import os
        import uuid

        from ninetrix._internals.types import ConfigurationError

        tag = tag or f"agentfile/{self.config.name}:latest"

        # Check that ninetrix CLI is available
        if not shutil.which("ninetrix") and not shutil.which("agentfile"):
            raise ConfigurationError(
                "The 'ninetrix' CLI was not found in PATH.\n"
                "  Why: agent.build() requires the CLI to call 'ninetrix build'.\n"
                "  Fix: pip install ninetrix  or ensure the CLI is on your PATH."
            )

        cli = shutil.which("ninetrix") or shutil.which("agentfile")

        # Write agentfile.yaml to a temp file
        yaml_str = self.to_yaml()
        tmp_dir = tempfile.gettempdir()
        yaml_path = os.path.join(tmp_dir, f"agentfile_{uuid.uuid4().hex[:8]}.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_str)

        try:
            result = subprocess.run(
                [cli, "build", "--file", yaml_path, "--tag", tag],
                capture_output=True,
                text=True,
            )
            success = result.returncode == 0

            if success and push:
                docker = shutil.which("docker")
                if docker:
                    subprocess.run(
                        [docker, "push", tag],
                        capture_output=True,
                        text=True,
                    )
        except Exception as exc:
            from ninetrix._internals.types import ConfigurationError as CE
            raise CE(
                f"agent.build() failed while running the CLI.\n"
                f"  Why: {type(exc).__name__}: {exc}\n"
                "  Fix: ensure 'ninetrix' CLI and 'docker' are installed and on PATH."
            ) from exc

        return {
            "image": tag,
            "yaml_path": yaml_path,
            "success": success,
        }

    async def deploy(
        self,
        org_id: Optional[str] = None,
        api_key: Optional[str] = None,
        region: str = "iad",
    ) -> dict[str, Any]:
        """Deploy this agent to Ninetrix Cloud.

        Serialises the agent to YAML and calls
        ``POST /v1/deployments`` on the Ninetrix Cloud API.

        Args:
            org_id:       Cloud organization ID.  If omitted, resolved from
                          :class:`~ninetrix._internals.tenant.TenantContext` or
                          ``NINETRIX_ORG_ID`` env var.
            api_key:      Ninetrix API key.  If omitted, resolved from
                          :class:`~ninetrix._internals.tenant.TenantContext` or
                          ``NINETRIX_API_KEY`` env var.
            region:       Fly.io region to deploy to (default: ``"iad"``).

        Returns:
            ``{"deployment_id": str, "url": str, "status": "deploying"}``

        Raises:
            CredentialError: If org_id or api_key cannot be resolved.

        Example::

            result = await agent.deploy(org_id="org_abc123", api_key="nxt_...")
            print(result["url"])   # "https://my-agent.ninetrix.app"
        """
        import os
        from ninetrix._internals.types import CredentialError
        from ninetrix._internals.http import get_http_client

        # Resolve org_id
        if not org_id:
            from ninetrix._internals.tenant import get_tenant
            tenant = get_tenant()
            if tenant:
                org_id = tenant.org_id
            org_id = org_id or os.environ.get("NINETRIX_ORG_ID", "")

        # Resolve api_key
        if not api_key:
            from ninetrix._internals.tenant import get_tenant
            tenant = get_tenant()
            if tenant:
                api_key = tenant.api_key
            api_key = api_key or os.environ.get("NINETRIX_API_KEY", "")

        if not org_id or not api_key:
            raise CredentialError(
                "agent.deploy() requires org_id and api_key.\n"
                "  Why: could not find org_id or api_key in arguments, "
                "TenantContext, or environment variables.\n"
                "  Fix: pass org_id= and api_key= explicitly, or set "
                "NINETRIX_ORG_ID and NINETRIX_API_KEY environment variables."
            )

        yaml_str = self.to_yaml()
        api_base = os.environ.get("NINETRIX_API_URL", "https://api.ninetrix.io")

        client = get_http_client()
        resp = await client.post(
            f"{api_base}/v1/deployments",
            json={
                "yaml": yaml_str,
                "agent_name": self.config.name,
                "region": region,
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-Org-ID": org_id,
            },
            timeout=60.0,
        )

        if resp.status_code >= 400:
            raise CredentialError(
                f"agent.deploy() failed with HTTP {resp.status_code}.\n"
                f"  Response: {resp.text[:200]}\n"
                "  Fix: check your org_id and api_key."
            )

        data = resp.json()
        return {
            "deployment_id": data.get("deployment_id", ""),
            "url": data.get("url", ""),
            "status": data.get("status", "deploying"),
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"Agent(name={cfg.name!r}, provider={cfg.provider!r}, "
            f"model={cfg.model!r}, tools={len(cfg.local_tools)})"
        )
