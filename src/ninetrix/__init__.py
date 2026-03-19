"""
Ninetrix SDK — developer API for building production AI agents.

Version: 0.1.0
Install:  pip install ninetrix-sdk
Extras:   pip install 'ninetrix-sdk[serve]'     # Agent.serve() — FastAPI HTTP server
          pip install 'ninetrix-sdk[otel]'      # OpenTelemetry tracing
          pip install 'ninetrix-sdk[providers]' # anthropic + openai + google + litellm

──────────────────────────────────────────────────────────────────
1. DEFINE TOOLS  (@Tool)
──────────────────────────────────────────────────────────────────
Decorate any Python function. The decorator registers it without
wrapping — full type inference and IDE completion are preserved::

    from ninetrix import Tool

    @Tool
    def get_price(ticker: str) -> dict:
        \"\"\"Fetch the current stock price.

        Args:
            ticker: Stock ticker symbol, e.g. \"AAPL\".
        \"\"\"
        return {"ticker": ticker, "price": 189.50}

    # Group related tools into a Toolkit
    from ninetrix import Toolkit
    market_tools = Toolkit("market", tools=[get_price, get_volume])


──────────────────────────────────────────────────────────────────
2. BUILD AGENTS  (Agent)
──────────────────────────────────────────────────────────────────
::

    from ninetrix import Agent

    analyst = Agent(
        name="equity-analyst",
        provider="anthropic",          # "openai" | "google" | "litellm"
        model="claude-haiku-4-5-20251001",
        role="You are a concise equity research analyst.",
        tools=[get_price, market_tools],
        # optional budgets:
        max_turns=10,
        budget_tokens=50_000,
        budget_usd=0.05,
    )

    # Async run (preferred)
    result = await analyst.arun("What is Apple's current price?")
    print(result.output)          # str (or Pydantic model if output_type= set)
    print(result.tokens_used)     # int
    print(result.cost_usd)        # float
    print(result.thread_id)       # str — use to resume later

    # Sync run (safe inside any event loop)
    result = analyst.run("What is Apple's current price?")

    # Streaming
    async for event in analyst.stream("Summarise Apple"):
        if event.type == "text_delta":
            print(event.text, end="", flush=True)

    # Introspection (no API call)
    info   = analyst.info()        # AgentInfo
    issues = await analyst.validate()   # list[ValidationIssue]
    dry    = await analyst.dry_run("test prompt")   # DryRunResult

    # Structured output — output becomes a Pydantic model instance
    from pydantic import BaseModel
    class Report(BaseModel):
        summary: str
        risk: str
    typed_agent = Agent(..., output_type=Report)
    result = await typed_agent.arun("Analyse AAPL")
    print(result.output.risk)


──────────────────────────────────────────────────────────────────
3. COMPOSE WORKFLOWS  (@Workflow / Team)
──────────────────────────────────────────────────────────────────
Sequential pipeline with crash-resume checkpointing::

    from ninetrix import Workflow, InMemoryCheckpointer

    @Workflow(durable=True)
    async def report_pipeline(query: str) -> str:
        async with Workflow.step("research") as step:
            if not step.is_cached:
                result = await analyst.arun(query)
                step.set(result.output)
            research = step.value

        async with Workflow.step("write") as step:
            if not step.is_cached:
                result = await writer.arun(research)
                step.set(result.output)
            return step.value

    checkpointer = InMemoryCheckpointer()    # or PostgresCheckpointer
    report_pipeline.inject_checkpointer(checkpointer)
    result = await report_pipeline.arun(query, thread_id="run-001")

Human-in-the-loop gate — halt and wait for approval::

    async with Workflow.step("review", requires_approval=True) as step:
        if not step.is_cached:
            step.set({"draft": draft})
            return "PENDING_APPROVAL"   # caller polls; re-run same thread_id
        approved = step.value           # execution resumes here after approval

LLM-based dynamic routing (Team)::

    from ninetrix import Team
    support = Team(
        agents=[billing_agent, tech_agent, general_agent],
        router_provider="anthropic",
        router_model="claude-haiku-4-5-20251001",
    )
    result = await support.arun("I was charged twice this month.")
    print(result.routed_to, result.output)


──────────────────────────────────────────────────────────────────
4. OBSERVE  (debug / OpenTelemetry)
──────────────────────────────────────────────────────────────────
Local development — pretty-printer to stderr::

    from ninetrix import enable_debug
    enable_debug(agent=analyst)    # pretty-prints every event for analyst
    # or set NINETRIX_DEBUG=1 in the environment for all agents

Production — OpenTelemetry to any OTLP backend::

    from ninetrix import configure_otel
    configure_otel(
        endpoint="http://localhost:4317",   # Jaeger / Grafana Tempo / Datadog
        service_name="my-agent-service",
    )
    # Every Agent in the process now emits spans automatically.
    # Span tree per run:
    #   ninetrix.agent.run
    #   ├─ ninetrix.agent.turn   [turn=0, tokens=…]
    #   │  └─ ninetrix.tool.call [tool=get_price]
    #   └─ ninetrix.agent.turn   [turn=1, …]

Subscribe to raw events::

    from ninetrix import AgentEvent
    analyst._event_bus.subscribe("tool.*", lambda e: print(e))
    analyst._event_bus.subscribe("*",      lambda e: ...)   # every event


──────────────────────────────────────────────────────────────────
5. TEST  (MockTool / AgentSandbox)
──────────────────────────────────────────────────────────────────
::

    from ninetrix import AgentSandbox, MockTool
    import pytest

    @pytest.mark.asyncio
    async def test_analyst():
        mock_price = MockTool("get_price", return_value={"price": 100.0})
        async with AgentSandbox(analyst, tools=[mock_price]) as sb:
            result = await sb.run("What is the price of AAPL?")
        assert "100" in result.output
        mock_price.assert_called_once_with(ticker="AAPL")


──────────────────────────────────────────────────────────────────
6. SERVE / DEPLOY  (Agent.serve / Agent.deploy)
──────────────────────────────────────────────────────────────────
::

    # Expose agent as FastAPI HTTP server  (requires: pip install 'ninetrix-sdk[serve]')
    from ninetrix import serve_agent
    serve_agent(analyst, host="0.0.0.0", port=9000)
    # POST /invoke   {"prompt": "…", "thread_id": "…"}
    # GET  /info
    # GET  /health

    # Deploy to Ninetrix Cloud
    await analyst.deploy(api_key="nxt-…")


──────────────────────────────────────────────────────────────────
Full examples: sdk/examples/
  01_simple_agent.py          — @Tool + Agent.arun + Agent.run
  02_parallel_research.py     — fan-out parallel agents
  03_durable_pipeline.py      — durable @Workflow + PostgresCheckpointer
  04_multi_agent_team.py      — Team dynamic routing
  05_toolkit_streaming.py     — Toolkit + streaming
  06_streaming.py             — event-by-event streaming
  07_otel_tracing.py          — configure_otel + Jaeger + enable_debug
  08_hitl_approval.py         — HITL approval gate + approve/reject helpers
  09_multi_agent_handoff.py   — coordinator → specialist → governance pipeline
──────────────────────────────────────────────────────────────────
"""

from ninetrix.context import Ninetrix as Ninetrix
from ninetrix.tool import Tool as Tool
from ninetrix.registry import ToolDef as ToolDef, ToolRegistry as ToolRegistry, _registry
from ninetrix.discover import (
    discover_tools_in_file as discover_tools_in_file,
    discover_tools_in_files as discover_tools_in_files,
    load_local_tools as load_local_tools,
)
from ninetrix._internals.lifespan import (
    startup as startup,
    shutdown as shutdown,
    lifespan as lifespan,
)
from ninetrix.providers import (
    FallbackConfig as FallbackConfig,
    FallbackProviderAdapter as FallbackProviderAdapter,
)
from ninetrix.observability.logger import (
    NinetrixLogger as NinetrixLogger,
    enable_debug as enable_debug,
    get_logger as get_logger,
)
from ninetrix.observability.telemetry import (
    TelemetryEvent as TelemetryEvent,
    TelemetryCollector as TelemetryCollector,
    record_event as record_event,
)
from ninetrix.observability.errors import (
    ErrorContext as ErrorContext,
    error_context as error_context,
)
from ninetrix.observability.events import (
    AgentEvent as AgentEvent,
    EventBus as EventBus,
)
from ninetrix.observability.hooks import HooksMixin as HooksMixin
from ninetrix.runtime.streaming import StreamingRunner as StreamingRunner
from ninetrix.client.local import AgentClient as AgentClient
from ninetrix.client.remote import RemoteAgent as RemoteAgent
from ninetrix._internals.tenant import (
    TenantContext as TenantContext,
    set_tenant as set_tenant,
    get_tenant as get_tenant,
    require_tenant as require_tenant,
    tenant_scope as tenant_scope,
)
from ninetrix.runtime.history import MessageHistory as MessageHistory
from ninetrix.runtime.budget import BudgetTracker as BudgetTracker, BudgetUsage as BudgetUsage
from ninetrix.runtime.runner import AgentRunner as AgentRunner, RunnerConfig as RunnerConfig
from ninetrix.runtime.planner import Planner as Planner
from ninetrix.workflow.context import WorkflowContext as WorkflowContext, WorkflowBudgetTracker as WorkflowBudgetTracker
from ninetrix.workflow.workflow import (
    Workflow as Workflow,
    WorkflowRunner as WorkflowRunner,
    WorkflowTerminated as WorkflowTerminated,
)
from ninetrix.workflow.team import Team as Team, TeamResult as TeamResult
from ninetrix.tools.toolkit import Toolkit as Toolkit
from ninetrix.checkpoint.base import Checkpointer as Checkpointer
from ninetrix.checkpoint.memory import InMemoryCheckpointer as InMemoryCheckpointer
from ninetrix.checkpoint.postgres import PostgresCheckpointer as PostgresCheckpointer
from ninetrix.agent.config import AgentConfig as AgentConfig
from ninetrix.agent.introspection import (
    AgentInfo as AgentInfo,
    ValidationIssue as ValidationIssue,
    DryRunResult as DryRunResult,
)
from ninetrix.agent.agent import Agent as Agent
from ninetrix.export.writer import agent_to_yaml as agent_to_yaml
from ninetrix.export.loader import (
    load_agent_from_yaml as load_agent_from_yaml,
    load_all_agents_from_yaml as load_all_agents_from_yaml,
)
from ninetrix.testing import MockTool as MockTool, AgentSandbox as AgentSandbox
from ninetrix.observability.otel import (
    configure_otel as configure_otel,
    get_tracer as get_tracer,
    attach_otel_to_bus as attach_otel_to_bus,
)
from ninetrix.observability.debug import attach_debug_listener as attach_debug_listener
from ninetrix.agent.server import serve_agent as serve_agent
from ninetrix.runtime.dispatcher import (
    ToolSource as ToolSource,
    ToolDispatcher as ToolDispatcher,
    LocalToolSource as LocalToolSource,
    RegistryToolSource as RegistryToolSource,
    MCPToolSource as MCPToolSource,
    ComposioToolSource as ComposioToolSource,
)
from ninetrix.tools.context import ToolContext as ToolContext
from ninetrix._internals.types import (
    # Result / event types
    AgentResult as AgentResult,
    StreamEvent as StreamEvent,
    WorkflowResult as WorkflowResult,
    StepResult as StepResult,
    # Attachment helpers
    ImageAttachment as ImageAttachment,
    DocumentAttachment as DocumentAttachment,
    Attachment as Attachment,
    image as image,
    document as document,
    # Provider config
    ProviderConfig as ProviderConfig,
    # Protocols
    AgentProtocol as AgentProtocol,
    # Errors
    NinetrixError as NinetrixError,
    CredentialError as CredentialError,
    ProviderError as ProviderError,
    ToolError as ToolError,
    BudgetExceededError as BudgetExceededError,
    OutputParseError as OutputParseError,
    CheckpointError as CheckpointError,
    ApprovalTimeoutError as ApprovalTimeoutError,
    ConfigurationError as ConfigurationError,
    NetworkError as NetworkError,
)

__version__ = "0.1.0"
__all__ = [
    # v2 API — Ninetrix context + workflow improvements
    "Ninetrix",
    "WorkflowTerminated",
    # PR 32 — serve / build / deploy lifecycle methods
    "serve_agent",
    # PR 33 — debug pretty-printer
    "attach_debug_listener",
    # PR 31 — OpenTelemetry integration
    "configure_otel",
    "get_tracer",
    "attach_otel_to_bus",
    # PR 30 — MockTool + AgentSandbox
    "MockTool",
    "AgentSandbox",
    # PR 19 — YAML round-trip
    "agent_to_yaml",
    "load_agent_from_yaml",
    "load_all_agents_from_yaml",
    # PR 18 — Agent + AgentConfig + introspection
    "Agent",
    "AgentConfig",
    "AgentInfo",
    "ValidationIssue",
    "DryRunResult",
    # PR 28 — PostgresCheckpointer
    "PostgresCheckpointer",
    # PR 17 — Checkpointer + InMemoryCheckpointer
    "Checkpointer",
    "InMemoryCheckpointer",
    # PR 27 — Toolkit
    "Toolkit",
    # PR 26 — Team + TeamResult
    "Team",
    "TeamResult",
    # PR 25 — Workflow + WorkflowContext + WorkflowBudgetTracker
    "Workflow",
    "WorkflowRunner",
    "WorkflowContext",
    "WorkflowBudgetTracker",
    # PR 24 — Planner
    "Planner",
    # PR 16 — AgentRunner + RunnerConfig
    "AgentRunner",
    "RunnerConfig",
    # PR 23 — MCPToolSource + ComposioToolSource
    "MCPToolSource",
    "ComposioToolSource",
    # PR 15 — ToolDispatcher + ToolContext
    "ToolSource",
    "ToolDispatcher",
    "LocalToolSource",
    "RegistryToolSource",
    "ToolContext",
    # PR 14 — MessageHistory + BudgetTracker
    "MessageHistory",
    "BudgetTracker",
    "BudgetUsage",
    # PR 13 — TenantContext
    "TenantContext",
    "set_tenant",
    "get_tenant",
    "require_tenant",
    "tenant_scope",
    # PR 22 — AgentClient + RemoteAgent
    "AgentClient",
    "RemoteAgent",
    # PR 21 — StreamingRunner
    "StreamingRunner",
    # PR 20 — EventBus + HooksMixin
    "AgentEvent",
    "EventBus",
    "HooksMixin",
    # PR 9 — ErrorContext
    "ErrorContext",
    "error_context",
    # PR 8 — Telemetry
    "TelemetryEvent",
    "TelemetryCollector",
    "record_event",
    # PR 7 — Logger
    "NinetrixLogger",
    "enable_debug",
    "get_logger",
    # PR 6 — Providers
    "FallbackConfig",
    "FallbackProviderAdapter",
    # Phase 1 — @Tool decorator
    "Tool",
    "ToolDef",
    "ToolRegistry",
    "discover_tools_in_file",
    "discover_tools_in_files",
    "load_local_tools",
    "_registry",
    # PR 5 — lifespan
    "startup",
    "shutdown",
    "lifespan",
    # PR 1 — types
    "AgentResult",
    "StreamEvent",
    "WorkflowResult",
    "StepResult",
    "ImageAttachment",
    "DocumentAttachment",
    "Attachment",
    "image",
    "document",
    "ProviderConfig",
    "AgentProtocol",
    "NinetrixError",
    "CredentialError",
    "ProviderError",
    "ToolError",
    "BudgetExceededError",
    "OutputParseError",
    "CheckpointError",
    "ApprovalTimeoutError",
    "ConfigurationError",
    "NetworkError",
]
