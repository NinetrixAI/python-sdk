"""
North-star usage examples file.

This file defines what "done" looks like for the full SDK.
It starts mostly commented-out and gets uncommented as PRs land.
When the entire file runs without error, the SDK is complete.

Run with:   python tests/usage_examples.py
or:         pytest tests/usage_examples.py -v

Each block is gated on the PR that makes it work.
"""

# =============================================================================
# Phase 1 — @Tool decorator (DONE — PR 0 / existing)
# =============================================================================

from ninetrix import Tool, ToolDef, ToolRegistry, _registry

@Tool
def get_price(ticker: str) -> float:
    """Get the current stock price for a ticker symbol."""
    return 150.0

@Tool
def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web and return results."""
    return [{"title": "result", "url": "https://example.com"}]

# @Tool preserves full type inference — IDE sees get_price(ticker: str) -> float
price: float = get_price("AAPL")
assert price == 150.0, "Tool callable as-is"

tools = _registry.all()
assert any(t.name == "get_price" for t in tools)

print("✓ Phase 1: @Tool decorator")


# =============================================================================
# PR 1 — Types + py.typed
# =============================================================================

from ninetrix._internals.types import (
    AgentResult,
    WorkflowResult,
    StreamEvent,
    NinetrixError,
    CredentialError,
    ProviderError,
    ToolError,
    BudgetExceededError,
    OutputParseError,
    CheckpointError,
    ApprovalTimeoutError,
    ConfigurationError,
    NetworkError,
    AgentProtocol,
    LLMResponse,
    ToolCall,
    ImageAttachment,
    DocumentAttachment,
)
import ninetrix
import os

assert os.path.exists(
    os.path.join(os.path.dirname(ninetrix.__file__), "py.typed")
), "py.typed marker must exist"

# AgentResult is Generic
result: AgentResult[str] = AgentResult(
    output="hello",
    thread_id="t1",
    tokens_used=10,
    input_tokens=7,
    output_tokens=3,
    cost_usd=0.0001,
    steps=1,
)
assert result.output == "hello"
d = result.to_dict()
assert d["output"] == "hello"
assert d["error"] is None

# Error hierarchy — all inherit from NinetrixError
assert issubclass(CredentialError, NinetrixError)
assert issubclass(ProviderError, NinetrixError)
assert issubclass(ToolError, NinetrixError)
assert issubclass(BudgetExceededError, NinetrixError)
assert issubclass(OutputParseError, NinetrixError)
assert issubclass(CheckpointError, NinetrixError)
assert issubclass(ApprovalTimeoutError, NinetrixError)
assert issubclass(ConfigurationError, NinetrixError)
assert issubclass(NetworkError, NinetrixError)

# Error attributes
pe = ProviderError("test", status_code=429, provider="anthropic", retryable=True)
assert pe.retryable is True
ope = OutputParseError("bad json", raw_output="{bad}", attempts=2)
assert ope.attempts == 2

# LLMResponse + ToolCall
resp = LLMResponse(content="hi", tool_calls=[], input_tokens=5, output_tokens=3)
tc = ToolCall(id="c1", name="get_price", arguments={"ticker": "AAPL"})
assert tc.name == "get_price"

# ImageAttachment
img = ImageAttachment(url="https://example.com/img.png")
assert img.url is not None

print("✓ PR 1: Types + py.typed")


# =============================================================================
# PR 3 — NinetrixConfig
# =============================================================================

from ninetrix._internals.config import NinetrixConfig
cfg = NinetrixConfig.load()   # reads env → ~/.ninetrix/config.toml → defaults
assert cfg.default_provider in ("anthropic", "openai", "google", "litellm", "groq", "mistral")
print("✓ PR 3: NinetrixConfig")


# =============================================================================
# PR 4 — CredentialStore
# =============================================================================

# Uncomment when PR 4 lands:
#
# from ninetrix._internals.auth import CredentialStore
# from ninetrix._internals.config import NinetrixConfig
# creds = CredentialStore(NinetrixConfig.load())
# # key = creds.resolve("anthropic")  # reads ANTHROPIC_API_KEY or config
# print("✓ PR 4: CredentialStore")


# =============================================================================
# PR 6 — Providers
# =============================================================================

import sys, types as _types, unittest.mock as _mock
_fake_anthropic = _types.ModuleType("anthropic")
_fake_anthropic.AsyncAnthropic = _mock.MagicMock()  # type: ignore[attr-defined]
_fake_anthropic.AuthenticationError = Exception  # type: ignore[attr-defined]
_fake_anthropic.RateLimitError = Exception  # type: ignore[attr-defined]
_fake_anthropic.APIStatusError = Exception  # type: ignore[attr-defined]
if "anthropic" not in sys.modules:
    sys.modules["anthropic"] = _fake_anthropic
from ninetrix.providers.anthropic import AnthropicAdapter
from ninetrix.providers.base import LLMProviderAdapter
assert issubclass(AnthropicAdapter, LLMProviderAdapter)
from ninetrix.providers.fallback import FallbackConfig, FallbackProviderAdapter
from ninetrix import FallbackConfig as FC, FallbackProviderAdapter as FPA
assert FC is FallbackConfig
assert FPA is FallbackProviderAdapter
print("✓ PR 6: Providers")


# =============================================================================
# PR 7 — NinetrixLogger
# =============================================================================

from ninetrix.observability.logger import NinetrixLogger, enable_debug, get_logger
from ninetrix import NinetrixLogger as NL, enable_debug as ed, get_logger as gl
assert NL is NinetrixLogger
assert ed is enable_debug
assert gl is get_logger

log = NinetrixLogger("usage_examples")
log.info("logger works", key="value")
log.debug("debug suppressed at default WARNING level")
log.warning("warn visible")

log2 = get_logger("usage_examples.sub")
assert "usage_examples.sub" in log2._logger.name

print("✓ PR 7: NinetrixLogger")


# =============================================================================
# PR 8 — Telemetry
# =============================================================================

from ninetrix.observability.telemetry import TelemetryEvent, TelemetryCollector, record_event
from ninetrix import TelemetryEvent as TE, record_event as re_fn
assert TE is TelemetryEvent
assert re_fn is record_event

evt = TelemetryEvent(event="agent_run", provider="anthropic", success=True)
assert evt.event == "agent_run"
assert evt.provider == "anthropic"
d = evt.to_dict()
assert "machine_id" in d and len(d["machine_id"]) == 16

# record_event convenience wrapper
record_event("usage_test", provider="openai", tokens_used=100, success=True)

print("✓ PR 8: Telemetry")


# =============================================================================
# PR 9 — ErrorContext
# =============================================================================

from ninetrix.observability.errors import ErrorContext, error_context
from ninetrix import ErrorContext as EC, error_context as ec
assert EC is ErrorContext
assert ec is error_context

from ninetrix._internals.types import ProviderError
try:
    with ErrorContext(agent_name="analyst", thread_id="t-1", step=3):
        raise ProviderError("timeout", provider="anthropic")
except ProviderError as e:
    ctx = ErrorContext.get(e)
    assert ctx["agent_name"] == "analyst"
    assert ctx["thread_id"] == "t-1"
    assert ctx["step"] == 3

print("✓ PR 9: ErrorContext")


# =============================================================================
# PR 13 — TenantContext
# =============================================================================

from ninetrix._internals.tenant import TenantContext, set_tenant, get_tenant, require_tenant, tenant_scope
from ninetrix import TenantContext as TC, tenant_scope as ts
assert TC is TenantContext
assert ts is tenant_scope

import asyncio as _asyncio

async def _test_tenant():
    tc = TenantContext(workspace_id="ws-example", api_key="nxt_test")
    async with tenant_scope(tc) as active:
        assert active is tc
        assert get_tenant() is tc
        assert require_tenant().workspace_id == "ws-example"
    assert get_tenant() is None

_asyncio.run(_test_tenant())
print("✓ PR 13: TenantContext")


# =============================================================================
# PR 14 — MessageHistory + BudgetTracker
# =============================================================================

from ninetrix.runtime.history import MessageHistory
from ninetrix.runtime.budget import BudgetTracker

history = MessageHistory(max_tokens=8000)
history.append({"role": "user", "content": "hello"})
assert len(history.messages()) == 1

budget = BudgetTracker(max_usd=1.0)
budget.charge(input_tokens=100, output_tokens=50, model="claude-sonnet-4-6", provider="anthropic")
assert budget.usage().cost_usd > 0

print("✓ PR 14: MessageHistory + BudgetTracker")


# =============================================================================
# PR 14 — ToolDispatcher (local)
# =============================================================================

import asyncio as _asyncio
from ninetrix.runtime.dispatcher import ToolDispatcher, LocalToolSource
from ninetrix.registry import _registry as _reg

_defs = _reg.all()
_source = LocalToolSource(_defs)
_dispatcher = ToolDispatcher([_source])
_result = _asyncio.run(_dispatcher.call("get_price", {"ticker": "AAPL"}))
assert "150" in _result
print("✓ PR 15: ToolDispatcher (local)")


# =============================================================================
# PR 17 — InMemoryCheckpointer
# =============================================================================

import asyncio
from ninetrix.checkpoint.memory import InMemoryCheckpointer
cp = InMemoryCheckpointer()
asyncio.run(cp.save(thread_id="t1", agent_id="a1", history=[{"role": "user", "content": "hi"}], tokens_used=10))
saved = asyncio.run(cp.load("t1"))
assert saved["history"][0]["content"] == "hi"
print("✓ PR 17: InMemoryCheckpointer")


# =============================================================================
# PR 18 — Agent (core)
# =============================================================================

from ninetrix import Agent

# Basic agent construction — no API call yet
agent = Agent(
    name="analyst",
    provider="anthropic",
    model="claude-sonnet-4-6",
    role="Financial analyst",
    goal="Answer questions about company metrics.",
)

# info() — static summary, no network calls
summary = agent.info()
assert summary.name == "analyst"
assert summary.provider == "anthropic"
assert summary.tool_count == 0
assert summary.system_prompt_chars > 0

# run() and arun() require a real API key — skip live call in usage_examples
# result = agent.run("What is the price of AAPL?")
# assert isinstance(result.output, str)

print("✓ PR 18: Agent (info / validate / dry_run)")


# =============================================================================
# PR 17 — Agent with output_type (structured output)
# =============================================================================

# Uncomment when PR 17 + PR 6 land:
#
# from pydantic import BaseModel
#
# class StockReport(BaseModel):
#     ticker: str
#     recommendation: str
#     target_price: float
#     reasons: list[str]
#
# typed_agent = Agent(
#     provider="anthropic",
#     model="claude-sonnet-4-6",
#     output_type=StockReport,
# )
# # result = await typed_agent.arun("Analyze AAPL")
# # assert isinstance(result.output, StockReport)  # IDE: StockReport ✓
# # report: StockReport = result.output
# # assert report.ticker == "AAPL"
#
# print("✓ PR 17: output_type generic")


# =============================================================================
# PR 19 — YAML round-trip
# =============================================================================

import tempfile, os
from ninetrix import agent_to_yaml, load_agent_from_yaml, Agent as _Agent19

_a19 = _Agent19(
    name="roundtrip-bot",
    provider="anthropic",
    model="claude-sonnet-4-6",
    role="Data analyst",
    goal="Answer questions.",
    mcp_tools=["tavily"],
    max_budget_usd=1.0,
)
_yaml_text = _a19.to_yaml()
assert "roundtrip-bot" in _yaml_text
assert "mcp://tavily" in _yaml_text

with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as _f:
    _f.write(_yaml_text)
    _tmp = _f.name

_a19b = load_agent_from_yaml(_tmp)
os.unlink(_tmp)
assert _a19b.name == "roundtrip-bot"
assert _a19b.config.role == "Data analyst"
assert "tavily" in _a19b.config.mcp_tools
print("✓ PR 19: YAML round-trip (to_yaml / from_yaml)")


# =============================================================================
# PR 20 — Events + Hooks
# =============================================================================

from ninetrix import Agent, AgentEvent, EventBus, HooksMixin

_bus = EventBus()
_log: list[str] = []
_bus.subscribe("run.start", lambda e: _log.append("run.start"))
_bus.subscribe("tool.*", lambda e: _log.append(e.type))
_bus.subscribe("*", lambda e: None)  # global wildcard

_agent_hooks = Agent(provider="anthropic")
_fired: list[str] = []

@_agent_hooks.on("run.end")
def _on_end(event: AgentEvent):
    _fired.append(event.type)

assert callable(_agent_hooks.on)
assert callable(_agent_hooks.off)
assert callable(_agent_hooks.once)
assert isinstance(_agent_hooks._event_bus, EventBus)
print("✓ PR 20: Events + Hooks")


# =============================================================================
# PR 13 — TenantContext
# =============================================================================

# Uncomment when PR 13 lands:
#
# import os, asyncio
# from ninetrix import TenantContext, tenant_scope
# from ninetrix._internals.tenant import get_tenant
#
# # Tier 1: auto-init from env (single-tenant scripts — user does nothing)
# # If NINETRIX_API_KEY=test were set before import, get_tenant() would already be set.
#
# # Tier 2: middleware pattern (SaaS — set once per request, not per agent call)
# async def tenant_test():
#     assert get_tenant() is None   # before scope: not set
#     async with tenant_scope(TenantContext(workspace_id="ws-123", api_key="nxt_test")):
#         t = get_tenant()
#         assert t is not None and t.workspace_id == "ws-123"
#     assert get_tenant() is None   # after scope: restored
#
#     # Nested scopes work — inner overrides, outer restores
#     async with tenant_scope(TenantContext(workspace_id="outer")):
#         async with tenant_scope(TenantContext(workspace_id="inner")):
#             assert get_tenant().workspace_id == "inner"
#         assert get_tenant().workspace_id == "outer"
#
# # asyncio.run(tenant_test())
# # Tier 3: AgentSandbox(tenant=TenantContext(...)) — sandbox handles scope internally
# print("✓ PR 13: TenantContext")


# =============================================================================
# PR 22 — AgentClient + RemoteAgent (AgentProtocol polymorphism)
# =============================================================================

from ninetrix import Agent, AgentClient, RemoteAgent
from ninetrix._internals.types import AgentProtocol

_local  = Agent(name="analyst", provider="anthropic")
_docker = AgentClient("http://analyst:9000", name="analyst")
_cloud  = RemoteAgent("my-workspace/analyst", api_key="nxt_test")

assert isinstance(_local,  AgentProtocol), "Agent must satisfy AgentProtocol"
assert isinstance(_docker, AgentProtocol), "AgentClient must satisfy AgentProtocol"
assert isinstance(_cloud,  AgentProtocol), "RemoteAgent must satisfy AgentProtocol"
assert _docker.name == "analyst"
assert _cloud.name == "analyst"
print("✓ PR 22: AgentProtocol polymorphism")


# =============================================================================
# PR 21 — Streaming
# =============================================================================

import inspect as _inspect
from ninetrix import Agent, StreamEvent, StreamingRunner

_stream_agent = Agent(provider="anthropic")
assert _inspect.isasyncgenfunction(_stream_agent.stream)

_se = StreamEvent(type="token", content="hello")
assert _se.type == "token"
assert _se.structured_output is None  # new field

assert StreamingRunner is not None
print("✓ PR 21: Streaming")


# =============================================================================
# PR 23 — MCPToolSource + ComposioToolSource
# =============================================================================

from ninetrix import MCPToolSource, ComposioToolSource
from ninetrix.runtime.dispatcher import ToolDispatcher, _extract_mcp_result

_mcp = MCPToolSource("http://mcp-gateway:8080", token="dev-secret", workspace_id="ws1")
_composio = ComposioToolSource(apps=["GITHUB"], api_key="comp_test")
assert _mcp.tool_definitions() == []      # not initialized yet
assert _composio.tool_definitions() == []
assert _extract_mcp_result({"content": [{"type": "text", "text": "ok"}]}) == "ok"
_dispatcher = ToolDispatcher([_mcp, _composio])
assert len(_dispatcher._sources) == 2
print("✓ PR 23: MCPToolSource + ComposioToolSource")


# =============================================================================
# PR 23b — Workflow (sequential + parallel)
# =============================================================================

# Uncomment when PR 23 lands:
#
# import asyncio
# from ninetrix import Agent, Workflow
#
# planner = Agent(name="planner", provider="anthropic", model="claude-haiku-4-5-20251001")
# writer  = Agent(name="writer",  provider="anthropic", model="claude-sonnet-4-6")
#
# @Workflow
# async def pipeline(topic: str) -> str:
#     r1 = await planner.arun(f"Outline a report on: {topic}")
#     r2 = await writer.arun(f"Write based on this outline:\n{r1.output}")
#     return r2.output
#
# # Parallel
# async def parallel_test():
#     r1, r2 = await Workflow.parallel(
#         planner.arun("topic A"),
#         writer.arun("topic B"),
#     )
#     assert r1.output and r2.output
#
# print("✓ PR 23: Workflow")


# =============================================================================
# PR 24 — Planner (plan-then-execute mode)
# =============================================================================

from ninetrix import Planner
from ninetrix.runtime.runner import RunnerConfig

assert Planner is not None

_planner = Planner(RunnerConfig(name="bot"))
_plan = _planner.build_execution_prompt(
    "Do something",
    {"goal": "accomplish the task", "steps": ["step 1", "step 2"]},
)
assert "step 1" in _plan
assert "Execution Plan" in _plan or "Plan" in _plan

# Agent with planned execution mode
from ninetrix import Agent
_a = Agent(provider="anthropic", execution_mode="planned")
assert _a.config.execution_mode == "planned"

# Team block moved to PR 26
# from ninetrix import Team
# billing = Agent(name="billing", provider="anthropic", ...)
# support = Agent(name="support", provider="anthropic", ...)
# team = Team(agents=[billing, support], router_model="claude-haiku-4-5-20251001")
# result = team.run("I was charged twice")
# assert result.routed_to == "billing"

print("✓ PR 24: Planner")


# =============================================================================
# PR 25 — Workflow (sequential + parallel + fan_out + reduce + branch)
# =============================================================================

from ninetrix import Workflow, WorkflowBudgetTracker, WorkflowContext

assert Workflow is not None
assert WorkflowContext is not None
assert WorkflowBudgetTracker is not None

# Basic decorator — no parens
@Workflow
async def _wf_simple(msg: str) -> str:
    return msg + "!"

assert isinstance(_wf_simple, __import__("ninetrix").WorkflowRunner)

# With kwargs
@Workflow(name="custom", max_budget=5.0)
async def _wf_budget(msg: str) -> str:
    return msg

assert _wf_budget.__name__ == "custom"
assert _wf_budget._max_budget_usd == 5.0

# WorkflowBudgetTracker
import asyncio as _asyncio

_bt = WorkflowBudgetTracker(10.0)
_asyncio.run(_bt.charge(3.0, "agent"))
assert _bt.spent == 3.0
assert _bt.remaining == 7.0

print("✓ PR 25: Workflow + WorkflowBudgetTracker")


# =============================================================================
# PR 29 — Testing utilities
# =============================================================================

# Uncomment when PR 29 lands:
#
# import asyncio
# from ninetrix import Agent, Tool, MockTool, AgentSandbox
#
# mock_search = MockTool("search_web", returns=[{"title": "Paris", "url": "x.com"}])
#
# agent = Agent(
#     provider="anthropic",
#     model="claude-sonnet-4-6",
#     tools=[mock_search],
# )
#
# async def sandbox_test():
#     async with AgentSandbox(agent) as sb:
#         result = await sb.run("What is the capital of France?")
#     assert sb.cost_usd < 0.01
#     mock_search.assert_called_once()
#
# # asyncio.run(sandbox_test())
# print("✓ PR 29: MockTool + AgentSandbox")


# =============================================================================
# PR 19 — YAML round-trip (moved from original PR 28 — YAML is core)
# =============================================================================

# Uncomment when PR 28 lands:
#
# from ninetrix import Agent
# agent = Agent(
#     name="analyst",
#     provider="anthropic",
#     model="claude-sonnet-4-6",
#     role="Financial analyst",
#     tools=[get_price],
# )
# yaml_str = agent.to_yaml()
# assert "claude-sonnet-4-6" in yaml_str
# assert "get_price" in yaml_str
#
# agent2 = Agent.from_yaml(yaml_str)
# assert agent2.name == "analyst"
# print("✓ PR 28: YAML round-trip")


# =============================================================================
# PR 31 — agent.serve()
# =============================================================================

# Uncomment when PR 31 lands:
#
# from ninetrix import Agent
# agent = Agent(provider="anthropic", model="claude-sonnet-4-6")
# # agent.serve(port=9000)   # starts FastAPI /invoke + /health in-process
# print("✓ PR 31: agent.serve()")


print("\nAll uncommented examples passed.")
