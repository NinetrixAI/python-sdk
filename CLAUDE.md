# SDK CLAUDE.md — Session Anchor

Read this at the start of every session before touching any code.

---

## Current Status

**Current PR: 25** — `workflow/context.py` + `workflow/workflow.py`
**PR 19 done:** `export/writer.py` + `export/loader.py` — agent_to_yaml, load_agent_from_yaml, Agent.to_yaml/from_yaml
**PR 18 done:** `agent/config.py` + `agent/agent.py` + `agent/introspection.py` — Agent, AgentConfig, info/validate/dry_run
**PR 17 done:** `checkpoint/base.py` + `checkpoint/memory.py` — Checkpointer ABC + InMemoryCheckpointer
**PR 16 done:** `runtime/runner.py` — AgentRunner, RunnerConfig, direct mode, structured output parse+retry
**PR 15 done:** `tools/context.py` + `runtime/dispatcher.py` — ToolContext, LocalToolSource, RegistryToolSource, ToolDispatcher
**PR 14 done:** `runtime/history.py` + `runtime/budget.py` — MessageHistory, BudgetTracker, BudgetUsage
**PR 13 done:** `_internals/tenant.py` — TenantContext, set_tenant, get_tenant, require_tenant, tenant_scope
**PR 9 done:** `observability/errors.py` — ErrorContext, error_context
**PR 8 done:** `observability/telemetry.py` — TelemetryCollector, record_event, opt-out
**PR 7 done:** `observability/logger.py` — NinetrixLogger, enable_debug, get_logger
**PR 6 done:** `providers/` — AnthropicAdapter, OpenAIAdapter, GoogleAdapter, LiteLLMAdapter, FallbackProviderAdapter
**PR 5 done:** `_internals/http.py` + `_internals/lifespan.py`
**PR 4 done:** `_internals/auth.py`
**PR 3 done:** `_internals/config.py`
**PR 2 done:** `_internals/networking.py`
**PR 1 done:** `_internals/types.py` + `py.typed`
**Phase 1 done:** `@Tool` decorator (`tool.py`, `registry.py`, `schema.py`, `discover.py`)
**Full plan:** `../plans/sdk-phase-2-3-full-architecture.md`

Update "Current PR" here after every session.

---

## Package layout

```
sdk/
├── src/ninetrix/          ← installable package
│   ├── __init__.py        ← public API surface (re-exports only)
│   ├── py.typed           ← PEP 561 marker (create in PR 1)
│   ├── tool.py            ← Phase 1 (done)
│   ├── registry.py        ← Phase 1 (done)
│   ├── schema.py          ← Phase 1 (done)
│   ├── discover.py        ← Phase 1 (done)
│   ├── _internals/        ← private kernel (L1)
│   ├── tools/             ← @Tool ecosystem (L2)
│   ├── runtime/           ← agentic loop (L3)
│   ├── providers/         ← LLM adapters (L4)
│   ├── middleware/        ← request pipeline (L5)
│   ├── observability/     ← events, hooks, otel, streaming (L6)
│   ├── checkpoint/        ← persistence (L7)
│   ├── agent/             ← Agent class (L8)
│   ├── workflow/          ← Workflow + Team (L8)
│   ├── client/            ← AgentClient + RemoteAgent (L9)
│   ├── testing/           ← MockTool, AgentSandbox (L9)
│   └── export/            ← YAML round-trip (L9)
├── tests/
│   ├── usage_examples.py  ← north-star file (uncomment as PRs land)
│   └── test_*.py          ← existing Phase 1 tests
└── pyproject.toml
```

---

## Layer import rules — NEVER violate

```
L1  _internals/     → stdlib only. No ninetrix imports.
L2  tools/          → L1 + stdlib
L3  runtime/        → L1 + L2 + stdlib
L4  providers/      → L1 + stdlib + provider SDKs (anthropic, openai, google)
L5  middleware/     → L1 + L2 + L3 + stdlib
L6  observability/  → L1 + stdlib
L7  checkpoint/     → L1 + stdlib + asyncpg
L8  agent/          → all above
    workflow/       → all above
L9  client/         → L8 and below
    testing/        → L8 and below
    export/         → L8 and below
```

If you need to import something from a higher layer, the type/function belongs in a lower layer.
Check for circular imports after every PR with: `python -c "import ninetrix"`

---

## Non-negotiable conventions

- **Nothing flat at root** — every new file goes in a domain folder. Exception: existing Phase 1 files stay where they are until a dedicated migration PR.
- **`@Tool` returns `F` unchanged** — the decorator only registers, never wraps. Callers keep full type inference.
- **`AgentResult` is `Generic[T_Output]`** — `output: T_Output` (str by default, Pydantic model when `output_type=` set).
- **`AgentProtocol`** — `Agent`, `AgentClient`, `RemoteAgent` all satisfy this Protocol. `Workflow` and `Team` accept `AgentProtocol`, never `Agent` specifically.
- **`TenantContext` auto-init** — SDK sets it automatically from `NINETRIX_WORKSPACE_ID` + `NINETRIX_API_KEY` at import. End users never call `set_tenant()`. Only middleware authors use `tenant_scope()` (once, in one place). Tests use `AgentSandbox(tenant=...)`. See Addendum 8 in plan.
- **`Agent._build_runner()`** — single assembly point. All runtime objects (dispatcher, checkpointer, runner) created here.
- **`run()` event-loop safe** — uses `_run_in_thread()` guard. Never calls `asyncio.run()` directly.
- **All provider exceptions wrapped** — `anthropic.APIError` → `ProviderError`. Raw third-party exceptions never surface.
- **Error messages: what + why + how to fix** — every `raise` must include all three.
- **`py.typed` marker** — must exist at `src/ninetrix/py.typed` (empty file, PEP 561).
- **`RegistryToolSource` lazy init** — schemas fetched at `dispatcher.initialize()`, not at construction. Always call `await dispatcher.initialize()` before first run.
- **YAML is core** — `Agent.from_yaml()` / `agent.to_yaml()` live at PR 19 (not last). Every agent must be serializable from day one of the public API.

---

## `__init__.py` update rule

After each PR that adds a public symbol, add it to `__init__.py` with `as Name` re-export pattern:
```python
from ninetrix.agent.agent import Agent as Agent
```
The `as Name` pattern makes symbols explicitly public (mypy, pyright, pylance all respect this).

---

## PR sequence (32 total)

| PR | Files | Notes | Status |
|----|-------|-------|--------|
| 1 | `_internals/types.py` + `py.typed` | + `AgentProtocol` Protocol | ✅ |
| 2 | `_internals/networking.py` | RetryPolicy, CircuitBreaker | ✅ |
| 3 | `_internals/config.py` | NinetrixConfig layered resolution | ✅ |
| 4 | `_internals/auth.py` | CredentialStore | ✅ |
| 5 | `_internals/http.py` + `_internals/lifespan.py` | httpx singleton + SIGTERM | ✅ |
| 6 | `providers/` | All adapters + fallback + attachments + structured output | ⬜ |
| 7 | `observability/logger.py` | | ⬜ |
| 8 | `observability/telemetry.py` | | ⬜ |
| 9 | `observability/errors.py` | | ⬜ |
| 10 | `middleware/pipeline.py` | | ⬜ |
| 11 | `middleware/builtins.py` | | ⬜ |
| 12 | `middleware/tools.py` | | ⬜ |
| **13** | **`_internals/tenant.py`** | **TenantContext + set_tenant + tenant_scope — NEW** | ✅ |
| 14 | `runtime/history.py` + `runtime/budget.py` | | ✅ |
| 15 | `tools/context.py` + `runtime/dispatcher.py` | Local + **RegistryToolSource** (lazy skill loading) | ✅ |
| 16 | `runtime/runner.py` | Direct mode + structured output | ✅ |
| 17 | `checkpoint/base.py` + `checkpoint/memory.py` | | ⬜ |
| 18 | `agent/config.py` + `agent/agent.py` + `agent/introspection.py` | `_build_runner()`, validate/dry_run/info | ✅ |
| **19** | **`export/writer.py` + `export/loader.py`** | **Moved from 28 — YAML is core** | ✅ |
| 20 | `observability/events.py` + `observability/hooks.py` | | ✅ |
| 21 | `runtime/streaming.py` + `observability/streaming.py` + `Agent.stream()` | | ✅ |
| **22** | **`client/local.py` + `client/remote.py`** | **AgentClient + RemoteAgent — full AgentProtocol polymorphism** | ✅ |
| 23 | `runtime/dispatcher.py` | MCP + Composio | ✅ |
| 24 | `runtime/planner.py` | | ✅ |
| 25 | `workflow/context.py` + `workflow/workflow.py` | Sequential/parallel/fan_out/branch + WorkflowBudgetTracker | ⬜ |
| 26 | `workflow/team.py` | Accepts `list[AgentProtocol]` | ⬜ |
| 27 | `tools/toolkit.py` | | ⬜ |
| 28 | `checkpoint/postgres.py` | + SELECT FOR UPDATE | ⬜ |
| 29 | `workflow/workflow.py` durable=True | | ⬜ |
| 30 | `testing/` | MockTool, AgentSandbox (+ tenant= param), replay, assertions | ⬜ |
| 31 | `observability/otel.py` | | ⬜ |
| 32 | `agent.serve()` + `agent.build()` + `agent.deploy()` | | ⬜ |

---

## Planned gap reviews

- After PR 5 — review provider + middleware assumptions
- After PR 15 — review agent/workflow assumptions
- After PR 22 — review testing/export assumptions

---

## Testing rule — non-negotiable

Every PR must ship a `tests/test_<module>.py` **in the same session**, before marking the PR done.
- PRs 1–5: plain pytest, no mocks
- PR 6+ providers: `unittest.mock.AsyncMock` on the SDK — no real API calls
- PR 13+ async: `pytest-asyncio`, mark tests `@pytest.mark.asyncio`
- Real-LLM tests: `@pytest.mark.integration`, skip unless API key env var is set
- Full suite must pass (`pytest tests/`) before bumping "Current PR" in this file

## Session start ritual

1. Read this file (auto-loaded)
2. Check current PR number above
3. Read `sdk.md` in memory for status
4. Read the relevant PR section from the plan (by line range — don't load full file)
5. Read only the files you will modify
6. Implement
7. Run: `python -c "import ninetrix"` (circular import check)
8. Run full test suite: `pytest tests/` (must pass before proceeding)
9. Uncomment the relevant lines in `tests/usage_examples.py`
10. Update "Current PR" in this file
11. Update sdk.md memory: mark PR done
12. Update docs.md memory: mark PR row ✅ in the docs map table
