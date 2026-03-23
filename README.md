# ninetrix-sdk

Python SDK for [Ninetrix](https://ninetrix.io) — build, compose, and deploy AI agents in pure Python.

```bash
pip install ninetrix-sdk
```

---

## What it does

The Ninetrix SDK lets you define agents, tools, workflows, and multi-agent teams entirely in Python. Agents are portable — serialize to YAML, run locally, or deploy to Ninetrix Cloud.

---

## Core Concepts

### Agents

Define an agent with a role, tools, and an LLM provider. Run it synchronously, asynchronously, or as a streaming event source.

Agents support structured output (Pydantic models), token/cost budgets, and human-in-the-loop approval gates.

### Tools

The `@Tool` decorator turns any Python function into an agent tool. The function signature is automatically converted to a JSON Schema — no manual wiring needed.

Group related tools with `Toolkit` for cleaner organization.

### Tool Sources

Agents can use tools from multiple sources, all mixed together:

| Source | Description |
|--------|-------------|
| `@Tool` functions | Local Python functions |
| MCP servers | Via MCP Gateway (JSON-RPC) |
| OpenAPI specs | Any REST API with an OpenAPI 3.x spec |
| Composio | Composio SDK integrations |
| Community plugins | `pip install ninetrix-source-*` (auto-discovered) |

The plugin system uses standard Python `entry_points` — anyone can publish a new tool source.

### Workflows

The `@Workflow` decorator defines sequential pipelines that chain agents together. Workflows support:

- **Durable execution** — checkpoint every step, resume on crash
- **Declarative steps** — `run_step(name, fn)` for clean step definitions
- **Fan-out** — `map(agent, items, concurrency)` for parallel execution
- **Early exit** — `terminate(reason)` for explicit abort

### Teams

`Team` provides LLM-based dynamic routing across multiple agents. The router agent picks the best specialist for each request — no manual if/else chains.

### Ninetrix Context

The `Ninetrix` factory class sets provider and model defaults once. All agents, teams, and workflows created from the context inherit the configuration.

---

## Providers

Built-in LLM adapters with a unified interface:

- Anthropic (Claude)
- OpenAI (GPT)
- Google (Gemini)
- LiteLLM (100+ models)
- Fallback chains (try providers in order)

All provider exceptions are wrapped — raw third-party errors never surface.

---

## Observability

- **EventBus** — pub-sub event system for all agent lifecycle events
- **OpenTelemetry** — opt-in tracing (graceful no-op if OTEL SDK not installed)
- **Debug mode** — pretty-print all events to stderr
- **Dashboard telemetry** — `RunnerReporter` posts events for trace visualization
- **Token tracking** — per-turn and per-run budget monitoring with cost estimates

---

## Persistence

- **InMemoryCheckpointer** — for local dev and testing
- **PostgresCheckpointer** — production-grade with `SELECT FOR UPDATE`
- Durable workflows automatically resume from the last successful step on crash

---

## Testing

Built-in testing utilities for deterministic agent tests without real LLM calls:

- `MockTool` — fake tools with call tracking and assertions
- `MockProvider` — scripted LLM responses
- `AgentSandbox` — isolated test harness with full assertion API

---

## Deployment

Agents can be served, built, and deployed from code:

- `agent.serve()` — FastAPI HTTP server (`/invoke`, `/stream`, `/health`, `/info`)
- `agent.build()` — serialize to YAML and build a Docker image via `ninetrix build`
- `agent.deploy()` — push to Ninetrix Cloud

---

## YAML Serialization

Every agent is serializable from day one:

- `agent.to_yaml()` — export to `agentfile.yaml` format
- `Agent.from_yaml()` — load from YAML back into a live agent

This makes agents portable between code-first and YAML-first workflows.

---

## Optional Dependencies

Install only what you need:

```bash
pip install 'ninetrix-sdk[anthropic]'     # Anthropic provider
pip install 'ninetrix-sdk[openai]'        # OpenAI provider
pip install 'ninetrix-sdk[google]'        # Google provider
pip install 'ninetrix-sdk[providers]'     # All providers
pip install 'ninetrix-sdk[serve]'         # FastAPI server
pip install 'ninetrix-sdk[otel]'          # OpenTelemetry tracing
```

---

## License

Apache 2.0
