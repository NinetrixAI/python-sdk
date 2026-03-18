# Changelog

All notable changes to `ninetrix-sdk` are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [Unreleased]

### Added — PR 32: Agent Lifecycle Methods (serve / build / deploy)

- `Agent.serve(host, port, reload, app_only)` — starts a FastAPI/uvicorn HTTP server exposing the agent on `POST /invoke`, `GET /info`, `GET /health`, and `POST /stream` (SSE). Pass `app_only=True` to get the FastAPI app without starting uvicorn (useful for ASGI embedding and testing).
- `Agent.build(tag, push)` — serialises the agent to a temporary `agentfile.yaml` and calls `ninetrix build` as a subprocess. Returns `{"image", "yaml_path", "success"}`.
- `Agent.deploy(workspace_id, api_key, region)` — exports the agent to YAML and POSTs to `POST /v1/deployments` on Ninetrix Cloud. Resolves credentials from explicit args → `TenantContext` → environment variables. Returns `{"deployment_id", "url", "status"}`.
- `ninetrix.agent.server` module: `create_agent_app(agent)` and `serve_agent(agent, ...)`.
- `serve` optional extras in `pyproject.toml`: `pip install 'ninetrix-sdk[serve]'` installs `fastapi` and `uvicorn`.
- New tests: `tests/test_lifecycle.py` (25 tests; 6 skipped when FastAPI not installed).

### Added — PR 31: OpenTelemetry Integration

- `ninetrix.observability.otel` module with graceful degradation when `opentelemetry-api` is not installed.
- `configure_otel(endpoint, service_name, headers, insecure)` — sets up OTLP exporter and marks OTEL as active. No-op with a warning when `opentelemetry-api` is missing.
- `get_tracer(name)` — returns the active OTEL tracer, or a `_NoOpTracer` if not configured.
- `attach_otel_to_bus(event_bus)` — subscribes span handlers to an `EventBus` instance. Handles `run.start`, `run.end`, `turn.start`, `turn.end`, `tool.call`, `tool.result`, `error` events.
- Spans emitted: `ninetrix.agent.run` (root), `ninetrix.agent.turn`, `ninetrix.tool.call`, `ninetrix.checkpoint.save`.
- `_NoOpSpan` and `_NoOpTracer` safe fallbacks that do nothing when OTEL is absent.
- `cleanup_thread(thread_id)` — removes all span state for a completed run.
- `otel` optional extras: `pip install 'ninetrix-sdk[otel]'` installs the OTEL SDK + OTLP exporter.
- New exports in `ninetrix.__init__`: `configure_otel`, `get_tracer`, `attach_otel_to_bus`.
- New tests: `tests/test_otel.py` (23 tests; all pass without opentelemetry installed).

### Added — PR 30: Testing Framework (MockTool + AgentSandbox)

- `ninetrix.testing` package: `MockTool` and `AgentSandbox`.
- `MockTool(name, return_value, side_effect, schema)` — registers a fake tool in `ToolRegistry` backed by an async dispatch function. Supports `return_value`, callable `side_effect`, or exception `side_effect`. Tracks all calls with `call_count`, `calls`, `reset()`. Assertion helpers: `assert_called()`, `assert_called_with(**kwargs)`, `assert_call_count(n)`, `assert_not_called()`.
- `AgentSandbox(agent, script, tool_calls_script, provider)` — async context manager that replaces the agent's provider with `MockProvider` and its dispatcher with `SandboxedDispatcher`. `script=` accepts plain-text response list; `tool_calls_script=` accepts rich turn dicts with `tool_calls`. Assertions: `assert_tool_called(name)`, `assert_tool_call_count(name, n)`, `assert_no_tool_called(name)`. Properties: `tool_calls`, `turn_count`.
- `MockProvider` — cycles through a script of canned LLM responses. Supports both `script=` (plain text) and `tool_calls_script=` (with tool invocations).
- `SandboxedToolSource` / `SandboxedDispatcher` — dispatcher that only routes to registered `MockTool`s and logs all calls.
- `SandboxResult` — wraps `AgentResult` with a `.success` property.
- New exports in `ninetrix.__init__`: `MockTool`, `AgentSandbox`.
- New tests: `tests/test_testing.py` (48 tests).

---
