"""
agent/server.py — FastAPI HTTP server for serving an Agent over HTTP.

Layer: L8 (agent) — may import all lower layers + stdlib.

FastAPI is an OPTIONAL dependency.  If not installed, :func:`create_agent_app`
raises :exc:`~ninetrix._internals.types.ConfigurationError` with an install hint.

Endpoints
---------
``POST /invoke``    — run the agent synchronously, return AgentResult JSON.
``GET  /info``      — return AgentInfo JSON.
``GET  /health``    — return ``{"status": "ok", "agent": name}``.
``POST /stream``    — SSE endpoint that streams StreamEvent objects.

Request body (for /invoke and /stream)::

    {
        "input": "user message",
        "thread_id": "optional-id",
        "context": {}
    }

Usage (internal — called by Agent.serve())::

    from ninetrix.agent.server import create_agent_app, serve_agent

    app = create_agent_app(agent)           # returns FastAPI app
    serve_agent(agent, host="0.0.0.0", port=9000)  # blocking
"""

from __future__ import annotations

from typing import Any, Optional

from ninetrix._internals.types import ConfigurationError


# ---------------------------------------------------------------------------
# FastAPI detection
# ---------------------------------------------------------------------------

_HAS_FASTAPI = False

try:
    import fastapi  # noqa: F401
    _HAS_FASTAPI = True
except ImportError:
    pass


def _require_fastapi() -> None:
    if not _HAS_FASTAPI:
        raise ConfigurationError(
            "FastAPI is required to use Agent.serve().\n"
            "  Why: the 'fastapi' package is not installed.\n"
            "  Fix: pip install 'ninetrix-sdk[serve]'  or  pip install fastapi uvicorn"
        )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_agent_app(agent: Any) -> Any:
    """Create and return a FastAPI application that serves *agent* over HTTP.

    Args:
        agent: An :class:`~ninetrix.agent.agent.Agent` instance.

    Returns:
        A configured ``fastapi.FastAPI`` application.

    Raises:
        ConfigurationError: If FastAPI is not installed.

    Example::

        app = create_agent_app(agent)
        # Use with uvicorn or any ASGI server
        # uvicorn.run(app, host="0.0.0.0", port=9000)
    """
    _require_fastapi()

    import fastapi
    from fastapi import Request
    from fastapi.responses import JSONResponse, StreamingResponse

    app = fastapi.FastAPI(
        title=f"Ninetrix Agent: {agent.name}",
        description=f"HTTP interface for the '{agent.name}' agent.",
        version="1.0.0",
    )

    # ── /health ──────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health() -> dict:
        """Liveness check.  Always returns 200 when the server is running."""
        return {"status": "ok", "agent": agent.name}

    # ── /info ────────────────────────────────────────────────────────────────

    @app.get("/info")
    async def info() -> dict:
        """Return a read-only summary of the agent's configuration."""
        agent_info = agent.info()
        return {
            "name": agent_info.name,
            "provider": agent_info.provider,
            "model": agent_info.model,
            "tool_count": agent_info.tool_count,
            "local_tools": agent_info.local_tools,
            "mcp_tools": agent_info.mcp_tools,
            "composio_tools": agent_info.composio_tools,
            "has_persistence": agent_info.has_persistence,
            "execution_mode": agent_info.execution_mode,
            "max_budget_usd": agent_info.max_budget_usd,
            "max_turns": agent_info.max_turns,
            "system_prompt_chars": agent_info.system_prompt_chars,
        }

    # ── /invoke ──────────────────────────────────────────────────────────────

    @app.post("/invoke")
    async def invoke(request: Request) -> JSONResponse:
        """Run the agent and return the result as JSON.

        Request body::

            {"input": "...", "thread_id": "optional", "context": {}}

        Returns :class:`~ninetrix._internals.types.AgentResult` as JSON.
        """
        body = await request.json()
        user_input: str = body.get("input", "")
        thread_id: Optional[str] = body.get("thread_id") or None

        if not user_input:
            return JSONResponse(
                {"error": "Missing 'input' field in request body."},
                status_code=422,
            )

        try:
            result = await agent.arun(user_input, thread_id=thread_id)
            return JSONResponse(result.to_dict())
        except Exception as exc:
            return JSONResponse(
                {"error": str(exc), "type": type(exc).__name__},
                status_code=500,
            )

    # ── /stream ──────────────────────────────────────────────────────────────

    @app.post("/stream")
    async def stream(request: Request) -> StreamingResponse:
        """Stream agent events as Server-Sent Events (SSE).

        Request body::

            {"input": "...", "thread_id": "optional", "context": {}}

        Yields::

            data: {"type": "token", "content": "..."}\n\n
            data: {"type": "tool_start", "tool_name": "..."}\n\n
            data: {"type": "done", "tokens_used": N}\n\n
        """
        import json as _json

        body = await request.json()
        user_input: str = body.get("input", "")
        thread_id: Optional[str] = body.get("thread_id") or None

        if not user_input:
            async def error_stream():
                yield 'data: {"error": "Missing input"}\n\n'
            return StreamingResponse(error_stream(), media_type="text/event-stream")

        async def event_stream():
            try:
                async for event in agent.stream(user_input, thread_id=thread_id):
                    payload = {
                        "type": event.type,
                        "content": event.content,
                    }
                    if event.tool_name:
                        payload["tool_name"] = event.tool_name
                    if event.tool_args:
                        payload["tool_args"] = event.tool_args
                    if event.tool_result:
                        payload["tool_result"] = event.tool_result
                    if event.tokens_used:
                        payload["tokens_used"] = event.tokens_used
                    if event.cost_usd:
                        payload["cost_usd"] = event.cost_usd
                    if event.error:
                        payload["error"] = str(event.error)
                    if event.structured_output is not None:
                        try:
                            if hasattr(event.structured_output, "model_dump"):
                                payload["structured_output"] = event.structured_output.model_dump()
                            else:
                                payload["structured_output"] = event.structured_output
                        except Exception:
                            payload["structured_output"] = str(event.structured_output)
                    yield f"data: {_json.dumps(payload)}\n\n"
            except Exception as exc:
                yield f'data: {_json.dumps({"type": "error", "error": str(exc)})}\n\n'

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


# ---------------------------------------------------------------------------
# serve_agent — blocking server entry point
# ---------------------------------------------------------------------------


def serve_agent(
    agent: Any,
    *,
    host: str = "0.0.0.0",
    port: int = 9000,
    reload: bool = False,
) -> None:
    """Start a uvicorn server serving *agent* over HTTP.

    This is a blocking call — it does not return until the server is stopped
    (e.g. via ``Ctrl+C`` or SIGTERM).

    Args:
        agent:  :class:`~ninetrix.agent.agent.Agent` to serve.
        host:   Bind address (default: ``"0.0.0.0"``).
        port:   TCP port (default: ``9000``).
        reload: Enable hot-reload (useful during development).

    Raises:
        ConfigurationError: If FastAPI or uvicorn is not installed.

    Example::

        agent = Agent(name="bot", provider="anthropic", role="assistant")
        agent.serve(host="0.0.0.0", port=9000)
    """
    _require_fastapi()

    try:
        import uvicorn  # type: ignore[import]
    except ImportError as exc:
        raise ConfigurationError(
            "uvicorn is required to use Agent.serve().\n"
            "  Why: the 'uvicorn' package is not installed.\n"
            "  Fix: pip install uvicorn"
        ) from exc

    app = create_agent_app(agent)
    uvicorn.run(app, host=host, port=port, reload=reload)
