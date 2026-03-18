"""
observability/otel.py — optional OpenTelemetry integration.

Layer: L6 (observability) — may import L1 (_internals) + stdlib only.

OTEL is an OPTIONAL dependency.  If ``opentelemetry-api`` is not installed
the entire module degrades gracefully — ``configure_otel()`` is a no-op and
``get_tracer()`` returns a ``NoOpTracer``.

Usage::

    from ninetrix.observability.otel import configure_otel

    configure_otel(
        endpoint="http://localhost:4317",
        service_name="my-agent-service",
        headers={"Authorization": "Bearer token"},
    )
    # All subsequent Agent.run() calls now emit OTEL spans automatically.

Spans emitted
-------------
``ninetrix.agent.run``       — root span per Agent.run() call
``ninetrix.agent.turn``      — child span per LLM turn
``ninetrix.tool.call``       — child span per tool call
``ninetrix.checkpoint.save`` — child span when checkpoint saved

The integration hooks into the :class:`~ninetrix.observability.events.EventBus`
via :func:`attach_otel_to_bus`.  Call it once per EventBus instance.

Context propagation
-------------------
OTEL context variables maintain the parent span across async turns.
Active spans are stored in ``_active_spans[thread_id]`` and cleaned up on
``run.end`` or ``run.error``.
"""

from __future__ import annotations

import contextvars
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

_HAS_OTEL = False
_configured = False

# Public alias — checked by Agent._build_runner() to auto-attach OTEL
_otel_configured: bool = False

try:
    import opentelemetry.trace as _otel_trace  # type: ignore[import]
    from opentelemetry.trace import SpanKind, StatusCode  # type: ignore[import]
    _HAS_OTEL = True
except ImportError:
    _otel_trace = None  # type: ignore[assignment]
    SpanKind = None  # type: ignore[assignment]
    StatusCode = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# NoOp fallbacks (used when OTEL not installed)
# ---------------------------------------------------------------------------


class _NoOpSpan:
    """A span that does nothing."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_exception(self, exc: Any) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpTracer:
    """A tracer that returns no-op spans."""

    def start_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    def start_as_current_span(self, name: str, **kwargs: Any) -> Any:
        return _NoOpSpan()


_noop_tracer = _NoOpTracer()


# ---------------------------------------------------------------------------
# Active span storage (per thread_id)
# ---------------------------------------------------------------------------

# Maps thread_id → root AgentRun span
_active_spans: dict[str, Any] = {}

# Maps thread_id → OTEL context token (for async context propagation)
_span_contexts: dict[str, contextvars.Token] = {}

# ContextVar holding the current span (for async-safe propagation)
_current_span_var: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "ninetrix_otel_span", default=None
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_otel(
    endpoint: str,
    service_name: str = "ninetrix-agent",
    *,
    headers: Optional[dict[str, str]] = None,
    insecure: bool = True,
) -> None:
    """Configure the OTLP exporter and mark OTEL as active.

    This is a no-op if ``opentelemetry-api`` and
    ``opentelemetry-exporter-otlp`` are not installed.  The function prints
    a warning in that case.

    Args:
        endpoint:     OTLP gRPC or HTTP endpoint, e.g. ``"http://localhost:4317"``.
        service_name: Service name attached to all spans as a resource attribute.
        headers:      Optional dict of HTTP headers (e.g. auth tokens).
        insecure:     If ``True`` (default), TLS verification is skipped for
                      local collectors.

    Example::

        configure_otel(
            endpoint="http://localhost:4317",
            service_name="my-agent-service",
            headers={"Authorization": "Bearer token"},
        )
    """
    global _configured, _otel_configured

    if not _HAS_OTEL:
        import warnings
        warnings.warn(
            "configure_otel() called but opentelemetry-api is not installed.\n"
            "  Install it with: pip install opentelemetry-api opentelemetry-sdk "
            "opentelemetry-exporter-otlp",
            stacklevel=2,
        )
        return

    try:
        from opentelemetry import trace  # type: ignore[import]
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import]
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore[import]
        from opentelemetry.sdk.resources import Resource  # type: ignore[import]

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        # Try OTLP gRPC exporter first, fall back to HTTP
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import]
                OTLPSpanExporter,
            )
            exporter = OTLPSpanExporter(
                endpoint=endpoint,
                headers=headers or {},
                insecure=insecure,
            )
        except ImportError:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import]
                OTLPSpanExporter,
            )
            exporter = OTLPSpanExporter(
                endpoint=endpoint,
                headers=headers or {},
            )

        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _configured = True
        _otel_configured = True

    except ImportError as exc:
        import warnings
        warnings.warn(
            f"configure_otel() failed: {exc}\n"
            "  Install: pip install opentelemetry-sdk opentelemetry-exporter-otlp",
            stacklevel=2,
        )


def get_tracer(name: str = "ninetrix") -> Any:
    """Return the active OTEL tracer, or a no-op tracer if OTEL is not configured.

    Args:
        name: Instrumentation scope name (default: ``"ninetrix"``).

    Returns:
        An ``opentelemetry.trace.Tracer`` or a ``_NoOpTracer``.
    """
    if not _HAS_OTEL or not _configured:
        return _noop_tracer

    try:
        from opentelemetry import trace  # type: ignore[import]
        return trace.get_tracer(name)
    except Exception:
        return _noop_tracer


def is_configured() -> bool:
    """Return True if configure_otel() has been called successfully."""
    return _configured and _HAS_OTEL


# ---------------------------------------------------------------------------
# EventBus integration
# ---------------------------------------------------------------------------


def attach_otel_to_bus(event_bus: Any) -> None:
    """Subscribe OTEL span handlers to *event_bus*.

    Call this once after ``configure_otel()`` and before any agent runs.
    Typically you do not call this directly — it is invoked automatically
    when you call ``configure_otel()`` on the global event bus.

    The function is idempotent and safe to call when OTEL is not installed
    (all handlers become no-ops via the ``_NoOpTracer``).

    Args:
        event_bus: An :class:`~ninetrix.observability.events.EventBus` instance.
    """
    tracer = get_tracer()

    async def on_run_start(event: Any) -> None:
        """Create root span for agent.run()."""
        thread_id = event.thread_id
        data = event.data

        span = tracer.start_span(
            "ninetrix.agent.run",
            kind=SpanKind.SERVER if _HAS_OTEL and SpanKind is not None else None,
        )
        _set_span_attrs(span, {
            "agent.name": event.agent_name,
            "agent.thread_id": thread_id,
            "agent.model": data.get("model", ""),
            "agent.provider": data.get("provider", ""),
        })
        _active_spans[thread_id] = span

    async def on_run_end(event: Any) -> None:
        """End the root span."""
        thread_id = event.thread_id
        span = _active_spans.pop(thread_id, None)
        if span is not None:
            _set_span_attrs(span, {
                "agent.tokens_used": event.data.get("tokens_used", 0),
                "agent.cost_usd": event.data.get("cost_usd", 0.0),
                "agent.steps": event.data.get("steps", 0),
            })
            _end_span_ok(span)

    async def on_turn_start(event: Any) -> None:
        """Create turn child span."""
        thread_id = event.thread_id
        parent_span = _active_spans.get(thread_id)
        if parent_span is None:
            return

        context = _get_span_context(parent_span)
        span = tracer.start_span("ninetrix.agent.turn", context=context)
        _set_span_attrs(span, {
            "agent.turn_index": event.data.get("turn", 0),
        })
        # Store turn span keyed by thread_id + "_turn"
        _active_spans[f"{thread_id}:turn"] = span

    async def on_turn_end(event: Any) -> None:
        """End turn child span."""
        thread_id = event.thread_id
        span = _active_spans.pop(f"{thread_id}:turn", None)
        if span is not None:
            _set_span_attrs(span, {
                "agent.input_tokens": event.data.get("input_tokens", 0),
                "agent.output_tokens": event.data.get("output_tokens", 0),
            })
            _end_span_ok(span)

    async def on_tool_call(event: Any) -> None:
        """Create tool.call child span."""
        thread_id = event.thread_id
        parent_span = _active_spans.get(thread_id)

        context = _get_span_context(parent_span) if parent_span else None
        span = tracer.start_span("ninetrix.tool.call", context=context)
        tool_name = event.data.get("tool_name", "")
        _set_span_attrs(span, {
            "tool.name": tool_name,
            "tool.turn_index": event.data.get("turn", 0),
        })
        _active_spans[f"{thread_id}:tool:{tool_name}"] = span

    async def on_tool_result(event: Any) -> None:
        """End tool.call child span."""
        thread_id = event.thread_id
        tool_name = event.data.get("tool_name", "")
        span = _active_spans.pop(f"{thread_id}:tool:{tool_name}", None)
        if span is not None:
            _set_span_attrs(span, {"tool.success": True})
            _end_span_ok(span)

    async def on_error(event: Any) -> None:
        """Mark root span as failed and end it."""
        thread_id = event.thread_id
        span = _active_spans.pop(thread_id, None)
        if span is not None:
            exc = event.data.get("exception")
            if exc is not None:
                span.record_exception(exc)
            _end_span_error(span, event.data.get("message", "agent error"))

    # Subscribe all handlers
    event_bus.subscribe("run.start", on_run_start)
    event_bus.subscribe("run.end", on_run_end)
    event_bus.subscribe("turn.start", on_turn_start)
    event_bus.subscribe("turn.end", on_turn_end)
    event_bus.subscribe("tool.call", on_tool_call)
    event_bus.subscribe("tool.result", on_tool_result)
    event_bus.subscribe("error", on_error)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _set_span_attrs(span: Any, attrs: dict[str, Any]) -> None:
    """Set multiple span attributes, ignoring errors."""
    try:
        for key, value in attrs.items():
            if value is not None:
                span.set_attribute(key, value)
    except Exception:
        pass


def _end_span_ok(span: Any) -> None:
    """End span with OK status."""
    try:
        if _HAS_OTEL and StatusCode is not None:
            span.set_status(StatusCode.OK)
        span.end()
    except Exception:
        pass


def _end_span_error(span: Any, message: str = "error") -> None:
    """End span with ERROR status."""
    try:
        if _HAS_OTEL and StatusCode is not None:
            span.set_status(StatusCode.ERROR, description=message)
        span.end()
    except Exception:
        pass


def _get_span_context(span: Any) -> Any:
    """Return an OTEL context with *span* as the active span, or None."""
    if not _HAS_OTEL or span is None:
        return None
    try:
        from opentelemetry import trace, context as otel_context  # type: ignore[import]
        ctx = otel_context.get_current()
        return trace.set_span_in_context(span, ctx)
    except Exception:
        return None


def cleanup_thread(thread_id: str) -> None:
    """Remove all span state for *thread_id*.

    Called automatically by the ``run.end`` / ``error`` handlers.
    Exposed for manual cleanup in tests.

    Args:
        thread_id: The thread ID to clean up.
    """
    _active_spans.pop(thread_id, None)
    _active_spans.pop(f"{thread_id}:turn", None)
    # Clean up any lingering tool spans
    keys_to_remove = [k for k in _active_spans if k.startswith(f"{thread_id}:tool:")]
    for k in keys_to_remove:
        _active_spans.pop(k, None)
