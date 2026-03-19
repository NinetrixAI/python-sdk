"""
ninetrix._internals.trace
=========================
L1 kernel — stdlib only.

ContextVars that propagate a reporter and a parent trace-id through the
async call chain without any explicit parameter threading.

How it works
------------
When a ``WorkflowRunner`` or ``Team`` starts a run it calls
``reporter_scope(reporter, trace_id=<its-own-trace>)``.  Every ``Agent.arun()``
call that happens *inside* the scope automatically inherits both values:

- ``get_reporter()``     → the ``RunnerReporter`` to POST events to
- ``get_current_trace()``→ the workflow / team trace-id to use as
                           ``parent_trace_id`` in agent ``thread_started`` events

For a standalone ``Agent.arun()`` (no parent scope) both return ``None``.
``Agent.arun()`` then tries to auto-resolve a reporter and creates its own
scope if successful.

Layer note
----------
This module is L1 so it can be imported by *every* layer without circularity.
The reporter is typed as ``Any`` — the concrete ``RunnerReporter`` class lives
at L6 (observability) and satisfies the ``ReporterProtocol`` defined in types.py.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, AsyncGenerator

# ---------------------------------------------------------------------------
# ContextVars
# ---------------------------------------------------------------------------

#: Active RunnerReporter instance (or None).  Set by reporter_scope().
_current_reporter: ContextVar[Any] = ContextVar("_current_reporter", default=None)

#: Trace-id of the enclosing Workflow or Team run (or None for top-level).
#: AgentRunner reads this as ``parent_trace_id`` in its ``thread_started`` event.
_current_trace: ContextVar[str | None] = ContextVar("_current_trace", default=None)


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def get_reporter() -> Any:
    """Return the active RunnerReporter, or None if none is set."""
    return _current_reporter.get()


def get_current_trace() -> str | None:
    """Return the enclosing scope's trace-id, or None."""
    return _current_trace.get()


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def reporter_scope(
    reporter: Any,
    trace_id: str | None = None,
) -> AsyncGenerator[None, None]:
    """Async context manager — sets reporter (and optionally trace-id) for
    the duration of the block, then restores the previous values.

    Args:
        reporter: A :class:`~ninetrix.observability.reporter.RunnerReporter`
                  instance.  Stored as ``Any`` to avoid a circular import.
        trace_id: The caller's own trace-id.  Child ``Agent.arun()`` calls
                  will read this as their ``parent_trace_id``.
                  Pass ``None`` for standalone agent runs (no parent scope).
    """
    token_r = _current_reporter.set(reporter)
    token_t = _current_trace.set(trace_id) if trace_id is not None else None
    try:
        yield
    finally:
        _current_reporter.reset(token_r)
        if token_t is not None:
            _current_trace.reset(token_t)
