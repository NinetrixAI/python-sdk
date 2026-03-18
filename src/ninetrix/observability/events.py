"""
observability/events.py — AgentEvent dataclass + EventBus pub/sub engine.

Layer: L6 (observability) — may import L1 (_internals) + stdlib only.

All agent lifecycle events flow through EventBus.  Hooks, OTel integration,
and any user-defined listeners all subscribe through the same interface.

Event type naming convention
-----------------------------
Events use dot-separated namespaces:

    run.start       — agent.run() has been called, before first turn
    run.end         — agent.run() has completed (success or error)
    turn.start      — before each LLM call
    turn.end        — after each LLM response
    tool.call       — before a tool is dispatched
    tool.result     — after a tool returns its result
    budget.warning  — when budget crosses the warning threshold
    error           — on any unhandled exception during a run

Wildcard matching
-----------------
EventBus supports two kinds of wildcard subscriptions:

    ``"*"``         — all events regardless of type
    ``"tool.*"``    — all events whose type starts with ``"tool."``

Handlers can be sync or async (coroutines are awaited automatically).
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# AgentEvent
# ---------------------------------------------------------------------------


@dataclass
class AgentEvent:
    """A single lifecycle event emitted by the agent runtime.

    Attributes:
        type:        Dot-namespaced event type (e.g. ``"tool.call"``).
        thread_id:   Conversation / run ID.
        agent_name:  Name of the agent that emitted the event.
        data:        Event-specific payload dict.
    """

    type: str
    thread_id: str = ""
    agent_name: str = ""
    data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


Handler = Callable[[AgentEvent], Any]


class EventBus:
    """Lightweight pub/sub bus for agent lifecycle events.

    All agent events flow through a single ``EventBus`` instance.  The bus
    supports exact type matching, prefix wildcards (``"tool.*"``), and a
    global wildcard (``"*"``).

    Handlers may be sync or async — coroutines are ``await``-ed automatically.

    Example::

        bus = EventBus()

        async def on_tool(event: AgentEvent):
            print(f"Tool called: {event.data['tool_name']}")

        bus.subscribe("tool.call", on_tool)
        await bus.emit(AgentEvent(type="tool.call", data={"tool_name": "search"}))
    """

    def __init__(self) -> None:
        # Maps subscription key (exact type or wildcard pattern) → list[Handler]
        self._subscribers: dict[str, list[Handler]] = {}

    # ------------------------------------------------------------------
    # Subscribe / unsubscribe
    # ------------------------------------------------------------------

    def subscribe(self, event_type: str, handler: Handler) -> None:
        """Register *handler* for *event_type*.

        *event_type* can be:
        - An exact type: ``"tool.call"``
        - A prefix wildcard: ``"tool.*"``
        - The global wildcard: ``"*"``

        Args:
            event_type: Pattern to match.
            handler:    Sync or async callable receiving :class:`AgentEvent`.
        """
        self._subscribers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: str, handler: Handler) -> None:
        """Remove *handler* from *event_type*.  No-op if not registered.

        Args:
            event_type: The pattern the handler was registered under.
            handler:    The exact callable to remove.
        """
        bucket = self._subscribers.get(event_type)
        if bucket and handler in bucket:
            bucket.remove(handler)
            if not bucket:
                del self._subscribers[event_type]

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------

    async def emit(self, event: AgentEvent) -> None:
        """Emit *event* to all matching subscribers.

        Matching order:
        1. Exact-type subscribers (e.g. ``"tool.call"``).
        2. Prefix-wildcard subscribers whose prefix matches (e.g. ``"tool.*"``
           matches any event starting with ``"tool."``).
        3. Global wildcard subscribers (``"*"``).

        Handlers are called in registration order.  Async handlers are awaited.

        Args:
            event: The event to dispatch.
        """
        await self._call_bucket(event, event.type)

        # Prefix wildcards: "tool.*" matches "tool.call", "tool.result", …
        for key in list(self._subscribers):
            if key.endswith(".*"):
                prefix = key[:-2]  # strip ".*"
                if event.type.startswith(prefix + ".") or event.type == prefix:
                    await self._call_bucket(event, key)

        # Global wildcard
        if "*" in self._subscribers:
            await self._call_bucket(event, "*")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _call_bucket(self, event: AgentEvent, key: str) -> None:
        for handler in list(self._subscribers.get(key, [])):
            result = handler(event)
            if inspect.isawaitable(result):
                await result

    def clear(self) -> None:
        """Remove all subscribers.  Intended for test teardown."""
        self._subscribers.clear()

    def subscriber_count(self, event_type: str | None = None) -> int:
        """Return the number of registered handlers.

        Args:
            event_type: If given, count only handlers for that pattern.
                        If ``None``, count all handlers across all patterns.
        """
        if event_type is not None:
            return len(self._subscribers.get(event_type, []))
        return sum(len(v) for v in self._subscribers.values())

    def __repr__(self) -> str:
        total = self.subscriber_count()
        patterns = len(self._subscribers)
        return f"EventBus(patterns={patterns}, handlers={total})"
