"""
observability/hooks.py — HooksMixin: @agent.on() / off() / once() decorator API.

Layer: L6 (observability) — may import L1 (_internals) + stdlib only.

HooksMixin is mixed into Agent to give it event hook registration methods.
It manages its own EventBus instance and exposes a clean decorator API.

Usage (on Agent after PR 18)::

    @agent.on("tool.call")
    async def log_tool(event):
        print(f"Calling tool: {event.data['tool_name']}")

    @agent.once("run.end")
    def finish(event):
        print(f"Done. Cost: ${event.data.get('cost_usd', 0):.4f}")

    agent.off("tool.call", log_tool)   # remove a specific handler
"""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ninetrix.observability.events import AgentEvent, Handler


class HooksMixin:
    """Mixin that adds lifecycle hook registration to Agent.

    Mixed into :class:`~ninetrix.agent.agent.Agent` — users call
    ``agent.on()``, ``agent.off()``, and ``agent.once()`` instead of
    interacting with the :class:`~ninetrix.observability.events.EventBus`
    directly.

    Subclasses must call ``super().__init__()`` before using hook methods,
    which initialises ``self._event_bus``.
    """

    def __init__(self, **kwargs: Any) -> None:
        from ninetrix.observability.events import EventBus
        self._event_bus: "EventBus" = EventBus()
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Public hook API
    # ------------------------------------------------------------------

    def on(self, event_type: str) -> Callable:
        """Register a handler for *event_type*.

        Can be used as a decorator or called directly::

            # Decorator style
            @agent.on("tool.call")
            async def log_tool(event):
                print(event.data["tool_name"])

            # Direct call
            agent.on("tool.call")(log_tool)

        Supported event types:
            ``"run.start"``      — before first LLM turn
            ``"run.end"``        — after final response (or error)
            ``"turn.start"``     — before each LLM call
            ``"turn.end"``       — after each LLM response
            ``"tool.call"``      — before each tool invocation
            ``"tool.result"``    — after each tool returns
            ``"budget.warning"`` — when budget crosses threshold
            ``"error"``          — on any unhandled exception
            ``"tool.*"``         — wildcard: all ``tool.*`` events
            ``"*"``              — global wildcard: every event

        Args:
            event_type: Event name or wildcard pattern to subscribe to.

        Returns:
            A decorator that registers the decorated callable and returns it
            unchanged.
        """
        def decorator(fn: "Handler") -> "Handler":
            self._event_bus.subscribe(event_type, fn)
            return fn
        return decorator

    def off(self, event_type: str, handler: "Handler") -> None:
        """Remove a previously registered *handler* for *event_type*.

        No-op if the handler was never registered.

        Args:
            event_type: Pattern the handler was registered under.
            handler:    The exact callable to remove.
        """
        self._event_bus.unsubscribe(event_type, handler)

    def once(self, event_type: str) -> Callable:
        """Register a handler that fires exactly once then auto-removes.

        After the first matching event the handler is unsubscribed
        automatically.  Safe to call from sync or async handlers.

        Usage::

            @agent.once("run.end")
            async def on_first_end(event):
                print("First run completed!")

        Args:
            event_type: Event type to subscribe to (wildcards supported).

        Returns:
            Decorator that wraps the callable in a one-shot wrapper.
        """
        def decorator(fn: "Handler") -> "Handler":
            # Wrapper captures a mutable reference so it can remove itself.
            wrapper_ref: list["Handler"] = []

            def _wrapper(event: "AgentEvent") -> Any:
                self._event_bus.unsubscribe(event_type, wrapper_ref[0])
                return fn(event)

            wrapper_ref.append(_wrapper)
            self._event_bus.subscribe(event_type, _wrapper)
            return fn  # return the original fn (not the wrapper)
        return decorator
