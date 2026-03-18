"""
tools/context.py — ToolContext dependency injection for @Tool functions.

Layer: L2 (tools) — may import _internals + stdlib only.

@Tool functions that declare a ``ctx: ToolContext`` parameter receive a
populated context object at call time.  The SDK detects the parameter by
type annotation — the LLM never sees it in the tool schema.

Example::

    from ninetrix import Tool
    from ninetrix.tools.context import ToolContext

    @Tool
    def query_customers(sql: str, ctx: ToolContext) -> list[dict]:
        db = ctx.get("db")          # injected resource
        print(ctx.thread_id)        # request-scoped metadata
        return db.execute(sql)      # LLM only sees: query_customers(sql: str)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolContext:
    """Runtime context injected into @Tool functions at call time.

    Carries two kinds of data:

    * **Metadata** (``thread_id``, ``agent_name``, ``turn_index``): set by
      the runner before each tool call.
    * **Resources** (``_store``): arbitrary objects (DB connections, API
      clients, loggers) added by the caller via ``tool_context={"db": conn}``.

    Args:
        thread_id:  ID of the current conversation thread.
        agent_name: Name of the agent making the tool call.
        turn_index: Zero-based index of the current agent turn.

    The SDK detects ``ctx: ToolContext`` in function signatures by looking
    for this class.  Any parameter typed ``ToolContext`` is excluded from
    the LLM tool schema automatically.
    """

    # Marker read by schema.py — never import ToolContext there to avoid cycles.
    _is_tool_context: bool = field(default=True, init=False, repr=False, compare=False)

    thread_id: str = ""
    agent_name: str = ""
    turn_index: int = 0
    _store: dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Resource access
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Return a stored resource by key, or *default* if not found."""
        return self._store.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Return a stored resource by key; raises KeyError if not found."""
        try:
            return self._store[key]
        except KeyError:
            raise KeyError(
                f"ToolContext has no resource '{key}'. "
                "Why: the key was not added to tool_context= at agent.run() time. "
                f"Fix: pass tool_context={{'{key}': <value>}} to agent.run() or "
                "set tool_context= on AgentConfig."
            ) from None

    def __contains__(self, key: object) -> bool:
        return key in self._store


def _is_context_param(annotation: Any) -> bool:
    """Return True if *annotation* is (or looks like) a ToolContext type.

    Checks by class identity first, then by the ``_is_tool_context`` class
    attribute, allowing schema.py to call this without importing ToolContext.
    """
    if annotation is ToolContext:
        return True
    # Structural check: works when annotation was set as a string forward-ref
    # or when a subclass is used.
    return bool(getattr(annotation, "_is_tool_context", False))
