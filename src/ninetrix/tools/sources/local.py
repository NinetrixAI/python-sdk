"""
tools/sources/local.py — LocalToolSource for @Tool-decorated Python functions.

Layer: L2 (tools) — imports L1 (_internals) + stdlib only.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

from ninetrix.tools.base import ToolSource
from ninetrix.registry import ToolDef
from ninetrix._internals.types import ToolError
from ninetrix.tools.context import ToolContext, _is_context_param

_log = logging.getLogger("ninetrix.runtime.dispatcher")


def _truncate(d: dict, max_len: int = 200) -> str:
    s = str(d)
    return s if len(s) <= max_len else s[:max_len] + "\u2026"


class LocalToolSource(ToolSource):
    """Dispatches @Tool-decorated Python functions.

    Sync functions are run in a thread pool via ``asyncio.to_thread`` so they
    never block the event loop.  Async functions are awaited directly.

    If the function declares a ``ctx: ToolContext`` parameter, a
    :class:`~ninetrix.tools.context.ToolContext` is built from *tool_context*
    and injected.  The ``ctx`` parameter is excluded from the LLM schema.

    Args:
        tool_defs: List of :class:`~ninetrix.registry.ToolDef` instances.
        tool_context: Static key->resource mapping injected into every call.
    """

    source_type = "local"

    def __init__(
        self,
        tool_defs: list[ToolDef],
        tool_context: dict[str, Any] | None = None,
    ) -> None:
        self._tools: dict[str, ToolDef] = {td.name: td for td in tool_defs}
        self._tool_context: dict[str, Any] = tool_context or {}

    def tool_definitions(self) -> list[dict]:
        """Return OpenAI-compatible function-calling schemas for all tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": td.name,
                    "description": td.description,
                    "parameters": td.parameters,
                },
            }
            for td in self._tools.values()
        ]

    def handles(self, tool_name: str) -> bool:
        return tool_name in self._tools

    async def call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        thread_id: str = "",
        agent_name: str = "",
        turn_index: int = 0,
        extra_context: dict[str, Any] | None = None,
    ) -> str:
        """Execute a local tool.

        Args:
            tool_name:     Name of the tool to call.
            arguments:     LLM-provided arguments (ToolContext excluded).
            thread_id:     Current thread ID (injected into ToolContext).
            agent_name:    Agent name (injected into ToolContext).
            turn_index:    Current turn index (injected into ToolContext).
            extra_context: Per-request context merged over the static context.

        Returns:
            String representation of the tool's return value, or ``"(done)"``
            if the function returns None.

        Raises:
            ToolError: If the tool is not found or raises an exception.
        """
        if tool_name not in self._tools:
            raise ToolError(
                f"Tool '{tool_name}' is not registered in LocalToolSource. "
                "Why: the tool was not decorated with @Tool or not passed to "
                "LocalToolSource(). "
                "Fix: check the tool name and ensure it is registered."
            )

        td = self._tools[tool_name]
        kwargs = dict(arguments)

        # Inject ToolContext if the function signature requires it
        ctx_param = _find_context_param(td.fn)
        if ctx_param is not None:
            merged_store: dict[str, Any] = {**self._tool_context, **(extra_context or {})}
            ctx = ToolContext(
                thread_id=thread_id,
                agent_name=agent_name,
                turn_index=turn_index,
                _store=merged_store,
            )
            kwargs[ctx_param] = ctx

        _log.debug(f"dispatch tool={tool_name} source=local args={_truncate(arguments)}")
        try:
            if inspect.iscoroutinefunction(td.fn):
                result = await td.fn(**kwargs)
            else:
                result = await asyncio.to_thread(td.fn, **kwargs)
        except Exception as exc:
            raise ToolError(
                f"Tool '{tool_name}' raised {type(exc).__name__}: {exc}. "
                "Why: the tool function threw an unhandled exception. "
                "Fix: check the tool implementation or the arguments passed."
            ) from exc

        result_str = str(result) if result is not None else "(done)"
        _log.debug(f"tool={tool_name} result={result_str[:120]}")
        return result_str


def _find_context_param(fn: Any) -> str | None:
    """Return the name of the ToolContext parameter, or None if absent.

    Uses ``typing.get_type_hints()`` to resolve string annotations
    (produced by ``from __future__ import annotations``).  Falls back to
    raw annotation strings and checks the class name as a last resort.
    """
    import typing as _t

    # get_type_hints resolves string annotations using the function's module globals
    hints: dict[str, Any] = {}
    try:
        hints = _t.get_type_hints(fn)
    except Exception:
        try:
            hints = dict(getattr(fn, "__annotations__", None) or {})
        except Exception:
            return None

    for name, annotation in hints.items():
        if name == "return":
            continue
        # Resolved annotation: direct class identity or marker attribute
        if _is_context_param(annotation):
            return name
        # Unresolved string annotation (get_type_hints failed and we fell back)
        if isinstance(annotation, str) and annotation.split(".")[-1] == "ToolContext":
            return name

    return None
