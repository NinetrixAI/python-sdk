"""
runtime/dispatcher.py — ToolDispatcher routes tool calls to the correct source.

Layer: L3 (runtime) — may import L1 (_internals), L2 (tools), stdlib only.

Source classes have been moved to ``ninetrix.tools.sources.*`` (L2).
This module re-exports them for backwards compatibility.
"""

from __future__ import annotations

import logging
from typing import Any

# ── Re-exports for backwards compatibility ────────────────────────────────
# All existing code can keep importing from here:
#   from ninetrix.runtime.dispatcher import MCPToolSource, ToolSource, ...

from ninetrix.tools.base import ToolSource as ToolSource
from ninetrix.tools.sources.local import LocalToolSource as LocalToolSource
from ninetrix.tools.sources.registry import RegistryToolSource as RegistryToolSource
from ninetrix.tools.sources.mcp import MCPToolSource as MCPToolSource
from ninetrix.tools.sources.mcp import _extract_mcp_result as _extract_mcp_result
from ninetrix.tools.sources.composio import ComposioToolSource as ComposioToolSource
from ninetrix.tools.sources.openapi import OpenAPIToolSource as OpenAPIToolSource
from ninetrix.tools.sources.local import _find_context_param as _find_context_param

_log = logging.getLogger("ninetrix.runtime.dispatcher")


# ---------------------------------------------------------------------------
# ToolDispatcher
# ---------------------------------------------------------------------------

class ToolDispatcher:
    """Routes tool calls to the correct :class:`ToolSource`.

    Aggregates schemas from all sources into a single tool list for the LLM.
    On each turn the runner passes the LLM's tool name + arguments here;
    the dispatcher finds the right source and forwards the call.

    Args:
        sources: Ordered list of :class:`ToolSource` instances.  The first
                 source whose ``handles()`` returns True wins.

    Example::

        from ninetrix.runtime.dispatcher import ToolDispatcher, LocalToolSource

        dispatcher = ToolDispatcher([LocalToolSource(_registry.all())])
        await dispatcher.initialize()
        result = await dispatcher.call("my_tool", {"param": "value"})
    """

    def __init__(self, sources: list[ToolSource]) -> None:
        self._sources = sources

    async def initialize(self) -> None:
        """Validate config on all sources, then initialize each one.

        Must be awaited before the first ``call()``.
        Raises :class:`~ninetrix.ConfigurationError` if any source fails
        validation or initialization.
        """
        for source in self._sources:
            source.validate_config()
        for source in self._sources:
            await source.initialize()

    async def shutdown(self) -> None:
        """Shutdown all sources in reverse order.

        Errors are logged but do not stop remaining shutdowns.
        """
        for source in reversed(self._sources):
            try:
                await source.shutdown()
            except Exception:
                _log.warning(
                    f"Error shutting down {type(source).__name__}",
                    exc_info=True,
                )

    async def health_check(self) -> dict[str, bool]:
        """Return per-source health status.

        Keys are ``"{ClassName}:{source_type}"`` to disambiguate multiple
        sources of the same type.
        """
        result: dict[str, bool] = {}
        for source in self._sources:
            key = f"{type(source).__name__}:{source.source_type}"
            try:
                result[key] = await source.health_check()
            except Exception:
                result[key] = False
        return result

    def all_tool_definitions(self) -> list[dict]:
        """Return the merged tool schema list across all sources."""
        result: list[dict] = []
        for source in self._sources:
            result.extend(source.tool_definitions())
        return result

    async def call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        **kwargs: Any,
    ) -> str:
        """Dispatch a tool call to the first matching source.

        Args:
            tool_name:  The tool/skill name chosen by the LLM.
            arguments:  Parsed argument dict from the LLM response.
            **kwargs:   Forwarded to :class:`LocalToolSource` (thread_id, etc.).

        Returns:
            String result from the tool.
        """
        for source in self._sources:
            if source.handles(tool_name):
                if isinstance(source, LocalToolSource):
                    return await source.call(tool_name, arguments, **kwargs)
                return await source.call(tool_name, arguments)
        return f"Error: tool '{tool_name}' is not available."

    def __repr__(self) -> str:
        source_names = [type(s).__name__ for s in self._sources]
        return f"ToolDispatcher(sources={source_names})"
