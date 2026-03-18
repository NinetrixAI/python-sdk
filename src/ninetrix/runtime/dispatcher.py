"""
runtime/dispatcher.py — routes tool calls to the correct source.

Layer: L3 (runtime) — may import L1 (_internals), L2 (tools), stdlib only.

Sources shipped in this module:
  LocalToolSource   — @Tool-decorated Python functions (async-safe via asyncio.to_thread)
  RegistryToolSource — Ninetrix Skill Registry (lazy HTTP, proxied calls)

Sources added in PR 23 (MCP + Composio):
  MCPToolSource
  ComposioToolSource
  TransferToolSource
"""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Any

from ninetrix.registry import ToolDef
from ninetrix._internals.types import ToolError
from ninetrix._internals.http import get_http_client
from ninetrix._internals.tenant import get_tenant
from ninetrix.tools.context import ToolContext, _is_context_param


# ---------------------------------------------------------------------------
# ToolSource — abstract base
# ---------------------------------------------------------------------------

class ToolSource(ABC):
    """Base class for a single tool source (local / registry / MCP / Composio).

    Every source exposes three methods:

    * ``tool_definitions()`` — LiteLLM-compatible schema list passed to the LLM.
    * ``handles(tool_name)``  — True if this source owns the given tool name.
    * ``call(tool_name, arguments)`` — execute the tool, return string result.

    Sources that need async initialisation (HTTP fetch at startup) should also
    implement ``initialize()``; :class:`ToolDispatcher` calls it automatically.
    """

    @abstractmethod
    def tool_definitions(self) -> list[dict]:
        """Return LiteLLM-compatible tool schema list."""

    @abstractmethod
    async def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool call and return a string result."""

    @abstractmethod
    def handles(self, tool_name: str) -> bool:
        """Return True if this source handles the given tool name."""


# ---------------------------------------------------------------------------
# LocalToolSource
# ---------------------------------------------------------------------------

class LocalToolSource(ToolSource):
    """Dispatches @Tool-decorated Python functions.

    Sync functions are run in a thread pool via ``asyncio.to_thread`` so they
    never block the event loop.  Async functions are awaited directly.

    If the function declares a ``ctx: ToolContext`` parameter, a
    :class:`~ninetrix.tools.context.ToolContext` is built from *tool_context*
    and injected.  The ``ctx`` parameter is excluded from the LLM schema.

    Args:
        tool_defs: List of :class:`~ninetrix.registry.ToolDef` instances.
        tool_context: Static key→resource mapping injected into every call.

    Example::

        source = LocalToolSource(registry.all(), tool_context={"db": conn})
        dispatcher = ToolDispatcher([source])
        result = await dispatcher.call("query_customers", {"sql": "SELECT 1"})
    """

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

        return str(result) if result is not None else "(done)"


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


# ---------------------------------------------------------------------------
# RegistryToolSource
# ---------------------------------------------------------------------------

class RegistryToolSource(ToolSource):
    """Loads skills from the Ninetrix Skill Registry on first use.

    Lazy initialisation: schemas are fetched once via HTTP at
    ``initialize()`` time.  Tool calls are proxied to the registry API —
    no local Python code is downloaded.

    Args:
        skills:       List of skill names to load.
        registry_url: Base URL of the Ninetrix Skill Registry.
        api_key:      Bearer token for registry authentication.

    Example::

        source = RegistryToolSource(
            skills=["web_search", "code_interpreter"],
            registry_url="https://registry.ninetrix.io",
            api_key="nxt_...",
        )
        dispatcher = ToolDispatcher([source])
        await dispatcher.initialize()   # fetches schemas
        result = await dispatcher.call("web_search", {"query": "Python tips"})
    """

    def __init__(
        self,
        skills: list[str],
        registry_url: str,
        api_key: str,
    ) -> None:
        self._skills = skills
        self._registry_url = registry_url.rstrip("/")
        self._api_key = api_key
        self._schemas: list[dict] = []
        self._skill_names: set[str] = set()

    async def initialize(self) -> None:
        """Fetch schemas for the requested skills from the registry."""
        tenant = get_tenant()
        headers: dict[str, str] = {"Authorization": f"Bearer {self._api_key}"}
        if tenant:
            headers["X-Workspace-ID"] = tenant.workspace_id

        async with get_http_client() as client:
            resp = await client.post(
                f"{self._registry_url}/v1/skills/schemas",
                json={"skills": self._skills},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        self._schemas = data.get("schemas", [])
        self._skill_names = {
            s["function"]["name"] for s in self._schemas if "function" in s
        }

    def tool_definitions(self) -> list[dict]:
        return list(self._schemas)

    def handles(self, tool_name: str) -> bool:
        return tool_name in self._skill_names

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Proxy the call to the registry execution endpoint."""
        tenant = get_tenant()
        headers: dict[str, str] = {"Authorization": f"Bearer {self._api_key}"}
        if tenant:
            headers["X-Workspace-ID"] = tenant.workspace_id

        async with get_http_client() as client:
            resp = await client.post(
                f"{self._registry_url}/v1/skills/call",
                json={"skill": tool_name, "arguments": arguments},
                headers=headers,
                timeout=60.0,
            )

        if resp.status_code >= 400:
            raise ToolError(
                f"Skill '{tool_name}' returned HTTP {resp.status_code}. "
                f"Arguments: {arguments}. "
                f"Response: {resp.text[:200]}. "
                "Fix: check the skill name, arguments, and registry API key."
            )

        return resp.json().get("result", "")


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

        from ninetrix.registry import _registry
        from ninetrix.runtime.dispatcher import ToolDispatcher, LocalToolSource

        dispatcher = ToolDispatcher([LocalToolSource(_registry.all())])
        await dispatcher.initialize()
        result = await dispatcher.call("my_tool", {"param": "value"})
    """

    def __init__(self, sources: list[ToolSource]) -> None:
        self._sources = sources

    async def initialize(self) -> None:
        """Call ``initialize()`` on any source that supports it.

        Must be awaited before the first ``call()``.
        """
        for source in self._sources:
            if hasattr(source, "initialize") and inspect.iscoroutinefunction(
                getattr(source, "initialize")
            ):
                await source.initialize()

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
