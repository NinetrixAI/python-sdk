"""
runtime/dispatcher.py — routes tool calls to the correct source.

Layer: L3 (runtime) — may import L1 (_internals), L2 (tools), stdlib only.

Sources shipped in this module:
  LocalToolSource    — @Tool-decorated Python functions (async-safe via asyncio.to_thread)
  RegistryToolSource — Ninetrix Skill Registry (lazy HTTP, proxied calls)
  MCPToolSource      — Ninetrix MCP Gateway (JSON-RPC 2.0, PR 23)
  ComposioToolSource — Composio SDK (conditional import, PR 23)
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import warnings
from abc import ABC, abstractmethod
from typing import Any

_log = logging.getLogger("ninetrix.runtime.dispatcher")


def _truncate(d: dict, max_len: int = 200) -> str:
    s = str(d)
    return s if len(s) <= max_len else s[:max_len] + "…"

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

        client = get_http_client()
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

        client = get_http_client()
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
# MCPToolSource
# ---------------------------------------------------------------------------

class MCPToolSource(ToolSource):
    """Dispatches tool calls through the Ninetrix MCP Gateway.

    Sends JSON-RPC 2.0 requests to ``POST /v1/mcp/{workspace_id}`` on the
    gateway.  Tool schemas are fetched lazily at ``initialize()`` time and
    converted from MCP format to OpenAI-compatible format.

    Args:
        gateway_url:  Base URL of the MCP gateway, e.g.
                      ``"http://mcp-gateway:8080"``.
        token:        Bearer token for gateway authentication.
        workspace_id: Workspace identifier used in the URL path.

    Example::

        source = MCPToolSource(
            gateway_url="http://mcp-gateway:8080",
            token="dev-secret",
            workspace_id="my-workspace",
        )
        dispatcher = ToolDispatcher([source])
        await dispatcher.initialize()   # fetches tool schemas
        result = await dispatcher.call("slack__send_message", {"text": "hi"})
    """

    def __init__(
        self,
        gateway_url: str,
        token: str,
        workspace_id: str,
    ) -> None:
        self._url = gateway_url.rstrip("/")
        self._token = token
        self._workspace_id = workspace_id
        self._tool_schemas: list[dict] = []
        self._tool_names: set[str] = set()
        self._req_id = 0

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    async def initialize(self) -> None:
        """Fetch tool schemas from the MCP Gateway via ``tools/list``."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {},
        }
        client = get_http_client()
        resp = await client.post(
            f"{self._url}/v1/mcp/{self._workspace_id}",
            json=payload,
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=30.0,
        )
        if resp.status_code >= 400:
            raise ToolError(
                f"MCPToolSource: tools/list failed with HTTP {resp.status_code}.\n"
                f"  URL: {self._url}/v1/mcp/{self._workspace_id}\n"
                f"  Response: {resp.text[:200]}\n"
                "  Fix: check gateway_url, token, and workspace_id."
            )

        data = resp.json()
        if "error" in data and data["error"]:
            raise ToolError(
                f"MCPToolSource: tools/list returned JSON-RPC error: "
                f"{data['error'].get('message', data['error'])}.\n"
                "  Fix: check that the MCP gateway is running and the token is valid."
            )

        tools = data.get("result", {}).get("tools", [])
        # Convert MCP ToolSchema → OpenAI-compatible function-calling format
        self._tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("inputSchema") or {
                        "type": "object", "properties": {},
                    },
                },
            }
            for t in tools
            if isinstance(t, dict) and t.get("name")
        ]
        self._tool_names = {t["name"] for t in tools if t.get("name")}

    def tool_definitions(self) -> list[dict]:
        return list(self._tool_schemas)

    def handles(self, tool_name: str) -> bool:
        return tool_name in self._tool_names

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Dispatch a tool call via ``tools/call`` JSON-RPC."""
        _log.debug(f"dispatch tool={tool_name} source=mcp")
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        client = get_http_client()
        resp = await client.post(
            f"{self._url}/v1/mcp/{self._workspace_id}",
            json=payload,
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=60.0,
        )
        if resp.status_code >= 400:
            raise ToolError(
                f"MCPToolSource: tools/call '{tool_name}' failed with "
                f"HTTP {resp.status_code}.\n"
                f"  Response: {resp.text[:200]}"
            )

        data = resp.json()
        if "error" in data and data["error"]:
            error = data["error"]
            code = error.get("code", 0)
            msg = error.get("message", str(error))
            # -32010: integration not connected — include auth_url hint
            auth_url = (error.get("data") or {}).get("auth_url", "")
            hint = f"\n  Connect it at: {auth_url}" if auth_url else ""
            raise ToolError(
                f"MCP tool '{tool_name}' error (code {code}): {msg}.{hint}\n"
                "  Fix: check tool name, arguments, and that the MCP worker is connected."
            )

        return _extract_mcp_result(data.get("result", ""))

    def __repr__(self) -> str:
        return (
            f"MCPToolSource(gateway={self._url!r}, "
            f"workspace={self._workspace_id!r}, "
            f"tools={len(self._tool_names)})"
        )


def _extract_mcp_result(result: Any) -> str:
    """Extract a plain string from an MCP tools/call result.

    Handles:
    - Standard MCP: ``{"content": [{"type": "text", "text": "..."}], ...}``
    - Plain string
    - Arbitrary dicts/values (str-converted)
    """
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        content = result.get("content")
        if content and isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:
                    parts.append(str(item))
            return "\n".join(parts) if parts else str(result)
        if "result" in result:
            return str(result["result"])
    return str(result)


# ---------------------------------------------------------------------------
# ComposioToolSource
# ---------------------------------------------------------------------------

class ComposioToolSource(ToolSource):
    """Dispatches tool calls via the Composio SDK.

    Requires the ``composio-openai`` package::

        pip install composio-openai

    If the package is not installed, ``initialize()`` succeeds but returns no
    schemas, and any ``call()`` attempt raises :exc:`~ninetrix.ToolError` with
    an install hint.

    Args:
        apps:      List of Composio app names, e.g. ``["GITHUB", "SLACK"]``.
        api_key:   Composio API key.  Falls back to the ``COMPOSIO_API_KEY``
                   environment variable when not provided.
        entity_id: Composio entity/user ID (default: ``"default"``).

    Example::

        source = ComposioToolSource(apps=["GITHUB"], api_key="comp_...")
        dispatcher = ToolDispatcher([source])
        await dispatcher.initialize()
        result = await dispatcher.call("GITHUB_CREATE_ISSUE", {"title": "bug"})
    """

    def __init__(
        self,
        apps: list[str],
        api_key: str = "",
        entity_id: str = "default",
    ) -> None:
        self._apps = apps
        self._api_key = api_key
        self._entity_id = entity_id
        self._tool_schemas: list[dict] = []
        self._tool_names: set[str] = set()
        self._toolset: Any = None

    async def initialize(self) -> None:
        """Load Composio tool schemas.  No-op if SDK is not installed."""
        try:
            from composio_openai import ComposioToolSet  # type: ignore[import-untyped]
        except ImportError:
            warnings.warn(
                "ComposioToolSource: composio-openai is not installed. "
                "Run: pip install composio-openai",
                stacklevel=2,
            )
            return

        key = self._api_key or os.environ.get("COMPOSIO_API_KEY", "") or None
        self._toolset = ComposioToolSet(api_key=key, entity_id=self._entity_id)

        try:
            from composio import App  # type: ignore[import-untyped]
            app_values = [App(a) for a in self._apps]
        except Exception:
            app_values = self._apps  # type: ignore[assignment]

        try:
            schemas = await asyncio.to_thread(
                self._toolset.get_tools, apps=app_values
            )
            self._tool_schemas = schemas  # already OpenAI-compatible
            self._tool_names = {
                s["function"]["name"] for s in schemas if "function" in s
            }
        except Exception as exc:
            warnings.warn(
                f"ComposioToolSource: failed to load schemas for {self._apps}: {exc}",
                stacklevel=2,
            )

    def tool_definitions(self) -> list[dict]:
        return list(self._tool_schemas)

    def handles(self, tool_name: str) -> bool:
        return tool_name in self._tool_names

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a Composio action via the SDK."""
        _log.debug(f"dispatch tool={tool_name} source=composio")
        if self._toolset is None:
            raise ToolError(
                f"Composio tool '{tool_name}' cannot be called.\n"
                "  Why: composio-openai is not installed or initialize() was not called.\n"
                "  Fix: pip install composio-openai, then await dispatcher.initialize()."
            )

        try:
            from composio import Action  # type: ignore[import-untyped]
            action = Action(tool_name)
        except Exception:
            action = tool_name  # type: ignore[assignment]

        try:
            result = await asyncio.to_thread(
                self._toolset.execute_action,
                action=action,
                params=arguments,
                entity_id=self._entity_id,
            )
            return str(result) if result is not None else "(done)"
        except Exception as exc:
            raise ToolError(
                f"Composio tool '{tool_name}' failed: {type(exc).__name__}: {exc}.\n"
                "  Fix: check the tool name, arguments, and your Composio API key."
            ) from exc

    def __repr__(self) -> str:
        return (
            f"ComposioToolSource(apps={self._apps!r}, "
            f"tools={len(self._tool_names)})"
        )


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
