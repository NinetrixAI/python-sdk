"""
tools/sources/mcp.py — MCPToolSource for MCP Gateway JSON-RPC.

Layer: L2 (tools) — imports L1 (_internals) + stdlib only.
"""

from __future__ import annotations

import logging
from typing import Any

from ninetrix.tools.base import ToolSource
from ninetrix._internals.types import ToolError
from ninetrix._internals.http import get_http_client

_log = logging.getLogger("ninetrix.runtime.dispatcher")


class MCPToolSource(ToolSource):
    """Dispatches tool calls through the Ninetrix MCP Gateway.

    Sends JSON-RPC 2.0 requests to ``POST /v1/mcp/{org_id}`` on the
    gateway.  Tool schemas are fetched lazily at ``initialize()`` time and
    converted from MCP format to OpenAI-compatible format.

    Args:
        gateway_url:  Base URL of the MCP gateway.
        token:        Bearer token for gateway authentication.
        org_id:       Organization identifier used in the URL path.
        ctx:          Optional :class:`~ninetrix.tools.agent_context.AgentContext`.
    """

    source_type = "mcp"

    def __init__(
        self,
        gateway_url: str,
        token: str,
        org_id: str,
        ctx: Any | None = None,
    ) -> None:
        self._url = gateway_url.rstrip("/")
        self._token = token
        self._org_id = org_id
        self._ctx = ctx
        self._tool_schemas: list[dict] = []
        self._tool_names: set[str] = set()
        self._req_id = 0

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _http(self) -> Any:
        """Return the HTTP client — prefer AgentContext, fall back to singleton."""
        if self._ctx is not None and self._ctx.http is not None:
            return self._ctx.http
        return get_http_client()

    async def initialize(self) -> None:
        """Fetch tool schemas from the MCP Gateway via ``tools/list``."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {},
        }
        client = self._http()
        resp = await client.post(
            f"{self._url}/v1/mcp/{self._org_id}",
            json=payload,
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=30.0,
        )
        if resp.status_code >= 400:
            raise ToolError(
                f"MCPToolSource: tools/list failed with HTTP {resp.status_code}.\n"
                f"  URL: {self._url}/v1/mcp/{self._org_id}\n"
                f"  Response: {resp.text[:200]}\n"
                "  Fix: check gateway_url, token, and org_id."
            )

        data = resp.json()
        if "error" in data and data["error"]:
            raise ToolError(
                f"MCPToolSource: tools/list returned JSON-RPC error: "
                f"{data['error'].get('message', data['error'])}.\n"
                "  Fix: check that the MCP gateway is running and the token is valid."
            )

        tools = data.get("result", {}).get("tools", [])
        # Convert MCP ToolSchema -> OpenAI-compatible function-calling format
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
        client = self._http()
        resp = await client.post(
            f"{self._url}/v1/mcp/{self._org_id}",
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
            f"org={self._org_id!r}, "
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
