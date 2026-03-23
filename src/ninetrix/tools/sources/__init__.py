"""
tools/sources/ — built-in ToolSource implementations.

Layer: L2 (tools) — imports L1 (_internals) + stdlib only.

Each source lives in its own file for readability:
  local.py     — LocalToolSource (@Tool-decorated Python functions)
  registry.py  — RegistryToolSource (Ninetrix Skill Registry)
  mcp.py       — MCPToolSource (MCP Gateway JSON-RPC)
  composio.py  — ComposioToolSource (Composio SDK)
"""

from ninetrix.tools.sources.local import LocalToolSource as LocalToolSource
from ninetrix.tools.sources.registry import RegistryToolSource as RegistryToolSource
from ninetrix.tools.sources.mcp import MCPToolSource as MCPToolSource
from ninetrix.tools.sources.composio import ComposioToolSource as ComposioToolSource
from ninetrix.tools.sources.openapi import OpenAPIToolSource as OpenAPIToolSource

__all__ = [
    "LocalToolSource",
    "RegistryToolSource",
    "MCPToolSource",
    "ComposioToolSource",
    "OpenAPIToolSource",
]
