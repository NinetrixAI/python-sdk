"""
tools/base.py — ToolSource abstract base class.

Layer: L2 (tools) — imports stdlib only.

This is the plugin interface for integrating any tool ecosystem with
Ninetrix agents.  All built-in and community tool sources subclass this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ToolSource(ABC):
    """Base class for all tool sources — the plugin interface for integrating
    any tool ecosystem with Ninetrix agents.

    Built-in sources: Local, MCP, Composio, Registry, OpenAPI.
    Community sources: ``pip install ninetrix-source-X`` (entry_points discovery).

    **Required** (abstract):

    * ``tool_definitions()`` — LiteLLM-compatible schema list passed to the LLM.
    * ``handles(tool_name)``  — True if this source owns the given tool name.
    * ``call(tool_name, arguments)`` — execute the tool, return string result.

    **Lifecycle** (optional — override as needed):

    * ``initialize()``      — async setup (fetch schemas, open connections).
    * ``validate_config()`` — sync pre-flight check before initialize().
    * ``health_check()``    — async liveness probe between turns.
    * ``shutdown()``        — async cleanup on agent exit.

    Class attributes:

    * ``source_type``       — unique identifier, e.g. ``"local"``, ``"mcp"``.
    """

    source_type: str = "unknown"

    @abstractmethod
    def tool_definitions(self) -> list[dict]:
        """Return LiteLLM-compatible tool schema list."""

    @abstractmethod
    async def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool call and return a string result."""

    @abstractmethod
    def handles(self, tool_name: str) -> bool:
        """Return True if this source handles the given tool name."""

    # ── Lifecycle methods (optional — defaults are no-ops) ────────────

    async def initialize(self) -> None:
        """Async setup — called once before first use.

        Override to fetch schemas, open connections, authenticate, etc.
        Raise :class:`~ninetrix.ConfigurationError` on setup failure.
        """

    def validate_config(self) -> None:
        """Sync pre-flight validation — called before ``initialize()``.

        Override to check that required config is present.
        Raise :class:`~ninetrix.ConfigurationError` on problems.
        """

    async def health_check(self) -> bool:
        """Async liveness probe.  Return True if healthy, False otherwise."""
        return True

    async def shutdown(self) -> None:
        """Async cleanup — called on agent exit.

        Override to close connections, flush buffers, release temp files.
        """
