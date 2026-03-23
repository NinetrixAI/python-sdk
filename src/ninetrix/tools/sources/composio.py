"""
tools/sources/composio.py — ComposioToolSource for Composio SDK.

Layer: L2 (tools) — imports L1 (_internals) + stdlib only.
"""

from __future__ import annotations

import asyncio
import logging
import os
import warnings
from typing import Any

from ninetrix.tools.base import ToolSource
from ninetrix._internals.types import ToolError

_log = logging.getLogger("ninetrix.runtime.dispatcher")


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
    """

    source_type = "composio"

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
