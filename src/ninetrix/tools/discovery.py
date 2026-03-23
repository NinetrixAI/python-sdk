"""
tools/discovery.py — plugin discovery for community ToolSource implementations.

Layer: L2 (tools) — imports L1 (_internals) + stdlib only.

Built-in sources (mcp, openapi, composio, local) are registered automatically.
Community sources are discovered via Python entry_points::

    # In a community plugin's pyproject.toml:
    [project.entry-points."ninetrix.tool_sources"]
    jira = "ninetrix_source_jira:JiraToolSource"

After ``pip install ninetrix-source-jira``, the JiraToolSource is auto-discovered
and available for use with ``source: jira://...`` in agentfile.yaml.
"""

from __future__ import annotations

import importlib.metadata
import logging
from dataclasses import dataclass
from typing import Any

_log = logging.getLogger("ninetrix.tools.discovery")

_ENTRY_POINT_GROUP = "ninetrix.tool_sources"

# Scheme -> ToolSource class mapping
_SOURCE_REGISTRY: dict[str, type] = {}

_builtins_registered = False


def register_source(scheme: str, cls: type) -> None:
    """Register a ToolSource class for a URI scheme (e.g. ``"mcp"``).

    If a source is already registered for the scheme, a warning is logged
    and the existing registration is kept.
    """
    if scheme in _SOURCE_REGISTRY:
        existing = _SOURCE_REGISTRY[scheme]
        if existing is not cls:
            _log.warning(
                f"Scheme '{scheme}' already registered to {existing.__name__}, "
                f"ignoring {cls.__name__}"
            )
        return
    _SOURCE_REGISTRY[scheme] = cls


def get_source_class(scheme: str) -> type | None:
    """Look up a ToolSource class by URI scheme."""
    return _SOURCE_REGISTRY.get(scheme)


def registered_schemes() -> list[str]:
    """Return all registered scheme names."""
    return list(_SOURCE_REGISTRY.keys())


def auto_register_builtins() -> None:
    """Register all built-in source schemes.

    Safe to call multiple times — registration is idempotent.
    """
    global _builtins_registered
    if _builtins_registered:
        return

    from ninetrix.tools.sources.mcp import MCPToolSource
    from ninetrix.tools.sources.composio import ComposioToolSource
    from ninetrix.tools.sources.openapi import OpenAPIToolSource

    register_source("mcp", MCPToolSource)
    register_source("composio", ComposioToolSource)
    register_source("openapi", OpenAPIToolSource)
    # "local" and "builtin" are handled specially — not via URI scheme lookup

    _builtins_registered = True


def discover_and_register_plugins() -> list[SourcePlugin]:
    """Scan installed packages for ``ninetrix.tool_sources`` entry points.

    Returns:
        List of discovered plugins (including those that failed to load).
    """
    plugins: list[SourcePlugin] = []
    try:
        eps = importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP)
    except Exception:
        return plugins

    for ep in eps:
        try:
            cls = ep.load()
            scheme = ep.name
            register_source(scheme, cls)
            plugins.append(SourcePlugin(
                name=ep.name,
                source_class=cls,
                package=ep.dist.name if ep.dist else "unknown",
                error=None,
            ))
            _log.info(
                f"Discovered plugin: {scheme}:// -> {cls.__name__} "
                f"(from {ep.dist.name if ep.dist else 'unknown'})"
            )
        except Exception as exc:
            _log.warning(f"Failed to load plugin '{ep.name}': {exc}")
            plugins.append(SourcePlugin(
                name=ep.name,
                source_class=None,
                package=ep.dist.name if ep.dist else "unknown",
                error=str(exc),
            ))

    return plugins


def reset_registry() -> None:
    """Clear all registrations. Used by tests."""
    global _builtins_registered
    _SOURCE_REGISTRY.clear()
    _builtins_registered = False


@dataclass
class SourcePlugin:
    """Metadata about a discovered tool source plugin."""
    name: str
    source_class: type | None
    package: str
    error: str | None
