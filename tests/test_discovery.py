"""Tests for tools/discovery.py — plugin discovery and source registry."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ninetrix.tools.discovery import (
    register_source,
    get_source_class,
    registered_schemes,
    auto_register_builtins,
    discover_and_register_plugins,
    reset_registry,
)
from ninetrix.tools.base import ToolSource


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_registry():
    """Reset the source registry before and after each test."""
    reset_registry()
    yield
    reset_registry()


class _DummySource(ToolSource):
    source_type = "dummy"
    def tool_definitions(self): return []
    async def call(self, n, a): return ""
    def handles(self, n): return False


class _OtherSource(ToolSource):
    source_type = "other"
    def tool_definitions(self): return []
    async def call(self, n, a): return ""
    def handles(self, n): return False


# ── register_source / get_source_class ────────────────────────────────────


def test_register_and_get():
    register_source("dummy", _DummySource)
    assert get_source_class("dummy") is _DummySource


def test_get_unknown_returns_none():
    assert get_source_class("nonexistent") is None


def test_registered_schemes():
    register_source("alpha", _DummySource)
    register_source("beta", _OtherSource)
    schemes = registered_schemes()
    assert "alpha" in schemes
    assert "beta" in schemes


def test_duplicate_registration_same_class():
    """Re-registering the same class for a scheme is a no-op."""
    register_source("dup", _DummySource)
    register_source("dup", _DummySource)  # should not warn
    assert get_source_class("dup") is _DummySource


def test_collision_keeps_first(caplog):
    """If a different class tries to register the same scheme, first wins."""
    register_source("conflict", _DummySource)
    register_source("conflict", _OtherSource)

    # First registration wins
    assert get_source_class("conflict") is _DummySource


# ── auto_register_builtins ────────────────────────────────────────────────


def test_auto_register_builtins():
    auto_register_builtins()

    assert get_source_class("mcp") is not None
    assert get_source_class("composio") is not None
    assert get_source_class("openapi") is not None


def test_auto_register_builtins_idempotent():
    """Calling twice doesn't crash or duplicate."""
    auto_register_builtins()
    auto_register_builtins()

    assert get_source_class("mcp") is not None


# ── discover_and_register_plugins ─────────────────────────────────────────


def test_discover_plugins_with_mock_entry_point():
    """Simulate a community plugin being discovered via entry_points."""
    mock_ep = MagicMock()
    mock_ep.name = "custom"
    mock_ep.load.return_value = _DummySource
    mock_ep.dist = MagicMock()
    mock_ep.dist.name = "ninetrix-source-custom"

    with patch("ninetrix.tools.discovery.importlib.metadata.entry_points", return_value=[mock_ep]):
        plugins = discover_and_register_plugins()

    assert len(plugins) == 1
    assert plugins[0].name == "custom"
    assert plugins[0].source_class is _DummySource
    assert plugins[0].error is None

    # Should be registered
    assert get_source_class("custom") is _DummySource


def test_discover_plugins_handles_load_failure():
    """If a plugin fails to load, it's recorded but doesn't crash."""
    mock_ep = MagicMock()
    mock_ep.name = "broken"
    mock_ep.load.side_effect = ImportError("missing dependency")
    mock_ep.dist = MagicMock()
    mock_ep.dist.name = "ninetrix-source-broken"

    with patch("ninetrix.tools.discovery.importlib.metadata.entry_points", return_value=[mock_ep]):
        plugins = discover_and_register_plugins()

    assert len(plugins) == 1
    assert plugins[0].name == "broken"
    assert plugins[0].source_class is None
    assert "missing dependency" in plugins[0].error

    # Should NOT be registered
    assert get_source_class("broken") is None


def test_discover_plugins_empty():
    """No entry points installed — returns empty list."""
    with patch("ninetrix.tools.discovery.importlib.metadata.entry_points", return_value=[]):
        plugins = discover_and_register_plugins()

    assert plugins == []
