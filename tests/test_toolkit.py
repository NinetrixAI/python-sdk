"""Tests for tools/toolkit.py — PR 27."""

from __future__ import annotations

import pytest

from ninetrix import Tool, Toolkit
from ninetrix.registry import ToolDef, ToolRegistry, _registry


# ---------------------------------------------------------------------------
# Helpers — isolated registry per test where needed
# ---------------------------------------------------------------------------


def _make_tool_def(name: str = "my_tool", description: str = "Does stuff") -> ToolDef:
    return ToolDef(
        name=name,
        description=description,
        parameters={"type": "object", "properties": {}, "required": []},
        fn=lambda: None,
    )


# =============================================================================
# Construction
# =============================================================================


def test_toolkit_constructs():
    tk = Toolkit("my_tools")
    assert tk.name == "my_tools"
    assert tk.description == ""
    assert len(tk) == 0


def test_toolkit_description():
    tk = Toolkit("db", description="Database tools")
    assert tk.description == "Database tools"


def test_toolkit_from_tool_defs():
    td1 = _make_tool_def("a")
    td2 = _make_tool_def("b")
    tk = Toolkit("set", [td1, td2])
    assert len(tk) == 2


def test_toolkit_repr():
    tk = Toolkit("t")
    assert "t" in repr(tk)
    assert "Toolkit" in repr(tk)


# =============================================================================
# @toolkit.tool decorator
# =============================================================================


def test_toolkit_tool_decorator_registers():
    tk = Toolkit("test_decorator_reg")

    @tk.tool
    def greet(name: str) -> str:
        """Say hello to someone."""
        return f"Hello, {name}"

    assert len(tk) == 1
    assert tk.tools()[0].name == "greet"


def test_toolkit_tool_decorator_returns_fn():
    tk = Toolkit("test_decorator_return")

    @tk.tool
    def compute(x: int) -> int:
        """Compute x."""
        return x * 2

    # Should return the original callable (same as @Tool)
    assert callable(compute)
    assert compute(3) == 6


def test_toolkit_tool_descriptor_in_description():
    tk = Toolkit("test_desc")

    @tk.tool
    def describe_world() -> str:
        """Describe the world."""
        return "round"

    td = tk.tools()[0]
    assert "Describe the world" in td.description


def test_toolkit_tool_no_duplicates():
    """Adding the same function twice should not duplicate the entry."""
    tk = Toolkit("test_dedup")

    @tk.tool
    def unique_fn_xyz(x: str) -> str:
        """Unique fn."""
        return x

    # Calling _add again should not add a duplicate
    td = tk.tools()[0]
    tk._add(td)
    assert len(tk) == 1


# =============================================================================
# Constructing from @Tool-decorated callables
# =============================================================================


def test_toolkit_from_decorated_callables():
    @Tool
    def registered_tool_abc(x: int) -> str:
        """A registered tool."""
        return str(x)

    tk = Toolkit("from_callable", [registered_tool_abc])
    assert len(tk) == 1
    assert tk.tools()[0].name == "registered_tool_abc"


def test_toolkit_ignores_unknown_item():
    """Items not found in registry should be silently ignored."""
    class Unknown:
        __name__ = "nonexistent_xyzzy_tool"

    tk = Toolkit("safe", [Unknown()])
    assert len(tk) == 0


# =============================================================================
# .tools() API
# =============================================================================


def test_toolkit_tools_returns_copy():
    td = _make_tool_def("copy_test")
    tk = Toolkit("copy", [td])
    lst1 = tk.tools()
    lst2 = tk.tools()
    assert lst1 == lst2
    lst1.clear()
    assert len(tk) == 1  # original unaffected


# =============================================================================
# to_yaml()
# =============================================================================


def test_toolkit_to_yaml_empty():
    tk = Toolkit("empty")
    assert tk.to_yaml() == "tools: []\n"


def test_toolkit_to_yaml_with_tools():
    td = _make_tool_def("run_query", description="Execute SQL")
    tk = Toolkit("db", [td])
    yaml_str = tk.to_yaml()
    assert "tools:" in yaml_str
    assert "run_query" in yaml_str
    assert "Execute SQL" in yaml_str


def test_toolkit_to_yaml_multiple_tools():
    tk = Toolkit("multi", [
        _make_tool_def("a", "First tool"),
        _make_tool_def("b", "Second tool"),
    ])
    yaml_str = tk.to_yaml()
    assert "- name: a" in yaml_str
    assert "- name: b" in yaml_str


def test_toolkit_to_yaml_ends_with_newline():
    tk = Toolkit("t", [_make_tool_def("x")])
    assert tk.to_yaml().endswith("\n")


# =============================================================================
# Agent(tools=[toolkit]) integration
# =============================================================================


def test_agent_accepts_toolkit():
    """Agent should not raise when passed a Toolkit in tools=."""
    from ninetrix import Agent

    tk = Toolkit("agent_compat_kit")

    @tk.tool
    def ping_tool_xyz(msg: str) -> str:
        """Ping pong."""
        return msg

    # Construction should succeed without errors
    agent = Agent(provider="anthropic", tools=[tk])
    assert agent is not None


def test_agent_toolkit_visible_in_info():
    """Tools from a Toolkit should appear in agent.info()."""
    from ninetrix import Agent

    tk = Toolkit("info_test_kit")

    @tk.tool
    def query_data_unique(sql: str) -> str:
        """Run SQL query."""
        return sql

    agent = Agent(provider="anthropic", tools=[tk])
    info = agent.info()
    tool_names = [t for t in info.local_tools]
    assert "query_data_unique" in tool_names


def test_agent_toolkit_and_individual_tools():
    """Mix of Toolkit and individual @Tool functions should both register."""
    from ninetrix import Agent

    @Tool
    def individual_fn_unique(x: str) -> str:
        """Individual tool."""
        return x

    tk = Toolkit("mixed_kit_test")

    @tk.tool
    def kit_fn_unique(x: str) -> str:
        """Kit tool."""
        return x

    agent = Agent(provider="anthropic", tools=[individual_fn_unique, tk])
    info = agent.info()
    assert "individual_fn_unique" in info.local_tools
    assert "kit_fn_unique" in info.local_tools


# =============================================================================
# Public imports
# =============================================================================


def test_toolkit_importable_from_ninetrix():
    from ninetrix import Toolkit
    assert Toolkit is not None


def test_toolkit_importable_from_tools():
    from ninetrix.tools.toolkit import Toolkit
    assert Toolkit is not None
