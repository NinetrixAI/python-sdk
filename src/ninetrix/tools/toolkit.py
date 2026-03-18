"""
tools/toolkit.py — Toolkit: a named, reusable collection of @Tool functions.

Layer: L2 (tools) — may import L1 (_internals) + stdlib only.

A Toolkit groups related @Tool functions under a single name and description.
Pass a Toolkit to ``Agent(tools=[my_toolkit])`` just like individual @Tool
functions — the Agent unwraps it automatically.

Usage::

    from ninetrix.tools.toolkit import Toolkit

    db = Toolkit("database", description="SQL query helpers")

    @db.tool
    def query(sql: str) -> list[dict]:
        \"\"\"Execute a read-only SELECT.\"\"\"
        return run_sql(sql)

    @db.tool
    def schema(table: str) -> str:
        \"\"\"Return the schema of *table*.\"\"\"
        return describe_table(table)

    # Or build from existing @Tool-decorated functions:
    from ninetrix import Toolkit, Tool

    @Tool
    def search(q: str) -> str:
        \"\"\"Search the web.\"\"\"
        ...

    kit = Toolkit("search_tools", [search], description="Web search utilities")
"""

from __future__ import annotations

from typing import Any, Callable

from ninetrix.registry import ToolDef, ToolRegistry, _registry


class Toolkit:
    """A named, reusable collection of :class:`~ninetrix.ToolDef` entries.

    Toolkits group related tools under one label.  Passing a ``Toolkit`` to
    ``Agent(tools=[...])`` is equivalent to passing each of its tools
    individually — the Agent unwraps the Toolkit automatically.

    Args:
        name:        Identifier for this toolkit (used in YAML export).
        tools:       Optional initial list of ``@Tool``-decorated callables or
                     :class:`~ninetrix.ToolDef` instances.  Tools can also be
                     added later via the :meth:`tool` decorator.
        description: Human-readable description of what this toolkit provides.

    Example::

        from ninetrix import Toolkit

        analytics = Toolkit("analytics", description="Data analytics tools")

        @analytics.tool
        def top_customers(n: int = 10) -> list[dict]:
            \"\"\"Return the top N customers by revenue.\"\"\"
            ...

        @analytics.tool
        def monthly_revenue(month: str) -> float:
            \"\"\"Return revenue for the given month (YYYY-MM).\"\"\"
            ...

        agent = Agent(provider="anthropic", tools=[analytics])
    """

    def __init__(
        self,
        name: str,
        tools: list[Any] | None = None,
        *,
        description: str = "",
    ) -> None:
        self.name = name
        self.description = description
        self._tools: list[ToolDef] = []

        if tools:
            for t in tools:
                self._add(t)

    # ------------------------------------------------------------------
    # Decorator API
    # ------------------------------------------------------------------

    def tool(self, fn: Callable) -> Callable:
        """Decorator: register *fn* as a tool in this toolkit.

        ``@toolkit.tool`` also applies the global ``@Tool`` decorator so that
        the function is registered in the module-level registry and its schema
        is auto-generated from type annotations and docstrings.

        Returns:
            The original function, unchanged (same as ``@Tool``).
        """
        from ninetrix.tool import Tool
        wrapped = Tool(fn)
        td = _registry.get(fn.__name__)
        if td is not None and td not in self._tools:
            self._tools.append(td)
        return wrapped

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def tools(self) -> list[ToolDef]:
        """Return a copy of the registered :class:`~ninetrix.ToolDef` list."""
        return list(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"<Toolkit name={self.name!r} tools={len(self._tools)}>"

    # ------------------------------------------------------------------
    # YAML export
    # ------------------------------------------------------------------

    def to_yaml(self) -> str:
        """Return a YAML tools list fragment for this toolkit's tools.

        The fragment follows the ``agentfile.yaml`` schema::

            tools:
              - name: query
                description: Execute a read-only SELECT.
              - name: schema
                description: Return the schema of *table*.

        Returns:
            A YAML string starting with ``tools:``
        """
        if not self._tools:
            return "tools: []\n"

        lines = ["tools:"]
        for td in self._tools:
            lines.append(f"  - name: {td.name}")
            if td.description:
                # Inline — no multi-line needed for short descriptions
                safe_desc = td.description.replace("\n", " ").strip()
                lines.append(f"    description: {safe_desc!r}")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add(self, item: Any) -> None:
        """Add a ``ToolDef`` or an ``@Tool``-decorated callable."""
        if isinstance(item, ToolDef):
            if item not in self._tools:
                self._tools.append(item)
            return

        # @Tool-decorated callable — look up in global registry
        fn_name = getattr(item, "__name__", None) or getattr(item, "name", None)
        if fn_name:
            td = _registry.get(fn_name)
            if td is not None and td not in self._tools:
                self._tools.append(td)
                return

        # Not found — silently skip (may be pre-registration; caller can add later)
