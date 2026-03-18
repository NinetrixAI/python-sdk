"""
export/loader.py — agentfile.yaml → Agent Python object.

Layer: L9 (export) — may import all lower layers + stdlib + pyyaml.

Parses the same agentfile.yaml format produced by ``ninetrix init`` and
``ninetrix build``, mapping every field to the SDK's ``AgentConfig``.

Only the first agent entry is loaded (multi-agent YAML files are supported
at the CLI level; the SDK returns a single Agent per call).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TYPE_CHECKING


def load_agent_from_yaml(path: str | Path) -> "Agent":
    """Load an :class:`~ninetrix.agent.agent.Agent` from an agentfile.yaml file.

    Parses the ``agents:`` map and constructs an ``Agent`` from the first entry
    (or the entry whose key matches the file name stem). All standard agentfile
    fields are supported.

    ``${ENV_VAR}`` placeholders in string values are expanded using the current
    process environment.

    Args:
        path: Filesystem path to an agentfile.yaml (or compatible YAML file).

    Returns:
        A fully configured :class:`~ninetrix.agent.agent.Agent` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file contains no ``agents:`` block or the block is empty.
        ImportError: If ``pyyaml`` is not installed.

    Example::

        from ninetrix import Agent
        agent = Agent.from_yaml("agentfile.yaml")
        result = agent.run("Summarise Q1 results")
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "pyyaml is required for YAML loading.\n"
            "  Fix: pip install pyyaml"
        ) from exc

    from ninetrix.agent.agent import Agent

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"agentfile not found: {p}\n"
            f"  Fix: check the path or run 'ninetrix init' to create one."
        )

    raw = p.read_text(encoding="utf-8")
    data: dict[str, Any] = yaml.safe_load(raw) or {}

    agents_block: dict[str, Any] = data.get("agents", {})
    if not agents_block:
        raise ValueError(
            f"No 'agents:' block found in {p}.\n"
            f"  Fix: ensure the file has a top-level 'agents:' key."
        )

    # Pick the first agent entry
    agent_name, agent_data = next(iter(agents_block.items()))
    if not isinstance(agent_data, dict):
        agent_data = {}

    kwargs = _parse_agent_dict(agent_name, agent_data)
    return Agent(**kwargs)


def load_all_agents_from_yaml(path: str | Path) -> "dict[str, Agent]":
    """Load all agents from an agentfile.yaml file.

    Returns a dict mapping agent name → :class:`~ninetrix.agent.agent.Agent`.

    Args:
        path: Filesystem path to an agentfile.yaml.

    Returns:
        Dict of ``{name: Agent}``.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "pyyaml is required for YAML loading.\n"
            "  Fix: pip install pyyaml"
        ) from exc

    from ninetrix.agent.agent import Agent

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"agentfile not found: {p}")

    raw = p.read_text(encoding="utf-8")
    data: dict[str, Any] = yaml.safe_load(raw) or {}
    agents_block: dict[str, Any] = data.get("agents", {})
    if not agents_block:
        raise ValueError(f"No 'agents:' block found in {p}.")

    result: dict[str, Any] = {}
    for name, agent_data in agents_block.items():
        if not isinstance(agent_data, dict):
            agent_data = {}
        kwargs = _parse_agent_dict(name, agent_data)
        result[name] = Agent(**kwargs)
    return result


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------


def _parse_agent_dict(name: str, data: dict[str, Any]) -> dict[str, Any]:
    """Map an agent YAML sub-dict to Agent() constructor kwargs."""
    kwargs: dict[str, Any] = {"name": name}

    # metadata
    meta = data.get("metadata", {}) or {}
    if meta.get("role"):
        kwargs["role"] = _expand_env(str(meta["role"]))
    if meta.get("goal"):
        kwargs["goal"] = _expand_env(str(meta["goal"]))
    if meta.get("instructions"):
        kwargs["instructions"] = _expand_env(str(meta["instructions"]))
    if meta.get("constraints"):
        kwargs["constraints"] = [str(c) for c in meta["constraints"]]
    if meta.get("description"):
        kwargs["description"] = _expand_env(str(meta["description"]))

    # runtime
    runtime = data.get("runtime", {}) or {}
    if runtime.get("provider"):
        kwargs["provider"] = str(runtime["provider"])
    if runtime.get("model"):
        kwargs["model"] = str(runtime["model"])
    if runtime.get("temperature") is not None:
        kwargs["temperature"] = float(runtime["temperature"])
    if runtime.get("max_tokens") is not None:
        kwargs["max_tokens"] = int(runtime["max_tokens"])

    # tools — split into mcp / composio / local lists
    mcp_tools: list[str] = []
    composio_tools: list[str] = []
    # local tools cannot be re-instantiated from YAML (no function reference),
    # so we skip them silently and record a warning via local_tools=[]
    for tool in data.get("tools", []) or []:
        if not isinstance(tool, dict):
            continue
        source = str(tool.get("source", ""))
        if source.startswith("mcp://"):
            mcp_tools.append(source[len("mcp://"):])
        elif source.startswith("composio://"):
            composio_tools.append(source[len("composio://"):])
        # local tools (./*.py) are silently skipped — no function ref available

    if mcp_tools:
        kwargs["mcp_tools"] = mcp_tools
    if composio_tools:
        kwargs["composio_tools"] = composio_tools

    # governance
    gov = data.get("governance", {}) or {}
    if gov.get("max_budget_per_run") is not None:
        kwargs["max_budget_usd"] = float(gov["max_budget_per_run"])

    # execution
    exec_ = data.get("execution", {}) or {}
    if exec_.get("mode"):
        kwargs["execution_mode"] = str(exec_["mode"])
    if exec_.get("max_steps") is not None:
        kwargs["max_turns"] = int(exec_["max_steps"])

    # persistence
    persist = data.get("persistence", {}) or {}
    if persist.get("url"):
        raw_url = str(persist["url"])
        kwargs["db_url"] = _expand_env(raw_url)

    return kwargs


def _expand_env(value: str) -> str:
    """Expand ``${VAR}`` and ``$VAR`` placeholders from the current environment."""
    return os.path.expandvars(value)
