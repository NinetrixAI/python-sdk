"""
export/writer.py — AgentConfig → agentfile.yaml string.

Layer: L9 (export) — may import all lower layers + stdlib + pyyaml.

The output is valid agentfile.yaml that can be:
  - Round-tripped through Agent.from_yaml()
  - Passed directly to ``ninetrix build`` (CLI)

Only non-default / non-empty values are written, keeping the YAML clean.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ninetrix.agent.config import AgentConfig


def agent_to_yaml(config: "AgentConfig") -> str:
    """Convert an :class:`~ninetrix.agent.config.AgentConfig` to an agentfile.yaml string.

    The returned string is valid agentfile.yaml. It can be round-tripped
    through :func:`~ninetrix.export.loader.load_agent_from_yaml` or passed
    directly to ``ninetrix build``.

    Only fields that differ from their defaults are included. The agent
    name becomes the key under the top-level ``agents:`` map.

    Args:
        config: The agent configuration to serialise.

    Returns:
        A YAML string with a single entry under ``agents:``.

    Example::

        from ninetrix import Agent
        agent = Agent(name="analyst", provider="anthropic", role="Data analyst")
        print(agent.to_yaml())
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "pyyaml is required for YAML export.\n"
            "  Fix: pip install pyyaml"
        ) from exc

    data: dict[str, Any] = {"agents": {config.name: _config_to_dict(config)}}
    return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _config_to_dict(config: "AgentConfig") -> dict[str, Any]:
    """Build the agent sub-dict for the YAML output."""
    agent: dict[str, Any] = {}

    # ── metadata ──────────────────────────────────────────────────────────
    meta: dict[str, Any] = {}
    if config.role:
        meta["role"] = config.role
    if config.goal:
        meta["goal"] = config.goal
    if config.instructions:
        meta["instructions"] = config.instructions
    if config.constraints:
        meta["constraints"] = list(config.constraints)
    if config.description:
        meta["description"] = config.description
    if meta:
        agent["metadata"] = meta

    # ── runtime ───────────────────────────────────────────────────────────
    runtime: dict[str, Any] = {
        "provider": config.provider,
        "model": config.model,
    }
    if config.temperature != 0.0:
        runtime["temperature"] = config.temperature
    if config.max_tokens != 8192:
        runtime["max_tokens"] = config.max_tokens
    agent["runtime"] = runtime

    # ── tools ─────────────────────────────────────────────────────────────
    tools: list[dict[str, str]] = []

    # Local tools — group by source_file if available, fall back to name
    if config.local_tools:
        seen_sources: set[str] = set()
        for td in config.local_tools:
            source_file: str = getattr(td, "source_file", "") or ""
            name: str = getattr(td, "name", "") or getattr(td, "__name__", "local")
            if source_file and source_file not in seen_sources:
                seen_sources.add(source_file)
                source = source_file if source_file.startswith(".") else f"./{source_file}"
                tools.append({"name": name, "source": source})
            elif not source_file:
                tools.append({"name": name, "source": f"./{name}.py"})

    for tool_name in config.mcp_tools:
        tools.append({"name": tool_name, "source": f"mcp://{tool_name}"})

    for app_name in config.composio_tools:
        tools.append({"name": app_name.lower(), "source": f"composio://{app_name}"})

    if tools:
        agent["tools"] = tools

    # ── governance ────────────────────────────────────────────────────────
    if config.max_budget_usd > 0:
        agent["governance"] = {"max_budget_per_run": config.max_budget_usd}

    # ── execution ─────────────────────────────────────────────────────────
    if config.execution_mode != "direct" or config.max_turns != 20:
        exec_dict: dict[str, Any] = {}
        if config.execution_mode != "direct":
            exec_dict["mode"] = config.execution_mode
        if config.max_turns != 20:
            exec_dict["max_steps"] = config.max_turns
        if config.on_step_failure != "continue":
            exec_dict["on_step_failure"] = config.on_step_failure
        if exec_dict:
            agent["execution"] = exec_dict

    # ── persistence ───────────────────────────────────────────────────────
    if config.db_url:
        agent["persistence"] = {
            "provider": "postgres",
            "url": "${DATABASE_URL}",
        }

    return agent
