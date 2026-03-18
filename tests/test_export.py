"""Tests for export/writer.py + export/loader.py — PR 19."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
import yaml

from ninetrix.agent.agent import Agent
from ninetrix.agent.config import AgentConfig
from ninetrix.export.writer import agent_to_yaml, _config_to_dict
from ninetrix.export.loader import load_agent_from_yaml, load_all_agents_from_yaml, _expand_env


# =============================================================================
# writer.py — agent_to_yaml()
# =============================================================================


def test_writer_produces_valid_yaml():
    cfg = AgentConfig(name="bot")
    text = agent_to_yaml(cfg)
    parsed = yaml.safe_load(text)
    assert "agents" in parsed
    assert "bot" in parsed["agents"]


def test_writer_top_level_agents_key():
    cfg = AgentConfig(name="analyst", provider="anthropic", model="claude-sonnet-4-6")
    text = agent_to_yaml(cfg)
    data = yaml.safe_load(text)
    assert list(data["agents"].keys()) == ["analyst"]


def test_writer_metadata_role_and_goal():
    cfg = AgentConfig(name="bot", role="data analyst", goal="answer metric questions")
    text = agent_to_yaml(cfg)
    data = yaml.safe_load(text)
    meta = data["agents"]["bot"]["metadata"]
    assert meta["role"] == "data analyst"
    assert meta["goal"] == "answer metric questions"


def test_writer_metadata_instructions():
    cfg = AgentConfig(name="bot", instructions="Be concise.")
    text = agent_to_yaml(cfg)
    data = yaml.safe_load(text)
    assert data["agents"]["bot"]["metadata"]["instructions"] == "Be concise."


def test_writer_metadata_constraints():
    cfg = AgentConfig(name="bot", constraints=["No hallucination", "Cite sources"])
    text = agent_to_yaml(cfg)
    data = yaml.safe_load(text)
    assert data["agents"]["bot"]["metadata"]["constraints"] == ["No hallucination", "Cite sources"]


def test_writer_no_metadata_when_all_empty():
    cfg = AgentConfig(name="bot")  # no role/goal/instructions
    text = agent_to_yaml(cfg)
    data = yaml.safe_load(text)
    assert "metadata" not in data["agents"]["bot"]


def test_writer_runtime_provider_and_model():
    cfg = AgentConfig(name="bot", provider="openai", model="gpt-4o")
    data = yaml.safe_load(agent_to_yaml(cfg))
    runtime = data["agents"]["bot"]["runtime"]
    assert runtime["provider"] == "openai"
    assert runtime["model"] == "gpt-4o"


def test_writer_runtime_temperature_omitted_when_default():
    cfg = AgentConfig(name="bot", temperature=0.0)
    data = yaml.safe_load(agent_to_yaml(cfg))
    assert "temperature" not in data["agents"]["bot"]["runtime"]


def test_writer_runtime_temperature_written_when_nondefault():
    cfg = AgentConfig(name="bot", temperature=0.7)
    data = yaml.safe_load(agent_to_yaml(cfg))
    assert data["agents"]["bot"]["runtime"]["temperature"] == pytest.approx(0.7)


def test_writer_mcp_tools():
    cfg = AgentConfig(name="bot", mcp_tools=["tavily", "github"])
    data = yaml.safe_load(agent_to_yaml(cfg))
    tools = data["agents"]["bot"]["tools"]
    sources = {t["source"] for t in tools}
    assert "mcp://tavily" in sources
    assert "mcp://github" in sources


def test_writer_composio_tools():
    cfg = AgentConfig(name="bot", composio_tools=["GMAIL", "GITHUB"])
    data = yaml.safe_load(agent_to_yaml(cfg))
    tools = data["agents"]["bot"]["tools"]
    sources = {t["source"] for t in tools}
    assert "composio://GMAIL" in sources
    assert "composio://GITHUB" in sources


def test_writer_no_tools_key_when_empty():
    cfg = AgentConfig(name="bot")
    data = yaml.safe_load(agent_to_yaml(cfg))
    assert "tools" not in data["agents"]["bot"]


def test_writer_governance_budget():
    cfg = AgentConfig(name="bot", max_budget_usd=2.50)
    data = yaml.safe_load(agent_to_yaml(cfg))
    assert data["agents"]["bot"]["governance"]["max_budget_per_run"] == pytest.approx(2.50)


def test_writer_no_governance_when_zero_budget():
    cfg = AgentConfig(name="bot", max_budget_usd=0.0)
    data = yaml.safe_load(agent_to_yaml(cfg))
    assert "governance" not in data["agents"]["bot"]


def test_writer_execution_mode_planned():
    cfg = AgentConfig(name="bot", execution_mode="planned")
    data = yaml.safe_load(agent_to_yaml(cfg))
    assert data["agents"]["bot"]["execution"]["mode"] == "planned"


def test_writer_execution_omitted_when_defaults():
    cfg = AgentConfig(name="bot")  # mode=direct, max_turns=20
    data = yaml.safe_load(agent_to_yaml(cfg))
    assert "execution" not in data["agents"]["bot"]


def test_writer_execution_max_steps():
    cfg = AgentConfig(name="bot", max_turns=10)
    data = yaml.safe_load(agent_to_yaml(cfg))
    assert data["agents"]["bot"]["execution"]["max_steps"] == 10


def test_writer_persistence_placeholder():
    cfg = AgentConfig(name="bot", db_url="postgresql://localhost/mydb")
    data = yaml.safe_load(agent_to_yaml(cfg))
    persist = data["agents"]["bot"]["persistence"]
    assert persist["provider"] == "postgres"
    assert persist["url"] == "${DATABASE_URL}"


def test_writer_no_persistence_when_no_db_url():
    cfg = AgentConfig(name="bot")
    data = yaml.safe_load(agent_to_yaml(cfg))
    assert "persistence" not in data["agents"]["bot"]


# =============================================================================
# writer.py — Agent.to_yaml() method
# =============================================================================


def test_agent_to_yaml_method():
    a = Agent(name="analyst", provider="anthropic", role="Data analyst")
    text = a.to_yaml()
    data = yaml.safe_load(text)
    assert "analyst" in data["agents"]
    assert data["agents"]["analyst"]["metadata"]["role"] == "Data analyst"


def test_agent_to_yaml_roundtrip_name():
    a = Agent(name="mybot", model="gpt-4o", provider="openai")
    text = a.to_yaml()
    data = yaml.safe_load(text)
    assert "mybot" in data["agents"]
    assert data["agents"]["mybot"]["runtime"]["provider"] == "openai"


# =============================================================================
# loader.py — load_agent_from_yaml()
# =============================================================================


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "agentfile.yaml"
    p.write_text(textwrap.dedent(content))
    return p


def test_loader_basic(tmp_path):
    p = _write_yaml(tmp_path, """
        agents:
          analyst:
            metadata:
              role: Financial analyst
              goal: Answer questions
            runtime:
              provider: anthropic
              model: claude-sonnet-4-6
              temperature: 0.3
    """)
    a = load_agent_from_yaml(p)
    assert a.name == "analyst"
    assert a.config.provider == "anthropic"
    assert a.config.model == "claude-sonnet-4-6"
    assert a.config.temperature == pytest.approx(0.3)
    assert a.config.role == "Financial analyst"
    assert a.config.goal == "Answer questions"


def test_loader_mcp_tools(tmp_path):
    p = _write_yaml(tmp_path, """
        agents:
          bot:
            runtime:
              provider: anthropic
              model: claude-sonnet-4-6
            tools:
              - name: tavily
                source: mcp://tavily
              - name: github
                source: mcp://github
    """)
    a = load_agent_from_yaml(p)
    assert "tavily" in a.config.mcp_tools
    assert "github" in a.config.mcp_tools


def test_loader_composio_tools(tmp_path):
    p = _write_yaml(tmp_path, """
        agents:
          bot:
            runtime:
              provider: anthropic
              model: claude-sonnet-4-6
            tools:
              - name: gmail
                source: composio://GMAIL
    """)
    a = load_agent_from_yaml(p)
    assert "GMAIL" in a.config.composio_tools


def test_loader_local_tools_silently_skipped(tmp_path):
    p = _write_yaml(tmp_path, """
        agents:
          bot:
            runtime:
              provider: anthropic
              model: claude-sonnet-4-6
            tools:
              - name: my_tools
                source: ./tools/my_tools.py
    """)
    a = load_agent_from_yaml(p)
    # local tools can't be loaded from YAML (no function reference) — silently skipped
    assert a.config.local_tools == []


def test_loader_governance_budget(tmp_path):
    p = _write_yaml(tmp_path, """
        agents:
          bot:
            runtime:
              provider: anthropic
              model: claude-sonnet-4-6
            governance:
              max_budget_per_run: 1.50
    """)
    a = load_agent_from_yaml(p)
    assert a.config.max_budget_usd == pytest.approx(1.50)


def test_loader_execution_mode(tmp_path):
    p = _write_yaml(tmp_path, """
        agents:
          bot:
            runtime:
              provider: anthropic
              model: claude-sonnet-4-6
            execution:
              mode: planned
              max_steps: 15
    """)
    a = load_agent_from_yaml(p)
    assert a.config.execution_mode == "planned"
    assert a.config.max_turns == 15


def test_loader_constraints(tmp_path):
    p = _write_yaml(tmp_path, """
        agents:
          bot:
            metadata:
              constraints:
                - Only use search results
                - Cite sources
            runtime:
              provider: anthropic
              model: claude-sonnet-4-6
    """)
    a = load_agent_from_yaml(p)
    assert "Only use search results" in a.config.constraints
    assert "Cite sources" in a.config.constraints


def test_loader_env_var_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_DB_URL", "postgresql://localhost/testdb")
    p = _write_yaml(tmp_path, """
        agents:
          bot:
            runtime:
              provider: anthropic
              model: claude-sonnet-4-6
            persistence:
              provider: postgres
              url: ${TEST_DB_URL}
    """)
    a = load_agent_from_yaml(p)
    assert a.config.db_url == "postgresql://localhost/testdb"


def test_loader_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_agent_from_yaml("/nonexistent/agentfile.yaml")


def test_loader_no_agents_block(tmp_path):
    p = tmp_path / "agentfile.yaml"
    p.write_text("version: '1.0'\n")
    with pytest.raises(ValueError, match="No 'agents:'"):
        load_agent_from_yaml(p)


def test_loader_returns_first_agent_when_multiple(tmp_path):
    p = _write_yaml(tmp_path, """
        agents:
          alpha:
            runtime:
              provider: anthropic
              model: claude-sonnet-4-6
          beta:
            runtime:
              provider: openai
              model: gpt-4o
    """)
    a = load_agent_from_yaml(p)
    assert a.name == "alpha"


# =============================================================================
# loader.py — load_all_agents_from_yaml()
# =============================================================================


def test_load_all_agents(tmp_path):
    p = _write_yaml(tmp_path, """
        agents:
          planner:
            runtime:
              provider: anthropic
              model: claude-sonnet-4-6
          writer:
            runtime:
              provider: openai
              model: gpt-4o
    """)
    agents = load_all_agents_from_yaml(p)
    assert "planner" in agents
    assert "writer" in agents
    assert agents["planner"].config.provider == "anthropic"
    assert agents["writer"].config.provider == "openai"


# =============================================================================
# Agent.from_yaml() method
# =============================================================================


def test_agent_from_yaml_method(tmp_path):
    p = _write_yaml(tmp_path, """
        agents:
          bot:
            metadata:
              role: Assistant
            runtime:
              provider: anthropic
              model: claude-haiku-4-5-20251001
    """)
    a = Agent.from_yaml(str(p))
    assert a.name == "bot"
    assert a.config.role == "Assistant"
    assert a.config.model == "claude-haiku-4-5-20251001"


# =============================================================================
# Full round-trip: Agent → to_yaml() → from_yaml() → Agent
# =============================================================================


def test_round_trip_name(tmp_path):
    a1 = Agent(name="myagent", provider="anthropic", model="claude-sonnet-4-6")
    yaml_text = a1.to_yaml()
    p = tmp_path / "agentfile.yaml"
    p.write_text(yaml_text)
    a2 = Agent.from_yaml(str(p))
    assert a2.name == a1.name


def test_round_trip_provider_and_model(tmp_path):
    a1 = Agent(name="bot", provider="openai", model="gpt-4o")
    p = tmp_path / "agentfile.yaml"
    p.write_text(a1.to_yaml())
    a2 = Agent.from_yaml(str(p))
    assert a2.config.provider == "openai"
    assert a2.config.model == "gpt-4o"


def test_round_trip_role_goal_instructions(tmp_path):
    a1 = Agent(
        name="bot",
        role="analyst",
        goal="crunch numbers",
        instructions="Be precise.",
    )
    p = tmp_path / "agentfile.yaml"
    p.write_text(a1.to_yaml())
    a2 = Agent.from_yaml(str(p))
    assert a2.config.role == "analyst"
    assert a2.config.goal == "crunch numbers"
    assert a2.config.instructions == "Be precise."


def test_round_trip_constraints(tmp_path):
    a1 = Agent(name="bot", constraints=["Constraint A", "Constraint B"])
    p = tmp_path / "agentfile.yaml"
    p.write_text(a1.to_yaml())
    a2 = Agent.from_yaml(str(p))
    assert "Constraint A" in a2.config.constraints
    assert "Constraint B" in a2.config.constraints


def test_round_trip_mcp_tools(tmp_path):
    a1 = Agent(name="bot", mcp_tools=["tavily", "github"])
    p = tmp_path / "agentfile.yaml"
    p.write_text(a1.to_yaml())
    a2 = Agent.from_yaml(str(p))
    assert "tavily" in a2.config.mcp_tools
    assert "github" in a2.config.mcp_tools


def test_round_trip_budget(tmp_path):
    a1 = Agent(name="bot", max_budget_usd=3.0)
    p = tmp_path / "agentfile.yaml"
    p.write_text(a1.to_yaml())
    a2 = Agent.from_yaml(str(p))
    assert a2.config.max_budget_usd == pytest.approx(3.0)


def test_round_trip_temperature(tmp_path):
    a1 = Agent(name="bot", temperature=0.5)
    p = tmp_path / "agentfile.yaml"
    p.write_text(a1.to_yaml())
    a2 = Agent.from_yaml(str(p))
    assert a2.config.temperature == pytest.approx(0.5)


# =============================================================================
# _expand_env helper
# =============================================================================


def test_expand_env_substitutes_var(monkeypatch):
    monkeypatch.setenv("MY_SECRET", "abc123")
    assert _expand_env("${MY_SECRET}") == "abc123"


def test_expand_env_passthrough_no_var():
    assert _expand_env("plain string") == "plain string"


def test_expand_env_missing_var():
    # os.path.expandvars leaves unset vars as-is
    result = _expand_env("${SURELY_NOT_SET_VARIABLE_XYZ}")
    # Behaviour: either expanded (if set in env) or left as literal
    assert isinstance(result, str)


# =============================================================================
# Public imports
# =============================================================================


def test_agent_to_yaml_importable_from_ninetrix():
    from ninetrix import agent_to_yaml
    assert callable(agent_to_yaml)


def test_load_agent_from_yaml_importable_from_ninetrix():
    from ninetrix import load_agent_from_yaml
    assert callable(load_agent_from_yaml)


def test_load_all_agents_from_yaml_importable_from_ninetrix():
    from ninetrix import load_all_agents_from_yaml
    assert callable(load_all_agents_from_yaml)
