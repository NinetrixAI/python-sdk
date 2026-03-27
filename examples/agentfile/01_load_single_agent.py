"""
Example: Load a single agent from agentfile.yaml and run it.

Uses load_agent_from_yaml() to construct an Agent directly from a YAML
definition — the same format used by the Ninetrix CLI.

The YAML file (01_single_agent.yaml) defines a web research assistant.
The SDK maps every field to AgentConfig automatically:
  metadata.role/goal/instructions → system prompt
  runtime.provider/model          → LLM settings
  tools[].source: mcp://...       → MCPToolSource
  governance.max_budget_per_run   → budget cap

Prerequisites:
  pip install ninetrix pyyaml
  export ANTHROPIC_API_KEY=sk-ant-...
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from ninetrix import load_agent_from_yaml


YAML_PATH = Path(__file__).parent / "01_single_agent.yaml"


async def main() -> None:
    # Load the agent — identical to writing Agent(...) by hand
    agent = load_agent_from_yaml(YAML_PATH)

    print(f"Loaded agent: {agent.info().name!r}")
    print(f"Provider:     {agent.info().provider}")
    print(f"Model:        {agent.info().model}")
    print()

    question = "What are the main new features in Python 3.13?"
    print(f"Question: {question}\n")

    result = await agent.arun(question)

    print(f"Answer:\n{result.output}")
    print(f"\nTokens used: {result.tokens_used}")
    print(f"Cost:        ${result.cost_usd:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
