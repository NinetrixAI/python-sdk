"""
Example: Load multiple agents from agentfile.yaml and run a pipeline.

Uses load_all_agents_from_yaml() to construct every agent defined in the
YAML file as an Agent instance. The agents are then wired together in a
simple researcher → writer pipeline.

The YAML file (02_two_agents.yaml) defines:
  orchestrator  — coordinates the pipeline (unused here; we call agents directly)
  researcher    — searches the web, returns structured findings
  writer        — turns findings into a polished article

Calling agents directly from Python (instead of via the orchestrator)
shows the most explicit pattern: you control the pipeline in code.

Prerequisites:
  pip install ninetrix pyyaml
  export ANTHROPIC_API_KEY=sk-ant-...
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from ninetrix import load_all_agents_from_yaml


YAML_PATH = Path(__file__).parent / "02_two_agents.yaml"

TOPIC = "The current state of WebAssembly in server-side applications (2025)"


async def main() -> None:
    # Load every agent defined in the file
    agents = load_all_agents_from_yaml(YAML_PATH)

    print("Loaded agents:", list(agents.keys()))
    print()

    researcher = agents["researcher"]
    writer = agents["writer"]

    # ── Step 1: Research ─────────────────────────────────────────────────────
    print(f"[researcher] Researching: {TOPIC!r}")
    research = await researcher.arun(
        f"Research the following topic thoroughly and return structured findings:\n\n{TOPIC}"
    )
    print(f"[researcher] Done — {research.tokens_used} tokens\n")

    # ── Step 2: Write ────────────────────────────────────────────────────────
    print("[writer] Writing article...")
    article = await writer.arun(
        f"Turn these research findings into a polished 500-word technical blog post:\n\n"
        f"{research.output}"
    )
    print(f"[writer] Done — {article.tokens_used} tokens\n")

    # ── Result ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print(article.output)
    print("=" * 60)
    print(f"\nTotal tokens: {research.tokens_used + article.tokens_used}")
    print(f"Total cost:   ${research.cost_usd + article.cost_usd:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
