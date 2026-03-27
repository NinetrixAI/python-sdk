"""
Example 2 — Parallel fan-out with reduce.

Demonstrates:
  - @Workflow decorator
  - Workflow.fan_out() for concurrent agent calls
  - Workflow.reduce() to synthesize multiple outputs into one
  - WorkflowResult with step info and cost

Use case: research a list of topics in parallel, then synthesise into a report.

Prerequisites:
  pip install ninetrix anthropic
  export ANTHROPIC_API_KEY=sk-...
"""

from __future__ import annotations

import asyncio

from ninetrix import Agent, Workflow, WorkflowResult


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

researcher = Agent(
    name="researcher",
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    role=(
        "You are a concise research assistant. "
        "Given a topic, write a 2-3 sentence factual summary."
    ),
)

synthesizer = Agent(
    name="synthesizer",
    provider="anthropic",
    model="claude-sonnet-4-6",
    role=(
        "You are an expert editor. "
        "Combine multiple research summaries into a single coherent report. "
        "Preserve key facts. Keep the report under 300 words."
    ),
)


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

TOPICS = [
    "quantum computing breakthroughs in 2024",
    "large language model efficiency techniques",
    "neuromorphic chip architectures",
    "photonic computing for AI inference",
    "energy consumption of AI data centres",
]


@Workflow(max_budget=0.50)   # hard cap at $0.50 for the whole run
async def research_report(topics: list[str]) -> str:
    # Fan out: each topic researched concurrently (up to 5 at a time)
    summaries = await Workflow.fan_out(
        topics,
        lambda topic: researcher.arun(f"Research topic: {topic}"),
        concurrency=5,
    )

    # Reduce: synthesise all summaries into one report
    report = await Workflow.reduce(summaries, synthesizer)
    return report.output


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    print(f"Researching {len(TOPICS)} topics in parallel...\n")
    result: WorkflowResult = await research_report.arun(TOPICS)

    print("=" * 60)
    print(result.output)
    print("=" * 60)
    print(f"\nElapsed : {result.elapsed_seconds:.1f}s")
    print(f"Budget  : ${result.budget_limit_usd:.2f} cap, ${result.budget_remaining_usd:.4f} remaining")
    print(f"Thread  : {result.thread_id}")


if __name__ == "__main__":
    asyncio.run(main())
