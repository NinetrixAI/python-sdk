"""
Example 3 — Durable workflow with resume.

Demonstrates:
  - durable=True workflows that survive interruptions
  - Workflow.step() context manager
  - _StepResult.is_cached and .value for idempotent steps
  - InMemoryCheckpointer for local dev (swap for PostgresCheckpointer in prod)
  - Resuming a workflow by passing the same thread_id

Use case: a multi-step content pipeline where each stage is expensive.
          If the process crashes, re-running with the same thread_id
          skips already-completed steps.

Prerequisites:
  pip install ninetrix anthropic
  export ANTHROPIC_API_KEY=sk-...
"""

from __future__ import annotations

import asyncio

from ninetrix import Agent, Workflow, InMemoryCheckpointer, WorkflowResult


# ---------------------------------------------------------------------------
# Agents (one per pipeline stage)
# ---------------------------------------------------------------------------

researcher = Agent(
    name="researcher",
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    role="Research the given topic and return 5 key bullet points.",
)

writer = Agent(
    name="writer",
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    role="Turn bullet-point research into a well-structured 200-word blog introduction.",
)

editor = Agent(
    name="editor",
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    role="Polish the given draft for clarity and engagement. Return only the final text.",
)


# ---------------------------------------------------------------------------
# Durable workflow
# ---------------------------------------------------------------------------

@Workflow(durable=True)
async def content_pipeline(topic: str) -> str:
    # Step 1 — research
    async with Workflow.step("research") as step:
        if not step.is_cached:
            result = await researcher.arun(f"Research: {topic}")
            step.set(result.output)
        research = step.value
        print(f"  [research] {'(cached)' if step.is_cached else '(fresh)'}")

    # Step 2 — write draft
    async with Workflow.step("draft") as step:
        if not step.is_cached:
            result = await writer.arun(f"Write a blog intro based on:\n{research}")
            step.set(result.output)
        draft = step.value
        print(f"  [draft]    {'(cached)' if step.is_cached else '(fresh)'}")

    # Step 3 — edit
    async with Workflow.step("edit") as step:
        if not step.is_cached:
            result = await editor.arun(f"Edit this draft:\n{draft}")
            step.set(result.output)
        final = step.value
        print(f"  [edit]     {'(cached)' if step.is_cached else '(fresh)'}")

    return final


# ---------------------------------------------------------------------------
# Entry point — run twice with the same thread_id to show resume
# ---------------------------------------------------------------------------

async def main() -> None:
    # Use InMemoryCheckpointer for this demo.
    # In production, use PostgresCheckpointer:
    #   from ninetrix import PostgresCheckpointer
    #   cp = PostgresCheckpointer("postgresql://user:pass@localhost/mydb")
    checkpointer = InMemoryCheckpointer()
    content_pipeline.inject_checkpointer(checkpointer)

    topic = "The future of renewable energy storage"
    thread = "content-run-001"   # stable ID — change to resume a different run

    print(f"=== First run (thread: {thread}) ===")
    r1: WorkflowResult = await content_pipeline.arun(topic, thread_id=thread)
    print(f"\nCompleted: {r1.completed_steps}")
    print(f"Skipped  : {r1.skipped_steps}")
    print(f"\n{r1.output}\n")

    print(f"=== Second run — same thread_id (all steps cached) ===")
    r2: WorkflowResult = await content_pipeline.arun(topic, thread_id=thread)
    print(f"\nCompleted: {r2.completed_steps}")
    print(f"Skipped  : {r2.skipped_steps}")   # all three steps skipped


if __name__ == "__main__":
    asyncio.run(main())
