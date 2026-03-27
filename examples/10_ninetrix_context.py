"""
Example 10 — v2 API: Ninetrix context, run_step, terminate, and map.

Four real-life mini-examples in one file:

  A. Customer support router     — Ninetrix context + Team, no boilerplate
  B. Content moderation pipeline — guard step + Workflow.terminate
  C. Research fan-out            — Workflow.map with concurrency
  D. Self-correcting code writer — dynamic loop with per-attempt step caching

Prerequisites:
  pip install ninetrix anthropic
  export ANTHROPIC_API_KEY=sk-...
"""

from __future__ import annotations

import asyncio
from ninetrix import Ninetrix, Workflow, InMemoryCheckpointer, enable_debug

enable_debug()
# One context for the whole file — sets provider + model once.
nx = Ninetrix(
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    checkpointer=InMemoryCheckpointer(),
)


# ─────────────────────────────────────────────────────────────────────────────
# A. Customer support router
#    Minimal agent definitions — description auto-derives the system prompt
#    and is used by the Team router.  No router_provider / router_model needed.
# ─────────────────────────────────────────────────────────────────────────────

support_team = nx.team(
    "customer-support",
    agents=[
        nx.agent("billing",      description="Handles invoices, charges, and refunds"),
        nx.agent("tech-support", description="Debugs API errors, outages, and integrations"),
        nx.agent("general",      description="Answers product and onboarding questions"),
    ],
)

async def demo_support_router() -> None:
    print("=" * 60)
    print("A. Customer support router")
    print("=" * 60)

    questions = [
        "I was charged twice for my Pro plan this month.",
        "The REST API keeps returning a 429 even under my quota.",
        "How do I invite a teammate to my workspace?",
    ]

    for q in questions:
        result = await support_team.arun(q)
        print(f"Q: {q[:60]}")
        print(f"→ [{result.agent_name}] {result.output[:80]}...\n")


# ─────────────────────────────────────────────────────────────────────────────
# B. Content moderation pipeline
#    Guard step halts the workflow before publishing if content fails review.
#    result.terminated tells the caller whether to show an error or proceed.
# ─────────────────────────────────────────────────────────────────────────────

moderator = nx.agent(
    "moderator",
    role=(
        "You are a content safety reviewer. "
        "Check if the article contains harmful, misleading, or off-topic content. "
        "Reply with exactly PASS or FAIL:<reason>."
    ),
)

publisher = nx.agent(
    "publisher",
    role=(
        "You are a technical editor. "
        "Format the article for publication: add a headline, summary, and tags. "
        "Keep the body unchanged."
    ),
)

@Workflow(durable=True)
async def publication_pipeline(article: str) -> str:
    # Guard: moderate before any further processing
    verdict = await Workflow.run_step(
        "moderate",
        lambda: moderator.arun(article)
    )

    if not verdict.output.strip().upper().startswith("PASS"):
        # Explicit abort — caller checks result.terminated
        return Workflow.terminate(f"Blocked by moderation: {verdict.output.strip()}")

    # Only reaches here if moderation passed
    published = await Workflow.run_step(
        "publish",
        lambda: publisher.arun(article)
    )
    return published.output


async def demo_moderation() -> None:
    print("=" * 60)
    print("B. Content moderation pipeline")
    print("=" * 60)

    articles = [
        "Python 3.13 ships with a new JIT compiler that speeds up numeric code by 20%.",
        "BUY THIS STOCK NOW — guaranteed 500% returns in 30 days!!!",
    ]

    for i, article in enumerate(articles):
        result = await publication_pipeline.arun(article, thread_id=f"pub-{i}")
        if result.terminated:
            print(f"✗ BLOCKED: {result.termination_reason}")
        else:
            print(f"✓ PUBLISHED:\n{result.output[:200]}...")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# C. Research fan-out
#    Run one agent over a list of topics in parallel.
#    Workflow.map handles concurrency and order automatically.
# ─────────────────────────────────────────────────────────────────────────────

researcher = nx.agent(
    "researcher",
    role=(
        "You are a concise tech analyst. "
        "Write a 2-sentence summary of the given topic. "
        "Be factual and neutral."
    ),
)

synthesizer = nx.agent(
    "synthesizer",
    role=(
        "You are an editor. "
        "Combine multiple short summaries into one cohesive 150-word briefing. "
        "Use clear section breaks."
    ),
)

@nx.workflow(durable=True)
async def market_briefing(topics: list[str]) -> str:
    # Fan-out: researcher runs on each topic, max 3 simultaneous calls
    results = await Workflow.map(
        researcher,
        topics,
        prefix="Write a 2-sentence summary of: ",
        concurrency=3,
        step_prefix="research",   # each topic gets its own cache key
    )

    # Reduce: synthesizer merges all summaries into one briefing
    combined = "\n\n".join(f"• {r.output}" for r in results)
    briefing = await Workflow.run_step(
        "synthesize",
        lambda: synthesizer.arun(f"Combine these summaries:\n\n{combined}")
    )
    return briefing.output


async def demo_fan_out() -> None:
    print("=" * 60)
    print("C. Research fan-out")
    print("=" * 60)

    topics = [
        "OpenAI's GPT-5 release",
        "Anthropic Claude 4 capabilities",
        "Google Gemini Ultra performance benchmarks",
        "Meta Llama 3 open-source model",
    ]

    result = await market_briefing.arun(topics, thread_id="briefing-001")
    print(result.output)
    print(f"\nSteps: completed={result.completed_steps}")

    # Re-run same thread — all research steps hit cache, only synthesize re-runs
    print("\n[Re-running same thread — cache hit expected for all research steps]")
    result2 = await market_briefing.arun(topics, thread_id="briefing-001")
    print(f"Skipped (cached): {result2.skipped_steps}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# D. Self-correcting code writer
#    The agent writes code, a tester checks it, and a fixer revises on failure.
#    Dynamic step names (f"test_{attempt}") give each iteration its own cache key
#    so a crash at attempt 2 resumes from exactly that iteration.
# ─────────────────────────────────────────────────────────────────────────────

coder = nx.agent(
    "coder",
    role=(
        "You are a Python expert. "
        "Write clean, correct Python code for the given task. "
        "Output ONLY the code, no explanations."
    ),
)

code_reviewer = nx.agent(
    "code-reviewer",
    role=(
        "You are a Python code reviewer. "
        "Check the code for correctness and style. "
        "Reply with PASS if the code is correct, or FAIL:<reason> if not."
    ),
)

@nx.workflow(durable=True)
async def coding_assistant(task: str) -> str:
    # Initial implementation
    code = await Workflow.run_step(
        "write_code",
        lambda: coder.arun(f"Write Python code for: {task}")
    )

    # Review-and-fix loop — max 3 attempts
    for attempt in range(3):
        review = await Workflow.run_step(
            f"review_{attempt}",
            lambda: code_reviewer.arun(f"Review this code:\n{code.output}")
        )

        verdict = review.output.strip().upper()
        if verdict.startswith("PASS"):
            return code.output

        # Revision requested — fix and re-review
        code = await Workflow.run_step(
            f"fix_{attempt}",
            lambda: coder.arun(
                f"Fix this code:\n{code.output}\n\nReviewer note: {review.output}"
            )
        )

    # Exhausted retries — return best attempt with a warning
    return Workflow.terminate(
        f"Could not produce reviewable code after 3 attempts for: {task}"
    )


async def demo_self_correcting() -> None:
    print("=" * 60)
    print("D. Self-correcting code writer")
    print("=" * 60)

    result = await coding_assistant.arun(
        "binary search on a sorted list",
        thread_id="code-bs-001"
    )

    if result.terminated:
        print(f"✗ Gave up: {result.termination_reason}")
    else:
        print("✓ Final code:")
        print(result.output)
        print(f"\nAttempts: completed={result.completed_steps}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    # await demo_support_router()
    await demo_moderation()
    # await demo_fan_out()
    # await demo_self_correcting()

if __name__ == "__main__":
    asyncio.run(main())
