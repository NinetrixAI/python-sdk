"""
Example 8 — Human-in-the-Loop (HITL) approval gate in a durable workflow.

Demonstrates:
  - Workflow.step(..., requires_approval=True) to mark a step as needing
    human sign-off before the workflow proceeds
  - How the workflow halts at the gate and returns "PENDING_APPROVAL"
  - How a human (or external system) inspects the pending payload and
    either approves or rejects it via the checkpointer
  - How re-running with the same thread_id resumes from the approved gate

Use case: a financial report pipeline that drafts a client-facing
          report and requires a compliance officer to approve it
          before it is sent.  If rejected, the report is revised and
          re-submitted for approval.

Flow:
  1. analyze   — analyst agent summarises the portfolio data         [auto]
  2. draft     — writer agent produces the client report draft       [auto]
  3. compliance-review — compliance officer reads and approves/rejects [HITL gate]
  4. send      — delivery agent formats the approved report          [auto, after approval]

Prerequisites:
  pip install ninetrix anthropic
  export ANTHROPIC_API_KEY=sk-...
"""

from __future__ import annotations

import asyncio
import json

from ninetrix import Agent, Workflow, InMemoryCheckpointer, WorkflowResult


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

analyst = Agent(
    name="analyst",
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    role=(
        "You are a portfolio analyst. "
        "Given portfolio data, produce a concise 3-bullet performance summary."
    ),
)

writer = Agent(
    name="writer",
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    role=(
        "You are a financial writer. "
        "Turn bullet-point analysis into a polished, professional client-facing paragraph. "
        "Keep it under 100 words. Do not include specific investment recommendations."
    ),
)

delivery_agent = Agent(
    name="delivery",
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    role=(
        "You are a delivery formatter. "
        "Format the approved report for email: add a subject line, greeting, and sign-off. "
        "Keep the body unchanged."
    ),
)


# ---------------------------------------------------------------------------
# Shared checkpointer (injected before running)
# ---------------------------------------------------------------------------

checkpointer = InMemoryCheckpointer()


# ---------------------------------------------------------------------------
# Durable workflow with HITL gate
# ---------------------------------------------------------------------------

@Workflow(durable=True)
async def report_pipeline(portfolio_data: str, thread_id: str) -> str:
    """
    Draft a client report, pause for human approval, then format for delivery.

    Returns "PENDING_APPROVAL" if the gate has not been cleared yet.
    Returns the final formatted report once approved.
    """

    # ── Step 1: analyse ─────────────────────────────────────────────────────
    async with Workflow.step("analyze") as step:
        if not step.is_cached:
            result = await analyst.arun(f"Analyse this portfolio:\n{portfolio_data}")
            step.set(result.output)
        analysis = step.value
        print(f"  [analyze]  {'(cached)' if step.is_cached else '(fresh)'}")

    # ── Step 2: draft ────────────────────────────────────────────────────────
    async with Workflow.step("draft") as step:
        if not step.is_cached:
            result = await writer.arun(f"Write a client report based on:\n{analysis}")
            step.set(result.output)
        draft = step.value
        print(f"  [draft]    {'(cached)' if step.is_cached else '(fresh)'}")

    # ── Step 3: HITL gate ────────────────────────────────────────────────────
    # requires_approval=True saves this step as status="pending_approval".
    # get_completed_steps() only returns status="completed" steps — so on the
    # next resume this step will re-run unless the external system has flipped
    # it to "completed" (the approval action).
    async with Workflow.step("compliance-review", requires_approval=True) as step:
        if not step.is_cached:
            # Prepare the review payload for the compliance officer
            review_payload = {
                "draft": draft,
                "analysis": analysis,
                "status": "pending_approval",
                "thread_id": thread_id,
            }
            step.set(review_payload)
            print(f"  [compliance-review] gate OPEN — halting for human approval")
            print(f"    thread_id : {thread_id}")
            print(f"    Call approve(checkpointer, thread_id) to unblock.\n")
            # Halt the workflow — return sentinel to signal pending state
            return "PENDING_APPROVAL"

        # Execution resumes here only after the step is approved
        approved_payload = step.value
        print(f"  [compliance-review] gate PASSED — proceeding to delivery")

        # Check if the compliance officer requested a revision
        if isinstance(approved_payload, dict) and approved_payload.get("revision_note"):
            # Revise the draft and update for delivery
            revision_note = approved_payload["revision_note"]
            revision_result = await writer.arun(
                f"Revise this report:\n{draft}\n\nCompliance note: {revision_note}"
            )
            draft = revision_result.output

    # ── Step 4: delivery ─────────────────────────────────────────────────────
    async with Workflow.step("send") as step:
        if not step.is_cached:
            result = await delivery_agent.arun(f"Format for email delivery:\n{draft}")
            step.set(result.output)
        final = step.value
        print(f"  [send]     {'(cached)' if step.is_cached else '(fresh)'}")

    return final


# ---------------------------------------------------------------------------
# Approval helpers — what an external system (API, CLI, dashboard) would call
# ---------------------------------------------------------------------------

async def approve(cp: InMemoryCheckpointer, thread_id: str, revision_note: str = "") -> None:
    """Approve the compliance-review gate for *thread_id*.

    In production this would be called by a FastAPI endpoint:
        POST /v1/approvals/{thread_id}/approve

    The mechanism: flip the step status from "pending_approval" → "completed"
    so get_completed_steps() picks it up and skips the step on resume.
    """
    steps = cp._steps.get(thread_id, {})
    review_step = steps.get("compliance-review")
    if review_step is None:
        print("No pending review found for this thread.")
        return

    # Optionally attach a revision note from the compliance officer
    payload = review_step["result"]
    if revision_note:
        payload = {**payload, "revision_note": revision_note, "status": "approved"}
    else:
        payload = {**payload, "status": "approved"}

    await cp.save_step(
        thread_id=thread_id,
        step_name="compliance-review",
        step_index=review_step["step_index"],
        result=payload,
        status="completed",   # ← this is what unlocks the gate
    )
    print(f"  ✓ Step 'compliance-review' approved for thread {thread_id}")


async def reject(cp: InMemoryCheckpointer, thread_id: str, reason: str) -> None:
    """Reject (delete) the gate step so the workflow restarts from 'draft'.

    In production: POST /v1/approvals/{thread_id}/reject
    """
    steps = cp._steps.get(thread_id, {})
    if "compliance-review" in steps:
        del steps["compliance-review"]
    if "draft" in steps:
        del steps["draft"]   # force re-draft too
    print(f"  ✗ Report rejected. Reason: {reason}")
    print(f"    Re-run with the same thread_id to regenerate the draft.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

PORTFOLIO_DATA = """
Portfolio: Balanced Growth Fund
- Equities: 60%  (+12.4% YTD)
- Fixed income: 30%  (+2.1% YTD)
- Cash: 10%
- Benchmark: +8.9% YTD
- Top performer: MSFT +34%
- Underperformer: INTC -18%
"""

async def main() -> None:
    report_pipeline.inject_checkpointer(checkpointer)
    thread = "report-q4-2025-client-007"

    # ── Run 1: workflow reaches the HITL gate and halts ──────────────────────
    print("=== Run 1: initial run ===")
    r1: WorkflowResult = await report_pipeline.arun(PORTFOLIO_DATA, thread, thread_id=thread)
    print(f"Workflow output: {r1.output!r}")
    assert r1.output == "PENDING_APPROVAL"

    # ── Simulate compliance officer reviewing the draft ───────────────────────
    print("\n=== Compliance review (simulated) ===")
    # Inspect the pending payload
    pending = checkpointer._steps.get(thread, {}).get("compliance-review", {})
    draft_for_review = pending.get("result", {}).get("draft", "")
    print(f"Draft submitted for review:\n{draft_for_review[:200]}...\n")

    # Option A: approve as-is
    # await approve(checkpointer, thread)

    # Option B: approve with a revision note
    await approve(
        checkpointer,
        thread,
        revision_note="Add a disclaimer: 'Past performance is not indicative of future results.'",
    )

    # ── Run 2: re-run with same thread_id — gate is cleared ──────────────────
    print("\n=== Run 2: resume after approval ===")
    r2: WorkflowResult = await report_pipeline.arun(PORTFOLIO_DATA, thread, thread_id=thread)

    print(f"\nCompleted steps : {r2.completed_steps}")
    print(f"Skipped steps   : {r2.skipped_steps}")
    print(f"\n{'='*60}")
    print("FINAL APPROVED REPORT:")
    print("=" * 60)
    print(r2.output)

    # ── Demonstrate rejection path ────────────────────────────────────────────
    print("\n\n=== Rejection demo (separate thread) ===")
    thread2 = "report-q4-2025-client-008"
    r3: WorkflowResult = await report_pipeline.arun(PORTFOLIO_DATA, thread2, thread_id=thread2)
    print(f"Output: {r3.output!r}")  # PENDING_APPROVAL

    await reject(checkpointer, thread2, reason="Tone is too informal. Regenerate.")
    print("After rejection, re-run with the same thread_id to get a revised draft.")


if __name__ == "__main__":
    asyncio.run(main())
