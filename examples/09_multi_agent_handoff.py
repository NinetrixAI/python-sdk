"""
Example 9 — Multi-agent handoff with dynamic routing and governance.

Demonstrates:
  - A coordinator agent that classifies incoming requests at runtime
  - Specialist agents receiving handoffs (equity, credit, macro)
  - A governance agent that reviews every specialist output before delivery
  - Dynamic re-routing: coordinator decides the specialist based on content,
    not a static Team enum
  - A governance layer that blocks output if regulatory flags are raised
    and optionally routes to a senior reviewer
  - The whole pipeline wired as a durable @Workflow so every handoff is
    a named, checkpointed, crash-resumable step

Use case: a regulated financial-research desk where three analysts handle
          different query types, and a compliance officer reviews every
          response for investment-recommendation violations before it
          reaches the client.

Flow:
  1. classify   — coordinator reads the query and picks a specialist [auto]
  2. research   — the chosen specialist produces a research summary    [auto]
  3. govern     — compliance agent reviews the output                  [auto]
  4. [optional] senior-review — if governance flags the output, a
                 senior analyst rewrites it to pass compliance          [auto]
  5. deliver    — final reviewed output is returned                    [auto]

Governance rules enforced in this example:
  - No specific investment recommendations ("buy", "sell", "overweight")
  - No specific price targets
  - Past-performance disclaimer required in any return-figure response
  - Conflict-of-interest acknowledgement required

Prerequisites:
  pip install ninetrix anthropic
  export ANTHROPIC_API_KEY=sk-...
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ninetrix import Agent, Workflow, InMemoryCheckpointer, WorkflowResult


# ---------------------------------------------------------------------------
# Specialist agents
# ---------------------------------------------------------------------------

equity_analyst = Agent(
    name="equity-analyst",
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    role=(
        "You are an equity research analyst covering public equities. "
        "You summarise company fundamentals, valuation metrics, and recent news. "
        "Present factual information only. Do NOT make buy/sell recommendations "
        "or give specific price targets. Always include: "
        "'Past performance is not indicative of future results.' "
        "when discussing historical returns."
    ),
)

credit_analyst = Agent(
    name="credit-analyst",
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    role=(
        "You are a credit research analyst specialising in fixed income and "
        "corporate debt markets. "
        "Summarise issuer credit quality, debt metrics, and spread context. "
        "Present factual information only. Do NOT make buy/sell/overweight "
        "recommendations. Always include: "
        "'Past performance is not indicative of future results.' "
        "when referencing historical spreads."
    ),
)

macro_analyst = Agent(
    name="macro-analyst",
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    role=(
        "You are a macro-economic research analyst. "
        "Summarise central bank policy, macroeconomic indicators, and global "
        "themes. Do NOT make directional trading recommendations. "
        "Use the disclaimer: "
        "'These are macro observations, not investment advice.'"
    ),
)

# Map of specialist names → Agent objects (used by the coordinator dispatcher)
SPECIALISTS: dict[str, Agent] = {
    "equity": equity_analyst,
    "credit": credit_analyst,
    "macro": macro_analyst,
}


# ---------------------------------------------------------------------------
# Coordinator agent
# ---------------------------------------------------------------------------

COORDINATOR_PROMPT = """\
You are a research coordinator. Your only job is to classify an incoming
client question into EXACTLY ONE of: equity, credit, macro.

Rules:
- "equity"  → company fundamentals, stock performance, earnings, valuation
- "credit"  → bonds, debt, credit ratings, spreads, fixed income
- "macro"   → interest rates, central banks, inflation, GDP, currencies

Reply with ONLY the single word: equity, credit, or macro.
No punctuation, no explanation.
"""

coordinator = Agent(
    name="coordinator",
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    role=COORDINATOR_PROMPT,
)


# ---------------------------------------------------------------------------
# Governance agent
# ---------------------------------------------------------------------------

GOVERNANCE_PROMPT = """\
You are a regulatory compliance officer at a financial research firm.

Your job is to review analyst research summaries and flag ANY of these violations:

VIOLATIONS:
V1 - Contains a specific investment recommendation (buy, sell, overweight, underweight,
     accumulate, avoid, hold with conviction, etc.)
V2 - Contains a specific price target (e.g. "price target of $200")
V3 - Missing past-performance disclaimer when historical returns are cited
V4 - Missing conflict-of-interest disclaimer

If the text PASSES all checks, reply with exactly:
  PASS

If the text FAILS one or more checks, reply with:
  FAIL: <comma-separated list of violation codes>
  REASON: <one sentence per violation>

Do not rewrite the text. Only classify it.
"""

governance_agent = Agent(
    name="governance",
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    role=GOVERNANCE_PROMPT,
)

# ---------------------------------------------------------------------------
# Senior reviewer agent (remediation path)
# ---------------------------------------------------------------------------

senior_reviewer = Agent(
    name="senior-reviewer",
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    role=(
        "You are a senior financial analyst and editor. "
        "You receive a research summary that has failed compliance review, "
        "along with the specific violation codes. "
        "Rewrite the summary to fix every violation while keeping all "
        "accurate factual content. "
        "Always end with: 'Past performance is not indicative of future results. "
        "This is not investment advice.'"
    ),
)


# ---------------------------------------------------------------------------
# Governance result parser
# ---------------------------------------------------------------------------

@dataclass
class GovernanceResult:
    passed: bool
    violations: list[str]
    reason: str


def parse_governance(output: str) -> GovernanceResult:
    """Parse the governance agent's PASS / FAIL verdict."""
    text = output.strip()
    if text.upper().startswith("PASS"):
        return GovernanceResult(passed=True, violations=[], reason="")

    violations: list[str] = []
    reason = ""

    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("FAIL:"):
            codes = line[5:].strip()
            violations = [c.strip() for c in codes.split(",") if c.strip()]
        elif line.upper().startswith("REASON:"):
            reason = line[7:].strip()

    return GovernanceResult(passed=False, violations=violations, reason=reason)


# ---------------------------------------------------------------------------
# Shared checkpointer
# ---------------------------------------------------------------------------

checkpointer = InMemoryCheckpointer()


# ---------------------------------------------------------------------------
# Durable workflow
# ---------------------------------------------------------------------------

@Workflow(durable=True)
async def research_pipeline(query: str, thread_id: str) -> str:
    """
    Route a client research query through the right specialist,
    apply governance review, and optionally remediate before delivery.
    """

    # ── Step 1: classify ────────────────────────────────────────────────────
    async with Workflow.step("classify") as step:
        if not step.is_cached:
            result = await coordinator.arun(
                f"Classify this client question:\n{query}"
            )
            specialist_name = result.output.strip().lower().split()[0]  # robustness
            if specialist_name not in SPECIALISTS:
                specialist_name = "macro"  # safe fallback
            step.set(specialist_name)
        specialist_name = step.value
        print(f"  [classify]  specialist={specialist_name!r}  {'(cached)' if step.is_cached else '(fresh)'}")

    # ── Step 2: research ─────────────────────────────────────────────────────
    async with Workflow.step("research") as step:
        if not step.is_cached:
            specialist = SPECIALISTS[specialist_name]
            result = await specialist.arun(query)
            step.set(result.output)
        research_output = step.value
        print(f"  [research]  agent={specialist_name}  {'(cached)' if step.is_cached else '(fresh)'}")

    # ── Step 3: governance ───────────────────────────────────────────────────
    async with Workflow.step("govern") as step:
        if not step.is_cached:
            gov_result = await governance_agent.arun(
                f"Review this research summary:\n\n{research_output}"
            )
            step.set(gov_result.output)
        governance_raw = step.value
        verdict = parse_governance(governance_raw)
        print(f"  [govern]    passed={verdict.passed}  violations={verdict.violations}  {'(cached)' if step.is_cached else '(fresh)'}")

    # ── Step 4 (conditional): senior review ──────────────────────────────────
    async with Workflow.step("senior-review") as step:
        if not verdict.passed:
            if not step.is_cached:
                violation_summary = ", ".join(verdict.violations) or "unspecified"
                remediation_prompt = (
                    f"The following research summary failed compliance review.\n"
                    f"Violations: {violation_summary}\n"
                    f"Reason: {verdict.reason}\n\n"
                    f"Original summary:\n{research_output}"
                )
                result = await senior_reviewer.arun(remediation_prompt)
                step.set(result.output)
            final_research = step.value
            print(f"  [senior-review]  remediated  {'(cached)' if step.is_cached else '(fresh)'}")
        else:
            # No remediation needed — carry forward the original output
            step.set(research_output)
            final_research = research_output
            print(f"  [senior-review]  skipped (governance passed)")

    # ── Step 5: deliver ──────────────────────────────────────────────────────
    header = (
        f"[specialist: {specialist_name}]"
        + (f"  [remediated: {', '.join(verdict.violations)}]" if not verdict.passed else "")
    )

    return f"{header}\n\n{final_research}"


# ---------------------------------------------------------------------------
# Entry point — demonstrate routing + governance
# ---------------------------------------------------------------------------

SAMPLE_QUERIES = [
    # Should route to equity-analyst
    "What are Apple's key valuation metrics and how does its YTD return compare to the S&P 500?",

    # Should route to credit-analyst
    "How has Ford Motor's credit spread changed over the past year and what is its current leverage ratio?",

    # Should route to macro-analyst
    "What is the Federal Reserve's current stance on interest rates and how is it affecting inflation?",
]


async def main() -> None:
    research_pipeline.inject_checkpointer(checkpointer)

    for i, query in enumerate(SAMPLE_QUERIES):
        thread = f"research-{i+1:03d}"
        print(f"\n{'='*60}")
        print(f"QUERY: {query[:80]}...")
        print(f"thread_id: {thread}")
        print("-" * 60)

        result: WorkflowResult = await research_pipeline.arun(
            query, thread, thread_id=thread
        )

        print(f"\n{result.output}")
        print(f"\nsteps completed : {result.completed_steps}")
        print(f"steps skipped   : {result.skipped_steps}")

    # ── Demonstrate crash-resume: re-run step 1 query ─────────────────────
    print(f"\n\n{'='*60}")
    print("RESUME DEMO — re-run query 1 with same thread_id")
    print("All steps should be served from cache (no LLM calls).")
    print("-" * 60)

    thread = "research-001"
    result2: WorkflowResult = await research_pipeline.arun(
        SAMPLE_QUERIES[0], thread, thread_id=thread
    )
    print(f"\nOutput (should match first run):\n{result2.output[:200]}...")
    print(f"\ncached steps  : {result2.skipped_steps}")


if __name__ == "__main__":
    asyncio.run(main())
