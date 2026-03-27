"""
Example 4 — Multi-agent Team with LLM-based routing.

Demonstrates:
  - Building a Team of specialist agents
  - Dynamic routing: the Team uses an LLM to classify the incoming
    message and forward it to the right specialist
  - TeamResult with routing metadata
  - Wiring a Team inside a Workflow for full pipeline control

Use case: a customer-support router that sends billing questions to
          the billing agent, technical questions to the tech agent,
          and general questions to the general agent.

Prerequisites:
  pip install ninetrix anthropic
  export ANTHROPIC_API_KEY=sk-...
"""

from __future__ import annotations

import asyncio

from ninetrix import Agent, Team, Workflow

# ---------------------------------------------------------------------------
# Specialist agents
# ---------------------------------------------------------------------------

billing_agent = Agent(
    name="billing",
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    role=(
        "You are a billing specialist. "
        "Help customers with invoices, payments, refunds, and subscription changes. "
        "Be concise and friendly."
    ),
    description="Handles billing, payments, invoices, and subscription questions.",
)

tech_agent = Agent(
    name="tech-support",
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    role=(
        "You are a technical support engineer. "
        "Help customers debug issues, explain error messages, and walk them through fixes. "
        "Be precise and step-by-step."
    ),
    description="Handles technical issues, bugs, errors, and integration questions.",
)

general_agent = Agent(
    name="general",
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    role=(
        "You are a friendly general support agent. "
        "Answer product questions, onboarding queries, and anything that doesn't fit "
        "billing or tech support."
    ),
    description="Handles general product questions, onboarding, and feedback.",
)


# ---------------------------------------------------------------------------
# Team — routes automatically
# ---------------------------------------------------------------------------

support_team = Team(
    agents=[billing_agent, tech_agent, general_agent],
    router_provider="anthropic",
    router_model="claude-haiku-4-5-20251001",
    name="customer-support",
    description="Routes customer enquiries to the right specialist.",
)


# ---------------------------------------------------------------------------
# Workflow that uses the team
# ---------------------------------------------------------------------------

@Workflow
async def support_workflow(message: str) -> str:
    team_result = await support_team.arun(message)
    return (
        f"[routed to: {team_result.routed_to}]\n"
        f"{team_result.output}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

SAMPLE_QUESTIONS = [
    "I was charged twice for my subscription this month.",
    "The API keeps returning a 429 rate-limit error even though I'm under my quota.",
    "How do I add a second user to my account?",
]


async def main() -> None:
    for question in SAMPLE_QUESTIONS:
        print(f"\nQ: {question}")
        result = await support_workflow.arun(question)
        print(f"A: {result.output}")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
