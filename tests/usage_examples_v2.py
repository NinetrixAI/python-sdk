"""
usage_examples_v2.py — North-star for the v2 developer API.

This file defines the TARGET DX for the Ninetrix SDK.
It does NOT run today — it describes what the code should look like
after the v2 API changes are implemented.

Design goals:
  1. One `Ninetrix(provider=..., model=...)` context object eliminates
     all provider/model repetition across agents, teams, and routers.
  2. `nx.agent(name, description=...)` — minimal definition for routing agents;
     `description` auto-derives the system prompt when `role` is absent.
  3. `nx.team(name, [agents])` — routing is automatic; no `router_provider`
     or `router_model` params needed.
  4. Factory methods (`nx.agent`, `nx.team`) make context binding explicit
     (no global state / Flask-style app-context problems).
  5. `result.agent_name` instead of `result.routed_to`.
  6. `@nx.workflow(durable=True)` injects the context checkpointer automatically.
  7. `Workflow.run_step(name, fn)` — declarative step caching, no boilerplate.
  8. `Workflow.map(agent, items, ...)` — built-in fan-out with concurrency control.
  9. Everything that worked in v1 still works — v2 is additive, not breaking.

Run this file once the v2 implementation lands:
    python tests/usage_examples_v2.py
    pytest tests/usage_examples_v2.py -v
"""

# =============================================================================
# 0. IMPORTS
# =============================================================================

import asyncio
from ninetrix import Ninetrix, Tool, Workflow, InMemoryCheckpointer


# =============================================================================
# 1. CONTEXT OBJECT
#    One place to set provider, model, and global defaults.
#    All nx.agent / nx.team / nx.workflow calls inherit these.
# =============================================================================

nx = Ninetrix(
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    # Optional global limits — any agent / team created via nx inherits them:
    # budget_usd=1.0,
    # max_turns=10,
    # checkpointer=InMemoryCheckpointer(),
)

# You can create multiple contexts for different environments:
# nx_prod = Ninetrix(provider="anthropic", model="claude-sonnet-4-6")
# nx_test = Ninetrix(provider="anthropic", model="claude-haiku-4-5-20251001")


# =============================================================================
# 2. DEFINE TOOLS  (unchanged from v1 — @Tool is already clean)
# =============================================================================

@Tool
def get_stock_price(ticker: str) -> dict:
    """Fetch the current stock price for a ticker symbol.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
    """
    prices = {"AAPL": 189.50, "MSFT": 415.20}
    return {"ticker": ticker.upper(), "price": prices.get(ticker.upper(), 0.0)}


@Tool
def get_account_balance(account_id: str) -> dict:
    """Fetch the current balance for a customer account.

    Args:
        account_id: Customer account identifier.
    """
    return {"account_id": account_id, "balance": 1024.50, "currency": "USD"}


# =============================================================================
# 3. SINGLE AGENT — minimal vs full definition
# =============================================================================

# --- Minimal: description only ---
# `role` is auto-derived: "You are a helpful assistant that handles {description}."
# Good for routing agents that don't need a custom system prompt.
simple_analyst = nx.agent(
    "analyst",
    description="Answers questions about stock prices and market data",
    tools=[get_stock_price],
)

# --- Full control: explicit role overrides description ---
# `description` is still used by the Team router.
# `role` becomes the actual system prompt.
compliance_analyst = nx.agent(
    "compliance",
    description="Checks research output for regulatory violations",
    role=(
        "You are a regulatory compliance officer at a financial firm. "
        "Review content for investment recommendation violations. "
        "Reply PASS or FAIL:<codes>."
    ),
    # Override the global model for this one agent:
    model="claude-sonnet-4-6",
)

# --- Agent with all options ---
senior_analyst = nx.agent(
    "senior-analyst",
    description="Deep fundamental analysis of public companies",
    role="You are a senior equity analyst with 20 years of experience...",
    tools=[get_stock_price],
    max_turns=15,
    budget_usd=0.10,
)

# Run (same as v1 — unchanged)
async def _test_single_agent():
    result = await simple_analyst.arun("What is the price of AAPL?")
    print(f"Output : {result.output}")
    print(f"Tokens : {result.tokens_used}  |  Cost: ${result.cost_usd:.4f}")
    print(f"Thread : {result.thread_id}")


# =============================================================================
# 4. TEAM — auto-routing, no boilerplate
# =============================================================================

# v1 required: router_provider="anthropic", router_model="claude-haiku-4-5-20251001"
# v2: the team inherits provider+model from nx; routing is automatic.

support_team = nx.team(
    "customer-support",
    agents=[
        nx.agent("billing",      description="Handles invoices, payments, and refunds"),
        nx.agent("tech-support", description="Debugs API errors, integration issues, and outages"),
        nx.agent("general",      description="General product questions and onboarding"),
    ],
)

# --- Result has agent_name instead of routed_to ---
async def _test_team():
    questions = [
        "I was charged twice for my subscription this month.",
        "The API keeps returning a 429 rate-limit error.",
        "How do I add a second user to my account?",
    ]
    for question in questions:
        result = await support_team.run(question)
        print(f"Q: {question[:60]}")
        print(f"  → agent: {result.agent_name}  |  {result.output[:80]}...")
        print()


# You can still pass explicit agents defined elsewhere:
# support_team = nx.team("support", agents=[billing_agent, tech_agent])

# Or mix nx.agent with plain Agent (v1-style) — both are AgentProtocol:
# from ninetrix import Agent
# v1_agent = Agent(name="legacy", provider="anthropic", model="...", role="...")
# mixed_team = nx.team("mixed", agents=[v1_agent, nx.agent("new", description="...")])


# =============================================================================
# 5. DURABLE WORKFLOW — checkpointer from context
# =============================================================================

# v1 required: report_pipeline.inject_checkpointer(checkpointer)
# v2: @nx.workflow() injects the context's checkpointer automatically.
# If nx has no checkpointer set, it defaults to InMemoryCheckpointer.

@nx.workflow(durable=True)
async def research_pipeline(query: str) -> str:
    async with Workflow.step("research") as step:
        if not step.is_cached:
            result = await simple_analyst.arun(query)
            step.set(result.output)
        research = step.value

    async with Workflow.step("format") as step:
        if not step.is_cached:
            result = await senior_analyst.arun(
                f"Format this for a client report:\n{research}"
            )
            step.set(result.output)
        return step.value


# Provide a shared checkpointer via context (for cross-service persistence):
nx_durable = Ninetrix(
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    checkpointer=InMemoryCheckpointer(),   # or PostgresCheckpointer("postgresql://...")
)

@nx_durable.workflow(durable=True)
async def durable_pipeline_shared(query: str) -> str:
    async with Workflow.step("step1") as step:
        if not step.is_cached:
            result = await simple_analyst.arun(query)
            step.set(result.output)
    return step.value


async def _test_workflow():
    # First run — executes all steps
    r1 = await research_pipeline.arun("Analyse AAPL", thread_id="report-001")
    print(f"Run 1: {r1.output[:100]}...")
    print(f"  completed: {r1.completed_steps}, skipped: {r1.skipped_steps}")

    # Re-run same thread — all steps cached
    r2 = await research_pipeline.arun("Analyse AAPL", thread_id="report-001")
    print(f"Run 2 (cached): skipped={r2.skipped_steps}")


# =============================================================================
# 6. HITL GATE — no change in mechanics, cleaner setup
# =============================================================================

nx_hitl = Ninetrix(
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    checkpointer=InMemoryCheckpointer(),
)

@nx_hitl.workflow(durable=True)
async def approval_pipeline(draft: str) -> str:
    # Step 1: generate
    async with Workflow.step("generate") as step:
        if not step.is_cached:
            result = await simple_analyst.arun(f"Improve this draft:\n{draft}")
            step.set(result.output)
        content = step.value

    # Step 2: HITL gate — halts and waits for human approval
    async with Workflow.step("review", requires_approval=True) as step:
        if not step.is_cached:
            step.set({"content": content, "status": "pending"})
            return "PENDING_APPROVAL"
        approved = step.value

    return approved.get("content", content)


# =============================================================================
# 7. MULTI-AGENT HANDOFF WITH GOVERNANCE
# =============================================================================

research_desk = nx.team(
    "research-desk",
    agents=[
        nx.agent("equity",  description="Company fundamentals, stocks, earnings, valuation"),
        nx.agent("credit",  description="Corporate bonds, credit ratings, spreads, debt"),
        nx.agent("macro",   description="Central banks, interest rates, inflation, GDP"),
    ],
)

# Governance wraps any team or agent
governance = nx.agent(
    "compliance",
    description="Reviews research output for regulatory violations",
    role=(
        "You are a compliance officer. "
        "Check for: specific investment recommendations (buy/sell), "
        "specific price targets, missing past-performance disclaimers. "
        "Reply PASS or FAIL:<violations>."
    ),
)

@nx.workflow(durable=True)
async def governed_research(query: str) -> str:
    # Route to specialist
    async with Workflow.step("research") as step:
        if not step.is_cached:
            result = await research_desk.run(query)
            step.set({"output": result.output, "agent": result.agent_name})
        research = step.value

    # Governance review
    async with Workflow.step("govern") as step:
        if not step.is_cached:
            verdict = await governance.arun(
                f"Review this research output:\n{research['output']}"
            )
            step.set(verdict.output)
        gov_verdict = step.value

    if gov_verdict.strip().upper().startswith("FAIL"):
        return f"[BLOCKED by compliance: {gov_verdict}]"

    return f"[specialist: {research['agent']}]\n\n{research['output']}"


# =============================================================================
# 8. TESTING — sandbox still works the same way
# =============================================================================

# from ninetrix import AgentSandbox, MockTool
# import pytest
#
# @pytest.mark.asyncio
# async def test_team_routing():
#     mock_price = MockTool("get_stock_price", return_value={"price": 100.0})
#     async with AgentSandbox(simple_analyst, tools=[mock_price]) as sb:
#         result = await sb.run("What is the price of AAPL?")
#     assert "100" in result.output
#     mock_price.assert_called_once()
#
# @pytest.mark.asyncio
# async def test_governed_research():
#     # Override the nx checkpointer with a fresh one for each test
#     nx_test = Ninetrix(
#         provider="anthropic",
#         model="claude-haiku-4-5-20251001",
#         checkpointer=InMemoryCheckpointer(),
#     )
#     # ... rest of test


# =============================================================================
# 9. OBSERVABILITY — unchanged from v1, still works
# =============================================================================

# from ninetrix import enable_debug, configure_otel
#
# enable_debug(agent=simple_analyst)   # pretty-printer per agent
# # or NINETRIX_DEBUG=1 for all agents
#
# configure_otel(
#     endpoint="http://localhost:4317",
#     service_name="research-desk",
# )
# # Every agent in the process emits spans automatically after this.


# =============================================================================
# DELTA: v1 → v2 at a glance
# =============================================================================
#
#  v1                                          v2
#  ──────────────────────────────────────────  ──────────────────────────────
#  Agent(name="billing",                       nx.agent(
#      provider="anthropic",                       "billing",
#      model="claude-haiku-4-5-20251001",          description="Handles billing",
#      role="You are a billing specialist...",  )   ← role auto-derived
#      description="Handles billing",
#  )
#
#  Team(                                       nx.team("support", [
#      agents=[...],                               nx.agent(...),
#      router_provider="anthropic",                nx.agent(...),
#      router_model="claude-haiku-4-5-20251001",   nx.agent(...),
#      name="support",                         ])
#  )
#
#  report_pipeline.inject_checkpointer(cp)     @nx.workflow(durable=True)
#  @Workflow(durable=True)                     ← checkpointer from context
#
#  result.routed_to                            result.agent_name
#
#  Everything in v1 still works. v2 is purely additive.


# =============================================================================
# 10. DECLARATIVE STEP CACHING  (Workflow.run_step)
#
# Problem with v1:
#   - Every step requires 4 lines of boilerplate (async with, if not cached,
#     step.set(), step.value)
#   - The caching mechanics obscure the business logic
#
# v2: Workflow.run_step(name, fn) handles caching transparently.
#   - `fn` is only called if the step is not already cached
#   - Returns whatever `fn()` returns (generic T) — no magic extraction
#   - The step name is the cache key; same semantics as Workflow.step()
# =============================================================================

# --- v1 (current) ---
# @Workflow(durable=True)
# async def content_pipeline_v1(topic: str) -> str:
#     async with Workflow.step("research") as step:
#         if not step.is_cached:
#             result = await researcher.arun(f"Research: {topic}")
#             step.set(result.output)
#         research = step.value
#
#     async with Workflow.step("draft") as step:
#         if not step.is_cached:
#             result = await writer.arun(f"Write blog intro for: {research}")
#             step.set(result.output)
#         draft = step.value
#
#     async with Workflow.step("edit") as step:
#         if not step.is_cached:
#             result = await editor.arun(f"Edit this draft: {draft}")
#             step.set(result.output)
#     return step.value

# --- v2 (target) ---
# @nx.workflow(durable=True, max_budget=0.10)
# async def content_pipeline(topic: str) -> str:
#     # Workflow.run_step(name, fn):
#     #   - fn is a zero-arg async callable (lambda or async def)
#     #   - awaited only if not cached; cached value returned directly
#     #   - return type: AgentResult (caller uses .output as needed)
#     research = await Workflow.run_step(
#         "research",
#         lambda: researcher.arun(f"Research: {topic}")
#     )
#
#     draft = await Workflow.run_step(
#         "draft",
#         lambda: writer.arun(f"Write blog intro for: {research.output}")
#     )
#
#     final = await Workflow.run_step(
#         "edit",
#         lambda: editor.arun(f"Edit this draft: {draft.output}")
#     )
#
#     return final.output


# When the step result is a plain value (not AgentResult), it's stored as-is:
# @nx.workflow(durable=True)
# async def etl_pipeline(record_id: str) -> dict:
#     raw = await Workflow.run_step(
#         "fetch",
#         lambda: fetch_record(record_id)   # returns dict, not AgentResult
#     )
#     enriched = await Workflow.run_step(
#         "enrich",
#         lambda: analyst.arun(f"Enrich: {raw}")
#     )
#     return enriched.output

# Design note: run_step is generic — Workflow.run_step[T](name, Callable[[], Awaitable[T]]) -> T
# When fn returns AgentResult, the caller accesses .output explicitly.
# This keeps type safety: IDE knows the return type from the lambda signature.
# The v1 async-with form remains available for steps that need HITL
# (requires_approval=True) or conditional branching that can't fit in a lambda.


# =============================================================================
# 11. ROUTER PATTERN  (branch on step result)
#
# run_step extracts .output from AgentResult automatically.
# Return type is T — the inner output type:
#   Agent(output_type=None)          → run_step returns str
#   Agent(output_type=SomeModel)     → run_step returns SomeModel
#   lambda: fetch_db(id) -> dict     → run_step returns dict (no extraction needed)
#
# This makes step results directly comparable — no .output everywhere.
# =============================================================================

# @nx.workflow(durable=True)
# async def smart_support_pipeline(query: str) -> str:
#     # category is str (auto-extracted from AgentResult.output)
#     category = await Workflow.run_step(
#         "classify_intent",
#         lambda: intent_classifier.arun(query)
#     )
#
#     if category == "technical":
#         return await Workflow.run_step(
#             "tech_fix",
#             lambda: tech_agent.arun(query)
#         )
#     elif category == "billing":
#         return await Workflow.run_step(
#             "billing_fix",
#             lambda: billing_agent.arun(query)
#         )
#
#     # Default: unhandled category
#     return "Forwarding to human agent..."


# Structured routing — agent returns a Pydantic model:
#
# class Intent(BaseModel):
#     category: str          # "technical" | "billing" | "general"
#     urgency: str           # "low" | "medium" | "high"
#     summary: str
#
# intent_classifier = nx.agent(
#     "intent-classifier",
#     role="Classify the user's support query.",
#     output_type=Intent,
# )
#
# @nx.workflow(durable=True)
# async def smart_support_structured(query: str) -> str:
#     intent = await Workflow.run_step(
#         "classify_intent",
#         lambda: intent_classifier.arun(query)
#     )
#     # intent is Intent — full IDE completion, no .output needed
#     print(f"Category: {intent.category}, Urgency: {intent.urgency}")
#
#     if intent.urgency == "high":
#         return await Workflow.run_step(
#             "escalate",
#             lambda: escalation_agent.arun(f"URGENT: {query}")
#         )
#     if intent.category == "technical":
#         return await Workflow.run_step("tech_fix", lambda: tech_agent.arun(query))
#     return await Workflow.run_step("general_fix", lambda: general_agent.arun(query))


# =============================================================================
# 12. GUARD STEP + EARLY TERMINATION  (Workflow.terminate)
#
# Workflow.terminate(reason) stops execution early.
# WorkflowRunner catches it and returns:
#   WorkflowResult(output=None, terminated=True, termination_reason=reason)
# Callers check result.terminated before using result.output.
#
# Semantically different from `return "..."`:
#   return "..."          → success path, output is the string
#   Workflow.terminate()  → failure/abort path, caller handles it differently
# =============================================================================

# @nx.workflow(durable=True)
# async def publication_workflow(article: str) -> str:
#     # is_safe is str — auto-extracted from AgentResult.output
#     is_safe = await Workflow.run_step(
#         "safety_check",
#         lambda: moderator.arun(article)
#     )
#
#     if is_safe.strip().upper() != "PASS":
#         # Workflow.terminate() raises WorkflowTerminated internally.
#         # WorkflowRunner catches it and builds:
#         #   WorkflowResult(terminated=True, termination_reason="Content failed safety check")
#         return Workflow.terminate("Content failed safety check")
#
#     return await Workflow.run_step("publish", lambda: publisher.arun(article))

# Caller:
# result = await publication_workflow.arun(article_text, thread_id="pub-001")
# if result.terminated:
#     print(f"Blocked: {result.termination_reason}")
# else:
#     print(result.output)


# Multi-guard pipeline:
#
# @nx.workflow(durable=True)
# async def strict_pipeline(content: str) -> str:
#     for check_name, checker in [
#         ("pii_check",     pii_detector),
#         ("safety_check",  moderator),
#         ("legal_check",   legal_reviewer),
#     ]:
#         verdict = await Workflow.run_step(
#             check_name, lambda: checker.arun(content)
#         )
#         if verdict.strip().upper() != "PASS":
#             return Workflow.terminate(f"Failed {check_name}: {verdict}")
#
#     return await Workflow.run_step("process", lambda: processor.arun(content))


# =============================================================================
# 13. DYNAMIC LOOPS  (retry / self-correction patterns)
#
# Dynamic step names (f-strings) give each loop iteration its own cache key.
# If the workflow crashes on attempt 2, it resumes from "test_run_2" —
# not from the beginning.
#
# Lambda closure note: lambdas that capture loop variables are safe here
# because Workflow.run_step awaits immediately. The lambda is never deferred.
#
# Structured step results: use output_type=YourModel on the agent to get
# a Pydantic model back from run_step instead of a plain string.
# =============================================================================

# class TestResult(BaseModel):
#     passed: bool
#     error: str = ""
#
# tester = nx.agent("tester", role="Run the tests and report results.", output_type=TestResult)
#
# @nx.workflow(durable=True)
# async def coding_assistant(task: str) -> str:
#     code = await Workflow.run_step(
#         "write_code",
#         lambda: coder.arun(task)
#     )
#
#     for attempt in range(3):
#         # Each attempt has a unique step name → own cache key
#         test_result = await Workflow.run_step(
#             f"test_run_{attempt}",
#             lambda: tester.arun(code)
#         )
#         # test_result is TestResult (structured) — .passed and .error available
#         if test_result.passed:
#             return code
#
#         code = await Workflow.run_step(
#             f"fix_code_{attempt}",
#             lambda: coder.arun(f"Fix this code:\n{code}\n\nError:\n{test_result.error}")
#         )
#
#     # Exhausted all retries
#     return Workflow.terminate("Failed to produce working code after 3 attempts")

# Caller:
# result = await coding_assistant.arun("Write a binary search function", thread_id="code-001")
# if result.terminated:
#     print(f"Gave up: {result.termination_reason}")
# else:
#     print(result.output)
#
# Re-run same thread_id after crash:
# result = await coding_assistant.arun("Write a binary search function", thread_id="code-001")
# # Resumes from the step that was not yet cached — e.g. "fix_code_1"


# =============================================================================
# 14. FAN-OUT WITH CONCURRENCY CONTROL  (Workflow.map)
#
# Problem with v1:
#   - asyncio.gather([agent.arun(p) for p in prompts]) is the user's job
#   - No concurrency limit, no labelling, no caching per-item
#
# v2: Workflow.map(agent, items, ...) — built-in fan-out
#   - Runs agent on each item concurrently (up to `concurrency` at a time)
#   - `prefix` prepended to each item string
#   - Returns list[AgentResult] in the same order as items
#   - Design rationale: lives on Workflow, not Agent — keeps Agent lean
# =============================================================================

# --- v1 (current) ---
# @Workflow
# async def research_report_v1(topics: list[str]) -> str:
#     results = await asyncio.gather(*[
#         researcher.arun(f"Research topic: {t}") for t in topics
#     ])
#     summaries = [r.output for r in results]
#     report = await synthesizer.arun("\n\n".join(summaries))
#     return report.output

# --- v2 (target) ---
# @nx.workflow
# async def research_report(topics: list[str]) -> str:
#     # Workflow.map(agent, items, prefix=..., concurrency=N)
#     #   - Returns list[AgentResult], same order as `items`
#     #   - concurrency=5 means at most 5 arun() calls in flight at once
#     results = await Workflow.map(
#         researcher,
#         topics,
#         prefix="Research this topic in depth: ",
#         concurrency=5,
#     )
#
#     # Reduce is just Python — no special API needed
#     combined = "\n\n---\n\n".join(r.output for r in results)
#     report = await synthesizer.arun(f"Synthesise into one report:\n\n{combined}")
#     return report.output

# For durable fan-out (each item is a named step, cached independently):
# @nx.workflow(durable=True)
# async def durable_research(topics: list[str]) -> str:
#     results = await Workflow.map(
#         researcher,
#         topics,
#         prefix="Research: ",
#         concurrency=3,
#         step_prefix="research",   # creates steps: research_0, research_1, ...
#     )
#     combined = "\n\n".join(r.output for r in results)
#     summary = await Workflow.run_step(
#         "summarise",
#         lambda: synthesizer.arun(combined)
#     )
#     return summary.output

# Design note on .reduce():
# agent.reduce(items, template="...{{items}}...") was considered and rejected.
# Reason: it's just `agent.arun("\n\n".join(items))` — no new API needed.
# Template syntax adds complexity (Jinja2 in a Python file) for zero expressiveness gain.
# Users who need custom reduction just write the string themselves.


# =============================================================================
# FULL v2 DELTA TABLE
# =============================================================================
#
#  v1                                          v2
#  ──────────────────────────────────────────  ─────────────────────────────────────────
#  Agent(name="billing",                       nx.agent(
#      provider="anthropic",                       "billing",
#      model="claude-haiku-4-5-20251001",          description="Handles billing",
#      role="You are a billing specialist...",  )   ← role auto-derived
#      description="Handles billing",
#  )
#
#  Team(                                       nx.team("support", [
#      agents=[...],                               nx.agent(...),
#      router_provider="anthropic",            ])   ← router inherits from nx
#      router_model="claude-haiku",
#      name="support",
#  )
#
#  result.routed_to                            result.agent_name
#
#  report_pipeline                             @nx.workflow(durable=True)
#  .inject_checkpointer(cp)                    ← checkpointer from context
#  @Workflow(durable=True)
#
#  async with Workflow.step("x") as s:         result = await Workflow.run_step(
#      if not s.is_cached:                         "x", lambda: agent.arun(prompt)
#          r = await agent.arun(prompt)        )   ← returns str/model, not AgentResult
#          s.set(r.output)                         ← if category == "technical": works
#      x = s.value
#
#  return "blocked"                            return Workflow.terminate("reason")
#  # ambiguous — is it output or error?        # explicit abort; result.terminated=True
#
#  asyncio.gather(*[                           results = await Workflow.map(
#      agent.arun(f"Research: {t}")                agent, topics,
#      for t in topics                             prefix="Research: ",
#  ])                                              concurrency=5,
#                                              )
#
#  Everything in v1 still works. v2 is purely additive.
#
#
# =============================================================================
# IMPLEMENTATION CHECKLIST (SDK changes required)
# =============================================================================
#
#  [ ] ninetrix/context.py         — new Ninetrix class + nx.agent / nx.team / nx.workflow
#  [ ] agent/config.py             — auto-derive role from description when role absent
#  [ ] workflow/team.py            — router_provider / router_model optional (inherit from context)
#  [ ] _internals/types.py         — TeamResult.agent_name alias for routed_to
#  [ ] _internals/types.py         — WorkflowResult: add terminated: bool, termination_reason: str
#  [ ] workflow/workflow.py        — WorkflowRunner accepts checkpointer from Ninetrix context
#  [ ] workflow/workflow.py        — Workflow.run_step(name, fn): extracts .output from AgentResult
#  [ ] workflow/workflow.py        — Workflow.terminate(reason): raises WorkflowTerminated sentinel
#  [ ] workflow/workflow.py        — Workflow.map(agent, items, concurrency, step_prefix) class method
#  [ ] __init__.py                 — export Ninetrix


if __name__ == "__main__":
    print("usage_examples_v2.py — this file defines the TARGET DX.")
    print("It will run once the v2 API is implemented.")
    print()
    print("SDK changes required (see IMPLEMENTATION CHECKLIST above):")
    print("  1. ninetrix/context.py  — Ninetrix class + nx.agent / nx.team / nx.workflow")
    print("  2. agent/config.py      — auto-derive role from description")
    print("  3. workflow/team.py     — router_provider / router_model optional")
    print("  4. _internals/types.py  — TeamResult.agent_name alias")
    print("  5. workflow/workflow.py — checkpointer from context")
    print("  6. workflow/workflow.py — Workflow.run_step(name, fn)")
    print("  7. workflow/workflow.py — Workflow.map(agent, items, ...)")
    print("  8. __init__.py          — export Ninetrix")
