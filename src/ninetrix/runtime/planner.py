"""
runtime/planner.py — plan-then-execute mode for the agentic loop.

Layer: L3 (runtime) — may import L1 (_internals), L2 (tools), stdlib only.

When ``AgentConfig.execution_mode == "planned"``, ``AgentRunner`` creates a
``Planner`` and calls it once before the main tool-use loop.  The planner makes
a single cheap LLM call (low temperature, small token budget) to produce a
structured JSON plan, then :meth:`build_execution_prompt` injects that plan
into the user message so the execution loop has a concrete step-by-step guide.

Design decisions
----------------
* The LLM used for planning is the *same* provider + model configured on the
  agent.  A separate, cheaper model (e.g. Haiku) can be specified via
  ``RunnerConfig.planner_model`` in a future PR; for v1 we keep it simple.
* The plan schema is intentionally minimal: ``{"goal": str, "steps": list[str]}``.
  It's designed to be parseable even if the LLM wraps the JSON in a code fence.
* Parse failures silently fall back to returning ``{}`` so that a bad planner
  response never breaks the run — the execution loop just receives the original
  message unchanged.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ninetrix.runtime.runner import RunnerConfig


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = """\
You are a planning assistant.  Given a user request and a list of available
tools, produce a concise execution plan.

Respond ONLY with valid JSON — no markdown, no code fences, no explanation:
{"goal": "<one sentence: what the user wants to achieve>",
 "steps": ["<step 1>", "<step 2>", "<step 3>"]}

Rules:
- Be concrete: mention tool names where relevant.
- Maximum 6 steps.
- Each step must be a single actionable sentence.
- Do NOT include steps that are not needed to answer the request.
"""


class Planner:
    """Pre-execution planning step for plan-then-execute agent runs.

    ``Planner`` makes a single LLM call to produce a structured plan before
    the main tool-use loop.  The plan is injected into the user message so the
    agent has a concrete step-by-step guide.

    Used automatically by :class:`~ninetrix.AgentRunner` when
    ``RunnerConfig.execution_mode == "planned"``.  Most users never
    instantiate this directly.

    Args:
        config: The runner configuration from which model + provider settings
                are inherited.

    Example (direct use)::

        planner = Planner(RunnerConfig(name="bot"))
        plan = await planner.plan("Summarise the top 5 Hacker News stories",
                                  tool_defs, provider)
        exec_msg = planner.build_execution_prompt(
            "Summarise the top 5 Hacker News stories", plan
        )
        # exec_msg now contains the goal + steps injected into the message
    """

    def __init__(self, config: RunnerConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def plan(
        self,
        message: str,
        tool_defs: list[dict],
        provider: Any,
    ) -> dict:
        """Ask the LLM to produce a structured plan for *message*.

        Makes a single LLM call with a low token budget.  Parse failures are
        silently swallowed — the caller receives ``{}`` and must treat that as
        "no plan available; proceed directly."

        Args:
            message:   The user's original request.
            tool_defs: OpenAI-compatible tool schemas from the dispatcher.
            provider:  The LLM provider adapter to use for the planning call.

        Returns:
            A dict ``{"goal": str, "steps": list[str]}``, or ``{}`` on failure.
        """
        tool_names = [
            t["function"]["name"]
            for t in tool_defs
            if isinstance(t, dict) and "function" in t
        ]
        tools_section = (
            "Available tools: " + ", ".join(tool_names)
            if tool_names
            else "No tools available."
        )

        planning_messages = [
            {"role": "system", "content": _PLAN_SYSTEM},
            {
                "role": "user",
                "content": f"User request: {message}\n\n{tools_section}",
            },
        ]

        try:
            from ninetrix._internals.types import ProviderConfig
            pconfig = ProviderConfig(temperature=0.0, max_tokens=512)
            response = await provider.complete(
                planning_messages, [], config=pconfig
            )
            raw = (response.content or "").strip()
            return _parse_plan(raw)
        except Exception:
            return {}

    def build_execution_prompt(self, message: str, plan: dict) -> str:
        """Inject *plan* into *message* so the execution loop has a guide.

        If the plan is empty or has no steps, *message* is returned unchanged.

        Args:
            message: The original user request.
            plan:    The dict returned by :meth:`plan`.

        Returns:
            The original message, extended with a formatted plan block when
            *plan* is non-empty.
        """
        if not plan:
            return message

        goal = plan.get("goal", "")
        steps = plan.get("steps", [])

        if not steps:
            return message

        steps_text = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(steps))
        goal_line = f"Goal: {goal}\n" if goal else ""

        return (
            f"{message}\n\n"
            f"--- Execution Plan ---\n"
            f"{goal_line}"
            f"Steps:\n{steps_text}\n"
            f"--- End Plan ---\n\n"
            f"Execute the plan above step by step using the available tools."
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_plan(raw: str) -> dict:
    """Parse a JSON plan from a raw LLM response string.

    Handles:
    - Bare JSON: ``{"goal": "...", "steps": [...]}``
    - JSON wrapped in a code fence: ``\\`\\`\\`json ... \\`\\`\\`\\``
    - Extra whitespace / trailing text after the closing brace
    """
    if not raw:
        return {}

    # Strip code fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if fence_match:
        raw = fence_match.group(1).strip()

    # Extract first {...} block
    brace_match = re.search(r"\{[\s\S]*\}", raw)
    if brace_match:
        raw = brace_match.group(0)

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    if not isinstance(plan, dict):
        return {}

    # Normalise: goal must be a string, steps must be a list of strings
    goal = plan.get("goal", "")
    if not isinstance(goal, str):
        goal = str(goal)

    steps = plan.get("steps", [])
    if not isinstance(steps, list):
        steps = []
    steps = [str(s) for s in steps if s]

    return {"goal": goal, "steps": steps}
