"""
workflow/context.py — per-run workflow state and budget tracker.

Layer: L8 (workflow) — may import L1 (_internals) + stdlib only.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable

from ninetrix._internals.types import AgentResult, BudgetExceededError


# ---------------------------------------------------------------------------
# WorkflowBudgetTracker
# ---------------------------------------------------------------------------


class WorkflowBudgetTracker:
    """Thread-safe budget tracker for a whole workflow run.

    Charges are accumulated across all agents in the workflow.  When the total
    exceeds *max_budget_usd* a :class:`~ninetrix.BudgetExceededError` is raised
    inside the agent call that pushed spending over the limit.

    Args:
        max_budget_usd: Spending cap for the whole workflow.
                        ``0.0`` means unlimited.
    """

    def __init__(self, max_budget_usd: float) -> None:
        self._max = max_budget_usd
        self._spent = 0.0
        self._lock = asyncio.Lock()

    async def charge(self, cost_usd: float, agent_name: str = "") -> None:
        """Add *cost_usd* to the running total.

        Raises:
            BudgetExceededError: if total spending exceeds the cap.
        """
        if self._max <= 0.0:
            # No limit set — fast path, no lock needed
            self._spent += cost_usd
            return

        async with self._lock:
            self._spent += cost_usd
            if self._spent > self._max:
                raise BudgetExceededError(
                    f"Workflow exceeded ${self._max:.2f} budget "
                    f"(spent ${self._spent:.4f}). "
                    f"Raise max_budget= on the @Workflow decorator or check "
                    f"agent '{agent_name}' for unexpectedly large calls.",
                    budget_usd=self._max,
                    spent_usd=self._spent,
                )

    @property
    def remaining(self) -> float:
        """Remaining budget in USD.  ``float('inf')`` when no limit is set."""
        if self._max <= 0.0:
            return float("inf")
        return max(0.0, self._max - self._spent)

    @property
    def spent(self) -> float:
        """Total amount charged so far."""
        return self._spent


# ---------------------------------------------------------------------------
# WorkflowContext
# ---------------------------------------------------------------------------


class WorkflowContext:
    """Per-run execution context injected into ``@Workflow`` functions.

    Provides the concurrency / branching primitives that ``WorkflowRunner``
    delegates to.  End users interact with these via ``Workflow.parallel()``,
    ``Workflow.fan_out()`` etc. — they never instantiate this class directly.

    Args:
        thread_id:  Unique run identifier.
        durable:    Reserved for PR 29.  Must be ``False`` for now.
        max_budget_usd: Per-workflow spending cap.  ``0.0`` means unlimited.
    """

    def __init__(
        self,
        thread_id: str,
        durable: bool = False,
        max_budget_usd: float = 0.0,
    ) -> None:
        self._thread_id = thread_id
        self._durable = durable
        self._budget = WorkflowBudgetTracker(max_budget_usd)
        self._step_results: dict[str, AgentResult] = {}
        self._completed_steps: list[str] = []
        self._skipped_steps: list[str] = []

    # ------------------------------------------------------------------
    # Budget helper
    # ------------------------------------------------------------------

    async def _charge(self, result: AgentResult, agent_name: str) -> None:
        """Charge the workflow budget for one agent result."""
        await self._budget.charge(result.cost_usd, agent_name)

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def step(
        self,
        name: str,
        *,
        requires_approval: bool = False,
        timeout: float | None = None,
    ) -> AsyncIterator[None]:
        """Mark an explicit step boundary (reserved for durable mode — PR 29).

        In non-durable mode this is a transparent pass-through: the block runs
        normally and the step name is recorded for introspection.

        Args:
            name: Unique step name within the workflow.
            requires_approval: (durable only) Pause for human approval.
            timeout: (durable only) Per-step timeout in seconds.
        """
        if self._durable:
            raise NotImplementedError(
                "Durable workflows are not yet supported (PR 29). "
                "Use @Workflow(durable=False) or omit the durable= keyword."
            )
        try:
            yield
        finally:
            self._completed_steps.append(name)

    async def parallel(self, *coros: Any) -> list[Any]:
        """Run multiple coroutines concurrently.

        Returns results in the same order as the inputs.

        Example::

            r1, r2 = await Workflow.parallel(
                agent_a.arun("question"),
                agent_b.arun("question"),
            )
        """
        return list(await asyncio.gather(*coros))

    async def branch(
        self,
        condition: bool,
        *,
        if_true: Callable[[], Any],
        if_false: Callable[[], Any],
    ) -> Any:
        """Conditional branch — execute *if_true* or *if_false* callable.

        Both branches must be **async** zero-argument callables (typically
        lambdas or ``functools.partial``).

        Example::

            result = await Workflow.branch(
                len(docs) > 1000,
                if_true=lambda: summarizer.arun("Long: " + text),
                if_false=lambda: analyzer.arun("Short: " + text),
            )
        """
        if condition:
            return await if_true()
        return await if_false()

    async def fan_out(
        self,
        items: list[Any],
        agent_fn: Callable[[Any], Any],
        *,
        concurrency: int = 5,
    ) -> list[Any]:
        """Map *agent_fn* over *items* with concurrency control.

        Args:
            items:       List of inputs to process.
            agent_fn:    Async callable that accepts one item.
            concurrency: Maximum number of concurrent calls.

        Returns:
            Results in the same order as *items*.

        Example::

            summaries = await Workflow.fan_out(
                documents,
                lambda doc: summarizer.arun(f"Summarize: {doc}"),
                concurrency=5,
            )
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def _bounded(item: Any) -> Any:
            async with semaphore:
                return await agent_fn(item)

        return list(await asyncio.gather(*[_bounded(item) for item in items]))

    async def reduce(
        self,
        items: list[AgentResult],
        agent: Any,
        *,
        separator: str = "\n\n---\n\n",
    ) -> AgentResult:
        """Combine multiple :class:`~ninetrix.AgentResult` objects using *agent*.

        Concatenates the output strings from *items* (separated by *separator*)
        and passes the combined text to *agent.arun()*.  Natural pair with
        :meth:`fan_out`.

        Example::

            summaries = await Workflow.fan_out(docs, summarizer.arun)
            final    = await Workflow.reduce(summaries, synthesizer)
        """
        combined = separator.join(r.output for r in items)
        return await agent.arun(f"Synthesize the following:\n\n{combined}")
