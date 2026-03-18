"""
workflow/context.py — per-run workflow state and budget tracker.

Layer: L8 (workflow) — may import L1 (_internals) + stdlib only.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from ninetrix._internals.types import AgentResult, BudgetExceededError


# ---------------------------------------------------------------------------
# _StepResult — yielded by step() context manager
# ---------------------------------------------------------------------------


class _StepResult:
    """Yielded by ``async with Workflow.step("name") as step:``.

    On a **fresh run** ``step.is_cached`` is ``False`` and ``step.value``
    is ``None``.  Call ``step.set(value)`` inside the block to store the
    result for future resumes.

    On **resume** (the step was already completed in a prior run),
    ``step.is_cached`` is ``True`` and ``step.value`` holds the
    previously stored result.  Expensive operations can be skipped::

        async with Workflow.step("research") as step:
            if not step.is_cached:
                r = await researcher.arun(topic)
                step.set(r.output)          # persist for resume
            # step.value has the result either way
    """

    def __init__(self, cached_value: Any = None, is_cached: bool = False) -> None:
        self._cached = is_cached
        self._value: Any = cached_value
        self._new_value: Any = None
        self._has_new_value: bool = False

    @property
    def is_cached(self) -> bool:
        """``True`` if this step was already completed in a prior run."""
        return self._cached

    @property
    def value(self) -> Any:
        """The step result: cached value on resume, or value passed to :meth:`set`."""
        if self._has_new_value:
            return self._new_value
        return self._value

    def set(self, value: Any) -> None:
        """Store *value* as this step's result (persisted on durable runs)."""
        self._new_value = value
        self._has_new_value = True


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
        checkpointer: Any = None,
        cached_steps: dict[str, Any] | None = None,
    ) -> None:
        self._thread_id = thread_id
        self._durable = durable
        self._budget = WorkflowBudgetTracker(max_budget_usd)
        self._step_results: dict[str, Any] = {}
        self._completed_steps: list[str] = []
        self._skipped_steps: list[str] = []
        self._checkpointer = checkpointer
        # Pre-loaded step results from a previous run (used on resume)
        self._cached_steps: dict[str, Any] = cached_steps or {}
        self._step_index: int = 0

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
    ) -> AsyncIterator[_StepResult]:
        """Mark an explicit step boundary.

        Yields a :class:`_StepResult` object.  In durable mode, if this step
        was already completed in a prior run, ``step.is_cached`` is ``True``
        and ``step.value`` contains the previously stored result — allowing
        expensive calls to be skipped::

            async with Workflow.step("research") as step:
                if not step.is_cached:
                    r = await researcher.arun(topic)
                    step.set(r.output)       # persist for future resumes
                result = step.value

        In non-durable mode the block always executes; ``step.is_cached`` is
        always ``False``.

        Args:
            name:              Unique step name within the workflow run.
            requires_approval: When ``True`` (durable mode only), the step is
                               saved with ``status="pending_approval"`` before
                               execution so a HITL system can gate the run.
            timeout:           Reserved for future step-level timeout support.
        """
        self._step_index += 1
        is_cached = self._durable and name in self._cached_steps
        cached_value = self._cached_steps.get(name) if is_cached else None
        sr = _StepResult(cached_value=cached_value, is_cached=is_cached)

        if is_cached:
            self._skipped_steps.append(name)

        try:
            yield sr
        except Exception:
            raise
        finally:
            if not is_cached:
                self._completed_steps.append(name)
                # Persist step result to checkpointer (durable mode only)
                if self._durable and self._checkpointer is not None:
                    step_status = "pending_approval" if requires_approval else "completed"
                    result_to_save = sr.value
                    await self._checkpointer.save_step(
                        thread_id=self._thread_id,
                        step_name=name,
                        step_index=self._step_index,
                        result=result_to_save,
                        status=step_status,
                    )
                    self._step_results[name] = result_to_save

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
