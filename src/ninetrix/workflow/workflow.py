"""
workflow/workflow.py — @Workflow decorator and WorkflowRunner.

Layer: L8 (workflow) — may import L1 (_internals) + L8 (context) + stdlib only.

Usage::

    from ninetrix import Workflow, Agent

    researcher = Agent(provider="anthropic", role="Researcher")
    writer     = Agent(provider="anthropic", role="Writer")

    @Workflow
    async def pipeline(question: str) -> str:
        r1 = await researcher.arun(question)
        r2 = await writer.arun(r1.output)
        return r2.output

    result = pipeline.run("What is quantum computing?")
    print(result.output)

    # Parallel branches
    @Workflow(max_budget=1.00)
    async def multi(question: str) -> str:
        r1, r2 = await Workflow.parallel(
            researcher.arun(question),
            writer.arun(question),
        )
        return r1.output + "\\n---\\n" + r2.output
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextvars import ContextVar
from typing import Any, Callable

from ninetrix._internals.types import AgentResult, WorkflowResult
from ninetrix.workflow.context import WorkflowContext


# ---------------------------------------------------------------------------
# ContextVar — current running WorkflowContext
# ---------------------------------------------------------------------------

_current_ctx: ContextVar[WorkflowContext | None] = ContextVar(
    "_current_ctx", default=None
)


def _get_ctx() -> WorkflowContext:
    ctx = _current_ctx.get()
    if ctx is None:
        raise RuntimeError(
            "Workflow helpers (Workflow.parallel, Workflow.fan_out, etc.) "
            "must be called inside a @Workflow-decorated async function."
        )
    return ctx


# ---------------------------------------------------------------------------
# WorkflowRunner
# ---------------------------------------------------------------------------


class WorkflowRunner:
    """Wraps a ``@Workflow``-decorated async function.

    Provides ``.run()``, ``.arun()``, and exposes the thread_id + cost
    information via :class:`~ninetrix.WorkflowResult`.

    You never instantiate this directly — the :data:`Workflow` decorator
    does it automatically.
    """

    def __init__(
        self,
        fn: Callable,
        *,
        durable: bool,
        name: str,
        db_url: str | None,
        max_budget_usd: float,
    ) -> None:
        self._fn = fn
        self._durable = durable
        self._name = name
        self._db_url = db_url
        self._max_budget_usd = max_budget_usd
        # Preserve wrapped-function metadata
        self.__name__ = name
        self.__doc__ = fn.__doc__
        self._injected_checkpointer: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, *args: Any, thread_id: str | None = None, **kwargs: Any) -> WorkflowResult:
        """Run the workflow synchronously.

        Spawns a new event loop via :func:`asyncio.run`.  Safe to call from
        scripts and REPL; **not** safe to call from inside an async context —
        use :meth:`arun` instead.

        Args:
            *args:     Positional arguments forwarded to the workflow function.
            thread_id: Optional stable identifier for resumable runs.
            **kwargs:  Keyword arguments forwarded to the workflow function.

        Returns:
            :class:`~ninetrix.WorkflowResult`
        """
        return asyncio.run(self.arun(*args, thread_id=thread_id, **kwargs))

    async def arun(
        self, *args: Any, thread_id: str | None = None, **kwargs: Any
    ) -> WorkflowResult:
        """Run the workflow asynchronously.

        Args:
            *args:     Positional arguments forwarded to the workflow function.
            thread_id: Optional stable identifier for resumable runs.
            **kwargs:  Keyword arguments forwarded to the workflow function.

        Returns:
            :class:`~ninetrix.WorkflowResult`
        """
        thread_id = thread_id or uuid.uuid4().hex[:16]
        t0 = time.monotonic()

        # ── Durable mode setup ────────────────────────────────────────
        checkpointer = None
        cached_steps: dict = {}
        if self._durable:
            checkpointer = self._injected_checkpointer or await self._build_checkpointer()
            if checkpointer is not None:
                cached_steps = await checkpointer.get_completed_steps(thread_id)

        ctx = WorkflowContext(
            thread_id=thread_id,
            durable=self._durable,
            max_budget_usd=self._max_budget_usd,
            checkpointer=checkpointer,
            cached_steps=cached_steps,
        )

        token = _current_ctx.set(ctx)
        try:
            output = await self._fn(*args, **kwargs)
        finally:
            _current_ctx.reset(token)
            if checkpointer is not None and self._injected_checkpointer is None:
                try:
                    await checkpointer.disconnect()
                except Exception:
                    pass

        # Step results are plain values (not AgentResult) in durable mode
        total_tokens = 0
        total_cost = 0.0
        budget_remaining = (
            ctx._budget.remaining
            if self._max_budget_usd > 0.0
            else 0.0
        )

        return WorkflowResult(
            output=output,
            thread_id=thread_id,
            step_results=ctx._step_results,
            tokens_used=total_tokens,
            cost_usd=total_cost,
            elapsed_seconds=time.monotonic() - t0,
            completed_steps=list(ctx._completed_steps),
            skipped_steps=list(ctx._skipped_steps),
            budget_remaining_usd=budget_remaining,
            budget_limit_usd=self._max_budget_usd,
        )

    # ------------------------------------------------------------------
    # Checkpointer factory
    # ------------------------------------------------------------------

    async def _build_checkpointer(self) -> Any:
        """Build a checkpointer for durable mode.

        Resolution order:
        1. ``db_url`` kwarg on the decorator → PostgresCheckpointer
        2. ``DATABASE_URL`` env var → PostgresCheckpointer
        3. Fallback → InMemoryCheckpointer (useful in tests)
        """
        import os
        db_url = self._db_url or os.environ.get("DATABASE_URL", "")
        if db_url:
            try:
                from ninetrix.checkpoint.postgres import PostgresCheckpointer
                cp = PostgresCheckpointer(db_url)
                await cp.connect()
                return cp
            except ImportError:
                pass  # asyncpg not installed — fall through to in-memory
        from ninetrix.checkpoint.memory import InMemoryCheckpointer
        return InMemoryCheckpointer()

    def inject_checkpointer(self, checkpointer: Any) -> None:
        """Inject a pre-built checkpointer (used in tests to skip DB setup)."""
        self._injected_checkpointer = checkpointer

    def __repr__(self) -> str:  # pragma: no cover
        mode = "durable" if self._durable else "direct"
        return f"<WorkflowRunner name={self._name!r} mode={mode}>"


# ---------------------------------------------------------------------------
# Workflow decorator + proxy namespace
# ---------------------------------------------------------------------------


class _WorkflowDecorator:
    """Callable class that acts as both the ``@Workflow`` decorator and the
    namespace for workflow helper class methods (``Workflow.parallel``, etc.).

    The helpers delegate to the :class:`WorkflowContext` stored in the current
    :class:`~contextvars.ContextVar`.
    """

    def __call__(
        self,
        fn: Callable | None = None,
        *,
        durable: bool = False,
        name: str | None = None,
        db_url: str | None = None,
        max_budget: float = 0.0,
    ) -> Any:
        """Decorate an async function as a Workflow.

        Can be used with or without arguments::

            @Workflow
            async def simple(q: str) -> str: ...

            @Workflow(durable=True, max_budget=2.00)
            async def advanced(q: str) -> str: ...
        """
        def _make_runner(f: Callable) -> WorkflowRunner:
            return WorkflowRunner(
                fn=f,
                durable=durable,
                name=name or f.__name__,
                db_url=db_url,
                max_budget_usd=max_budget,
            )

        if fn is not None:
            # @Workflow used without parentheses
            return _make_runner(fn)
        # @Workflow(...) used with keyword arguments
        return _make_runner

    # ------------------------------------------------------------------
    # Proxy helpers — delegate to the active WorkflowContext
    # ------------------------------------------------------------------

    async def parallel(self, *coros: Any) -> list[Any]:
        """Run coroutines concurrently. Must be called inside a @Workflow function."""
        return await _get_ctx().parallel(*coros)

    async def fan_out(
        self,
        items: list[Any],
        agent_fn: Callable[[Any], Any],
        *,
        concurrency: int = 5,
    ) -> list[Any]:
        """Fan-out *agent_fn* over *items* with concurrency limit."""
        return await _get_ctx().fan_out(items, agent_fn, concurrency=concurrency)

    async def reduce(
        self,
        items: list[AgentResult],
        agent: Any,
        *,
        separator: str = "\n\n---\n\n",
    ) -> AgentResult:
        """Reduce a list of AgentResults to one using *agent*."""
        return await _get_ctx().reduce(items, agent, separator=separator)

    async def branch(
        self,
        condition: bool,
        *,
        if_true: Callable[[], Any],
        if_false: Callable[[], Any],
    ) -> Any:
        """Conditional branch between two async callables."""
        return await _get_ctx().branch(condition, if_true=if_true, if_false=if_false)

    def step(
        self,
        name: str,
        *,
        requires_approval: bool = False,
        timeout: float | None = None,
    ):
        """Mark an explicit step boundary (no-op in non-durable mode)."""
        return _get_ctx().step(
            name, requires_approval=requires_approval, timeout=timeout
        )


#: The ``@Workflow`` decorator.  Also exposes ``Workflow.parallel()``,
#: ``Workflow.fan_out()``, ``Workflow.reduce()``, ``Workflow.branch()``,
#: and ``Workflow.step()``.
Workflow = _WorkflowDecorator()
