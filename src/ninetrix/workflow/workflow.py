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
from contextlib import AbstractAsyncContextManager
from typing import Any, Callable, NoReturn, TypeVar, overload

from ninetrix._internals.types import AgentResult, WorkflowResult
from ninetrix._internals.trace import get_reporter, get_current_trace, reporter_scope
from ninetrix.workflow.context import WorkflowContext, _StepResult

T = TypeVar("T")


from contextlib import asynccontextmanager
from typing import AsyncIterator

@asynccontextmanager
async def _null_scope() -> AsyncIterator[None]:
    """No-op async context manager — used when reporter is None."""
    yield


# ---------------------------------------------------------------------------
# WorkflowTerminated — raised by Workflow.terminate(), caught by WorkflowRunner
# ---------------------------------------------------------------------------

class WorkflowTerminated(Exception):
    """Raised by :func:`Workflow.terminate` to signal early workflow exit.

    Never catch this in user code — the :class:`WorkflowRunner` handles it
    and sets ``WorkflowResult.terminated = True``.
    """

    def __init__(self, reason: str = "") -> None:
        super().__init__(reason)
        self.reason = reason


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

        # ── Reporter setup ────────────────────────────────────────────
        workflow_trace_id = uuid.uuid4().hex[:16]
        reporter = get_reporter()
        if reporter is None:
            try:
                from ninetrix.observability.reporter import RunnerReporter
                reporter = RunnerReporter.resolve()
            except Exception:
                reporter = None

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
        terminated = False
        termination_reason = ""
        output: Any = None
        try:
            if reporter is not None:
                await reporter.on_workflow_start(
                    thread_id=thread_id,
                    trace_id=workflow_trace_id,
                    workflow_name=self._name,
                )
            async with reporter_scope(reporter, trace_id=workflow_trace_id) if reporter else _null_scope():
                output = await self._fn(*args, **kwargs)
        except WorkflowTerminated as exc:
            terminated = True
            termination_reason = exc.reason
        finally:
            _current_ctx.reset(token)
            if checkpointer is not None and self._injected_checkpointer is None:
                try:
                    await checkpointer.disconnect()
                except Exception:
                    pass

        if reporter is not None:
            await reporter.on_workflow_complete(
                thread_id=thread_id,
                trace_id=workflow_trace_id,
                completed_steps=list(ctx._completed_steps),
                skipped_steps=list(ctx._skipped_steps),
                terminated=terminated,
                reason=termination_reason,
            )

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
            terminated=terminated,
            termination_reason=termination_reason,
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

    # ------------------------------------------------------------------
    # Overloads so IDEs know the return type in both usage forms:
    #   @Workflow               → WorkflowRunner
    #   @Workflow(durable=True) → Callable[[Callable], WorkflowRunner]
    # ------------------------------------------------------------------

    @overload
    def __call__(self, fn: Callable) -> WorkflowRunner: ...

    @overload
    def __call__(
        self,
        fn: None = None,
        *,
        durable: bool = ...,
        name: str | None = ...,
        db_url: str | None = ...,
        max_budget: float = ...,
    ) -> Callable[[Callable], WorkflowRunner]: ...

    def __call__(
        self,
        fn: Callable | None = None,
        *,
        durable: bool = False,
        name: str | None = None,
        db_url: str | None = None,
        max_budget: float = 0.0,
    ) -> WorkflowRunner | Callable[[Callable], WorkflowRunner]:
        """Decorate an async function as a Workflow.

        Can be used with or without arguments::

            @Workflow
            async def simple(q: str) -> str: ...

            @Workflow(durable=True, max_budget=2.00)
            async def advanced(q: str) -> str: ...

        Args:
            fn:          The async function to wrap (only when used bare, without parens).
            durable:     Enable step-level checkpointing and resume.
            name:        Override the workflow name (defaults to the function name).
            db_url:      PostgreSQL URL for the checkpointer.  Falls back to
                         ``DATABASE_URL`` env var, then in-memory.
            max_budget:  Hard spending cap in USD for the whole run.  ``0.0`` = unlimited.

        Returns:
            :class:`WorkflowRunner` (bare usage) or a decorator (with-parens usage).
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

    async def run_step(self, name: str, fn: Callable[[], Any]) -> Any:
        """Execute *fn* and cache its result under *name*.

        On the first run, ``fn`` is awaited and its result is persisted (durable
        mode) or stored in memory (non-durable).  On subsequent runs with the
        same ``thread_id``, the cached result is returned immediately — ``fn``
        is **not** called again.

        Return type matches whatever ``fn()`` returns — fully generic.  When
        ``fn`` returns an :class:`~ninetrix.AgentResult`, the caller receives
        the full ``AgentResult`` and accesses ``.output`` explicitly::

            result = await Workflow.run_step("research", lambda: agent.arun(query))
            print(result.output)   # str or Pydantic model depending on output_type

        For plain-value lambdas the result is returned as-is::

            score = await Workflow.run_step("score", lambda: calculate(user_id))
            print(score + 10)

        Args:
            name: Unique step name within this workflow run.
            fn:   Zero-argument async callable (lambda or ``async def``).

        Returns:
            Whatever ``fn()`` returns, or the cached value from a prior run.
        """
        ctx = _get_ctx()
        reporter = get_reporter()
        trace_id = get_current_trace() or ""

        # Resume path: return cached result without calling fn
        is_cached = ctx._durable and name in ctx._cached_steps
        if is_cached:
            ctx._skipped_steps.append(name)
            if reporter is not None:
                await reporter.on_workflow_step_completed(
                    thread_id=ctx._thread_id,
                    trace_id=trace_id,
                    step_name=name,
                    cached=True,
                )
            return ctx._cached_steps[name]

        # Fresh path: execute fn, persist result
        if reporter is not None:
            await reporter.on_workflow_step_started(
                thread_id=ctx._thread_id,
                trace_id=trace_id,
                step_name=name,
            )

        ctx._step_index += 1
        result = await fn()
        ctx._completed_steps.append(name)
        ctx._step_results[name] = result

        if ctx._durable and ctx._checkpointer is not None:
            await ctx._checkpointer.save_step(
                thread_id=ctx._thread_id,
                step_name=name,
                step_index=ctx._step_index,
                result=result,
                status="completed",
            )

        if reporter is not None:
            await reporter.on_workflow_step_completed(
                thread_id=ctx._thread_id,
                trace_id=trace_id,
                step_name=name,
                cached=False,
            )

        return result

    def terminate(self, reason: str = "") -> NoReturn:
        """Abort the workflow early with an optional reason.

        The :class:`WorkflowRunner` catches the internal exception and returns
        a :class:`~ninetrix.WorkflowResult` with ``terminated=True`` and
        ``termination_reason`` set to *reason*.

        Use this to signal a **failure or guard condition** — semantically
        different from ``return "..."`` which signals a successful result::

            is_safe = await Workflow.run_step("check", lambda: moderator.arun(text))
            if is_safe.output.strip().upper() != "PASS":
                return Workflow.terminate("Failed safety check")

        Callers check the result::

            result = await pipeline.arun(text, thread_id="pub-001")
            if result.terminated:
                print(f"Blocked: {result.termination_reason}")
            else:
                print(result.output)

        Args:
            reason: Human-readable explanation of why the workflow was stopped.
        """
        raise WorkflowTerminated(reason)

    async def map(
        self,
        agent: Any,
        items: list[Any],
        *,
        prefix: str = "",
        concurrency: int = 5,
        step_prefix: str | None = None,
    ) -> list[Any]:
        """Fan-out *agent* over *items* with a concurrency limit.

        Each item is converted to a prompt string: ``f"{prefix}{item}"``.
        Results are returned in the same order as *items*.

        When *step_prefix* is set, each item becomes a named durable step
        (``{step_prefix}_0``, ``{step_prefix}_1``, …).  Crashed runs resume
        from the first un-cached item.

        Must be called inside a ``@Workflow``-decorated function.

        Args:
            agent:        Agent (or any :class:`~ninetrix.AgentProtocol`) to run.
            items:        List of input items.
            prefix:       String prepended to each item when building the prompt.
            concurrency:  Maximum simultaneous ``arun`` calls.
            step_prefix:  When set, makes each item a named durable step.

        Returns:
            ``list[AgentResult]`` in the same order as *items*.

        Example::

            results = await Workflow.map(
                researcher,
                topics,
                prefix="Research in depth: ",
                concurrency=3,
            )
            combined = "\\n\\n".join(r.output for r in results)
        """
        ctx = _get_ctx()  # enforce: must be called inside a @Workflow function
        reporter = get_reporter()
        trace_id = get_current_trace() or ""
        effective_prefix = step_prefix or ""

        if reporter is not None and effective_prefix:
            await reporter.on_workflow_map_started(
                thread_id=ctx._thread_id,
                trace_id=trace_id,
                step_prefix=effective_prefix,
                item_count=len(items),
            )

        semaphore = asyncio.Semaphore(concurrency)

        async def _run_one(i: int, item: Any) -> Any:
            prompt = f"{prefix}{item}" if prefix else str(item)
            async with semaphore:
                if step_prefix is not None:
                    step_name = f"{step_prefix}_{i}"
                    return await self.run_step(step_name, lambda: agent.arun(prompt))
                return await agent.arun(prompt)

        results = list(await asyncio.gather(*[_run_one(i, item) for i, item in enumerate(items)]))

        if reporter is not None and effective_prefix:
            cached_count = sum(
                1 for n in (f"{effective_prefix}_{i}" for i in range(len(items)))
                if n in ctx._skipped_steps
            )
            await reporter.on_workflow_map_completed(
                thread_id=ctx._thread_id,
                trace_id=trace_id,
                step_prefix=effective_prefix,
                completed=len(items) - cached_count,
                cached=cached_count,
            )

        return results

    def step(
        self,
        name: str,
        *,
        requires_approval: bool = False,
        timeout: float | None = None,
    ) -> AbstractAsyncContextManager[_StepResult]:
        """Mark an explicit step boundary.

        Yields a :class:`~ninetrix.workflow.context._StepResult` with:

        - ``step.is_cached`` — ``True`` on resume if this step already ran.
        - ``step.value``     — the stored result (cached or freshly set).
        - ``step.set(v)``    — persist *v* as this step's result.

        Example::

            async with Workflow.step("research") as step:
                if not step.is_cached:
                    result = await researcher.arun(topic)
                    step.set(result.output)
                data = step.value

        Args:
            name:              Unique step name within this workflow run.
            requires_approval: Save with ``pending_approval`` status (HITL gate).
            timeout:           Reserved for future per-step timeout support.
        """
        return _get_ctx().step(
            name, requires_approval=requires_approval, timeout=timeout
        )


#: The ``@Workflow`` decorator.  Also exposes ``Workflow.parallel()``,
#: ``Workflow.fan_out()``, ``Workflow.reduce()``, ``Workflow.branch()``,
#: and ``Workflow.step()``.
Workflow = _WorkflowDecorator()
