"""
ninetrix/context.py — Ninetrix factory context.

Layer: L8 — may import L8 (agent, workflow) and below.

The ``Ninetrix`` class is a lightweight factory that holds default provider,
model, and checkpointer settings.  All agents, teams, and workflows created
through it inherit these defaults — eliminating the need to repeat
``provider="anthropic"`` on every single constructor call.

Usage::

    from ninetrix import Ninetrix, Workflow

    nx = Ninetrix(provider="anthropic", model="claude-haiku-4-5-20251001")

    support = nx.team("support", [
        nx.agent("billing",      description="Handles invoices and payments"),
        nx.agent("tech-support", description="Debugs API errors"),
        nx.agent("general",      description="General product questions"),
    ])

    result = await support.arun("I was charged twice this month.")
    print(result.agent_name)   # "billing"
    print(result.output)

    @nx.workflow(durable=True)
    async def pipeline(query: str) -> str:
        r = await Workflow.run_step("research", lambda: analyst.arun(query))
        return r.output

    result = await pipeline.arun(query, thread_id="run-001")
"""

from __future__ import annotations

from typing import Any


class Ninetrix:
    """Factory context for building agents, teams, and workflows with shared defaults.

    Args:
        provider:     Default LLM provider for all agents and the team router.
                      One of ``"anthropic"``, ``"openai"``, ``"google"``, ``"litellm"``.
        model:        Default model name.
        checkpointer: Default checkpointer injected into durable workflows.
                      When ``None``, durable workflows fall back to
                      ``InMemoryCheckpointer`` (or ``DATABASE_URL`` env var).
        budget_usd:   Per-agent spending cap in USD (``0.0`` = unlimited).
        max_turns:    Default maximum LLM turns per agent run.

    Example::

        nx = Ninetrix(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            checkpointer=InMemoryCheckpointer(),
        )

        analyst = nx.agent("analyst", description="Answers questions about stocks")
        result  = await analyst.arun("What is AAPL's market cap?")
    """

    def __init__(
        self,
        *,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-6",
        checkpointer: Any = None,
        budget_usd: float = 0.0,
        max_turns: int = 20,
    ) -> None:
        self.provider = provider
        self.model = model
        self._checkpointer = checkpointer
        self._budget_usd = budget_usd
        self._max_turns = max_turns

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    def agent(
        self,
        name: str,
        *,
        description: str = "",
        role: str = "",
        model: str | None = None,
        tools: list | None = None,
        budget_usd: float | None = None,
        max_turns: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create an :class:`~ninetrix.Agent` with this context's defaults.

        Args:
            name:        Agent name (used for routing, logging, and export).
            description: What this agent handles.  When ``role`` is absent,
                         a minimal system prompt is derived automatically.
                         Also used by :class:`~ninetrix.Team` for routing.
            role:        Explicit system-prompt role.  Overrides description
                         for the system prompt; description is still used for
                         routing.
            model:       Override the context's default model for this agent.
            tools:       List of ``@Tool`` functions or :class:`~ninetrix.Toolkit`.
            budget_usd:  Per-run budget cap.  Defaults to context's ``budget_usd``.
            max_turns:   Max LLM turns.  Defaults to context's ``max_turns``.
            **kwargs:    Any other :class:`~ninetrix.Agent` constructor kwargs.

        Returns:
            :class:`~ninetrix.Agent`

        Example::

            analyst = nx.agent(
                "analyst",
                description="Answers questions about equities",
                tools=[get_price],
            )
            result = await analyst.arun("What is Apple's stock price?")
        """
        from ninetrix.agent.agent import Agent

        return Agent(
            name=name,
            provider=self.provider,
            model=model or self.model,
            description=description,
            role=role,
            tools=tools or [],
            max_budget_usd=budget_usd if budget_usd is not None else self._budget_usd,
            max_turns=max_turns if max_turns is not None else self._max_turns,
            **kwargs,
        )

    def team(
        self,
        name: str,
        agents: list,
        *,
        router_provider: str | None = None,
        router_model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create a :class:`~ninetrix.Team` that uses this context's provider as router.

        The team router inherits the context's ``provider`` and ``model`` so you
        never need to specify ``router_provider`` or ``router_model`` explicitly.

        Args:
            name:            Team name.
            agents:          List of agents (:class:`~ninetrix.AgentProtocol`).
            router_provider: Override the router provider (defaults to context's).
            router_model:    Override the router model (defaults to context's).
            **kwargs:        Any other :class:`~ninetrix.Team` constructor kwargs.

        Returns:
            :class:`~ninetrix.Team`

        Example::

            support = nx.team("support", [
                nx.agent("billing",      description="Handles invoices"),
                nx.agent("tech-support", description="Debugs API errors"),
                nx.agent("general",      description="General questions"),
            ])
            result = await support.arun("I was charged twice")
            print(result.agent_name)  # "billing"
        """
        from ninetrix.workflow.team import Team

        return Team(
            name=name,
            agents=agents,
            router_provider=router_provider or self.provider,
            router_model=router_model or self.model,
            **kwargs,
        )

    def workflow(
        self,
        fn: Any = None,
        *,
        durable: bool = False,
        max_budget: float = 0.0,
        name: str | None = None,
        db_url: str | None = None,
    ) -> Any:
        """Decorator that creates a :class:`~ninetrix.WorkflowRunner` with context defaults.

        Identical to ``@Workflow(...)`` but automatically injects the context's
        checkpointer into the runner when ``durable=True``.

        Can be used with or without parentheses::

            @nx.workflow
            async def simple(q: str) -> str:
                return await Workflow.run_step("answer", lambda: agent.arun(q))

            @nx.workflow(durable=True, max_budget=0.50)
            async def advanced(q: str) -> str:
                ...

        Args:
            fn:          The async function (bare usage without parens).
            durable:     Enable step-level checkpointing and resume.
            max_budget:  Hard USD spending cap for the whole run.
            name:        Override the workflow name (defaults to function name).
            db_url:      PostgreSQL URL for the checkpointer.  When set, overrides
                         the context's checkpointer with a new Postgres instance.

        Returns:
            :class:`~ninetrix.WorkflowRunner`
        """
        from ninetrix.workflow.workflow import WorkflowRunner

        def _make_runner(f: Any) -> WorkflowRunner:
            runner = WorkflowRunner(
                fn=f,
                durable=durable,
                name=name or f.__name__,
                db_url=db_url,
                max_budget_usd=max_budget or self._budget_usd,
            )
            # Inject context checkpointer so callers skip inject_checkpointer()
            if self._checkpointer is not None:
                runner.inject_checkpointer(self._checkpointer)
            return runner

        if fn is not None:
            # @nx.workflow used without parentheses
            return _make_runner(fn)
        # @nx.workflow(...) used with keyword arguments
        return _make_runner

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Ninetrix(provider={self.provider!r}, model={self.model!r}, "
            f"checkpointer={self._checkpointer!r})"
        )
