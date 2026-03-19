"""
tests/test_v2_api.py — Test suite for the v2 developer API.

Covers:
  - Ninetrix context factory: nx.agent / nx.team / nx.workflow
  - AgentConfig.system_prompt: auto-derive role from description
  - TeamResult.agent_name alias
  - WorkflowResult.terminated + termination_reason
  - Workflow.run_step: fresh run, resume from cache, plain-value lambda
  - Workflow.terminate: raises WorkflowTerminated, runner catches it
  - Workflow.map: fan-out with concurrency, step_prefix for durable caching
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from ninetrix import (
    Ninetrix,
    Workflow,
    WorkflowTerminated,
    InMemoryCheckpointer,
)
from ninetrix.agent.config import AgentConfig
from ninetrix.workflow.team import TeamResult
from ninetrix._internals.types import AgentResult, WorkflowResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent_result(output: str = "ok") -> AgentResult:
    return AgentResult(
        output=output,
        thread_id="t1",
        tokens_used=10,
        input_tokens=7,
        output_tokens=3,
        cost_usd=0.0001,
        steps=1,
    )


def _mock_agent(output: str = "ok") -> MagicMock:
    """Return an object that satisfies AgentProtocol duck-typing."""
    agent = MagicMock()
    agent.name = "mock-agent"
    agent.config = MagicMock()
    agent.config.name = "mock-agent"
    agent.config.description = "A mock agent"
    agent.arun = AsyncMock(return_value=_make_agent_result(output))
    return agent


# ===========================================================================
# AgentConfig.system_prompt — auto-derive role from description
# ===========================================================================

class TestSystemPromptDerivation:

    def test_role_set_uses_role(self):
        cfg = AgentConfig(role="senior data analyst", description="Handles queries")
        assert "senior data analyst" in cfg.system_prompt
        assert "You are a senior data analyst." in cfg.system_prompt

    def test_description_only_derives_prompt(self):
        cfg = AgentConfig(description="Handles invoices and payments")
        prompt = cfg.system_prompt
        assert "helpful assistant" in prompt
        assert "Handles invoices and payments" in prompt

    def test_neither_role_nor_description_gives_empty_prompt(self):
        cfg = AgentConfig()
        # No role, no description, no goal → empty prompt
        assert cfg.system_prompt == ""

    def test_role_takes_priority_over_description(self):
        cfg = AgentConfig(
            role="billing specialist",
            description="Handles invoices",
        )
        # role wins — description should not appear in role line
        assert "billing specialist" in cfg.system_prompt
        # But description itself should not double-appear as a second role
        prompt = cfg.system_prompt
        role_line_count = prompt.count("You are a")
        assert role_line_count == 1

    def test_description_as_system_prompt_is_complete_sentence(self):
        cfg = AgentConfig(description="Answers questions about AAPL stock")
        prompt = cfg.system_prompt
        # Should end with a period
        first_line = prompt.split("\n\n")[0]
        assert first_line.endswith(".")


# ===========================================================================
# TeamResult.agent_name alias
# ===========================================================================

class TestTeamResultAgentName:

    def test_agent_name_is_routed_to(self):
        result = TeamResult(
            output="Here is your invoice.",
            routed_to="billing",
            agent_result=_make_agent_result(),
        )
        assert result.agent_name == "billing"
        assert result.agent_name == result.routed_to

    def test_agent_name_reflects_routing(self):
        for name in ["billing", "tech-support", "general"]:
            r = TeamResult(output="", routed_to=name, agent_result=_make_agent_result())
            assert r.agent_name == name


# ===========================================================================
# WorkflowResult.terminated
# ===========================================================================

class TestWorkflowResultTerminated:

    def test_default_not_terminated(self):
        result = WorkflowResult(
            output="done",
            thread_id="t1",
            step_results={},
            tokens_used=0,
            cost_usd=0.0,
            elapsed_seconds=0.1,
        )
        assert result.terminated is False
        assert result.termination_reason == ""

    def test_terminated_fields(self):
        result = WorkflowResult(
            output=None,
            thread_id="t1",
            step_results={},
            tokens_used=0,
            cost_usd=0.0,
            elapsed_seconds=0.1,
            terminated=True,
            termination_reason="Content failed safety check",
        )
        assert result.terminated is True
        assert result.termination_reason == "Content failed safety check"


# ===========================================================================
# WorkflowTerminated exception
# ===========================================================================

class TestWorkflowTerminated:

    def test_is_exception(self):
        exc = WorkflowTerminated("test reason")
        assert isinstance(exc, Exception)
        assert exc.reason == "test reason"

    def test_empty_reason(self):
        exc = WorkflowTerminated()
        assert exc.reason == ""

    def test_terminate_method_raises(self):
        with pytest.raises(WorkflowTerminated) as exc_info:
            Workflow.terminate("Safety check failed")
        assert exc_info.value.reason == "Safety check failed"

    @pytest.mark.asyncio
    async def test_runner_catches_terminated(self):
        @Workflow
        async def guarded(text: str) -> str:
            if text == "bad":
                return Workflow.terminate("Rejected: bad content")
            return "ok"

        result = await guarded.arun("bad")
        assert result.terminated is True
        assert result.termination_reason == "Rejected: bad content"
        assert result.output is None

    @pytest.mark.asyncio
    async def test_runner_passes_through_normal_return(self):
        @Workflow
        async def normal(text: str) -> str:
            return f"processed: {text}"

        result = await normal.arun("hello")
        assert result.terminated is False
        assert result.output == "processed: hello"
        assert result.termination_reason == ""

    @pytest.mark.asyncio
    async def test_terminate_without_reason(self):
        @Workflow
        async def abort_empty() -> str:
            return Workflow.terminate()

        result = await abort_empty.arun()
        assert result.terminated is True
        assert result.termination_reason == ""


# ===========================================================================
# Workflow.run_step
# ===========================================================================

class TestWorkflowRunStep:

    @pytest.mark.asyncio
    async def test_fresh_run_calls_fn(self):
        calls = []

        @Workflow
        async def pipeline(x: int) -> int:
            result = await Workflow.run_step("double", lambda: asyncio.coroutine(lambda: x * 2)())
            return result

        # Use a simpler approach with a coroutine factory
        async def double(x):
            return x * 2

        @Workflow
        async def pipeline2(x: int) -> int:
            calls.append("called")
            result = await Workflow.run_step("double", lambda: double(x))
            return result

        r = await pipeline2.arun(5)
        assert r.output == 10
        assert calls == ["called"]

    @pytest.mark.asyncio
    async def test_durable_run_caches_step(self):
        cp = InMemoryCheckpointer()
        call_count = [0]

        async def expensive():
            call_count[0] += 1
            return "computed"

        @Workflow(durable=True)
        async def pipeline() -> str:
            return await Workflow.run_step("expensive", expensive)

        pipeline.inject_checkpointer(cp)

        # First run — executes
        r1 = await pipeline.arun(thread_id="t-cache-1")
        assert r1.output == "computed"
        assert call_count[0] == 1
        assert "expensive" in r1.completed_steps

        # Second run — step is cached, fn NOT called again
        r2 = await pipeline.arun(thread_id="t-cache-1")
        assert r2.output == "computed"
        assert call_count[0] == 1  # still 1 — not called again
        assert "expensive" in r2.skipped_steps

    @pytest.mark.asyncio
    async def test_plain_value_lambda(self):
        """run_step returns the raw value when fn returns a non-AgentResult."""

        async def compute():
            return 42

        @Workflow
        async def pipeline() -> int:
            return await Workflow.run_step("compute", compute)

        r = await pipeline.arun()
        assert r.output == 42

    @pytest.mark.asyncio
    async def test_agent_result_returned_as_is(self):
        """run_step returns full AgentResult — caller uses .output explicitly."""
        mock = _mock_agent("analysis complete")

        @Workflow
        async def pipeline(q: str) -> str:
            result = await Workflow.run_step("analyse", lambda: mock.arun(q))
            # result is AgentResult — .output accessed explicitly
            return result.output

        r = await pipeline.arun("What is AAPL?")
        assert r.output == "analysis complete"

    @pytest.mark.asyncio
    async def test_run_step_outside_workflow_raises(self):
        with pytest.raises(RuntimeError, match="Workflow helpers"):
            await Workflow.run_step("bad", lambda: asyncio.sleep(0))

    @pytest.mark.asyncio
    async def test_multiple_steps_sequence(self):
        cp = InMemoryCheckpointer()
        order = []

        async def step_a():
            order.append("a")
            return "A"

        async def step_b():
            order.append("b")
            return "B"

        @Workflow(durable=True)
        async def pipeline() -> str:
            a = await Workflow.run_step("step_a", step_a)
            b = await Workflow.run_step("step_b", step_b)
            return f"{a}-{b}"

        pipeline.inject_checkpointer(cp)

        r = await pipeline.arun(thread_id="seq-1")
        assert r.output == "A-B"
        assert order == ["a", "b"]
        assert r.completed_steps == ["step_a", "step_b"]


# ===========================================================================
# Workflow.map
# ===========================================================================

class TestWorkflowMap:

    @pytest.mark.asyncio
    async def test_map_calls_agent_for_each_item(self):
        call_args = []

        async def fake_arun(prompt, **kwargs):
            call_args.append(prompt)
            return _make_agent_result(f"result:{prompt}")

        agent = MagicMock()
        agent.arun = fake_arun

        @Workflow
        async def pipeline(items: list) -> list:
            return await Workflow.map(agent, items, prefix="Analyze: ")

        r = await pipeline.arun(["AAPL", "MSFT", "GOOG"])
        assert len(r.output) == 3
        assert all("result:Analyze: " in res.output for res in r.output)
        assert len(call_args) == 3

    @pytest.mark.asyncio
    async def test_map_preserves_order(self):
        async def arun_slow(prompt, **kwargs):
            # Simulate varying latency
            await asyncio.sleep(0.001 * (hash(prompt) % 5))
            return _make_agent_result(f"out:{prompt}")

        agent = MagicMock()
        agent.arun = arun_slow

        @Workflow
        async def pipeline(items: list) -> list:
            return await Workflow.map(agent, items)

        r = await pipeline.arun(["first", "second", "third"])
        outputs = [res.output for res in r.output]
        assert outputs == ["out:first", "out:second", "out:third"]

    @pytest.mark.asyncio
    async def test_map_concurrency_limit(self):
        active = [0]
        max_active = [0]

        async def arun_tracked(prompt, **kwargs):
            active[0] += 1
            max_active[0] = max(max_active[0], active[0])
            await asyncio.sleep(0.01)
            active[0] -= 1
            return _make_agent_result("ok")

        agent = MagicMock()
        agent.arun = arun_tracked

        @Workflow
        async def pipeline(items: list) -> list:
            return await Workflow.map(agent, items, concurrency=2)

        await pipeline.arun(list(range(8)))
        assert max_active[0] <= 2

    @pytest.mark.asyncio
    async def test_map_with_step_prefix_caches_items(self):
        cp = InMemoryCheckpointer()
        call_count = [0]

        async def arun_counted(prompt, **kwargs):
            call_count[0] += 1
            return _make_agent_result(f"result:{prompt}")

        agent = MagicMock()
        agent.arun = arun_counted

        @Workflow(durable=True)
        async def pipeline(items: list) -> list:
            return await Workflow.map(
                agent, items,
                prefix="Research: ",
                step_prefix="research",
            )

        pipeline.inject_checkpointer(cp)

        items = ["AAPL", "MSFT"]
        r1 = await pipeline.arun(items, thread_id="map-t1")
        assert call_count[0] == 2

        # Re-run — all items cached
        r2 = await pipeline.arun(items, thread_id="map-t1")
        assert call_count[0] == 2  # no new calls
        assert len(r2.skipped_steps) == 2

    @pytest.mark.asyncio
    async def test_map_outside_workflow_raises(self):
        agent = _mock_agent()
        with pytest.raises(RuntimeError, match="Workflow helpers"):
            await Workflow.map(agent, ["a", "b"])


# ===========================================================================
# Ninetrix context factory
# ===========================================================================

class TestNinetrixContext:

    def test_defaults(self):
        nx = Ninetrix(provider="anthropic", model="claude-haiku-4-5-20251001")
        assert nx.provider == "anthropic"
        assert nx.model == "claude-haiku-4-5-20251001"
        assert nx._checkpointer is None

    def test_agent_factory_sets_provider_and_model(self):
        nx = Ninetrix(provider="anthropic", model="claude-haiku-4-5-20251001")
        agent = nx.agent("billing", description="Handles invoices")
        assert agent.config.provider == "anthropic"
        assert agent.config.model == "claude-haiku-4-5-20251001"
        assert agent.config.name == "billing"
        assert agent.config.description == "Handles invoices"

    def test_agent_factory_model_override(self):
        nx = Ninetrix(provider="anthropic", model="claude-haiku-4-5-20251001")
        agent = nx.agent("specialist", description="Deep analysis", model="claude-sonnet-4-6")
        assert agent.config.model == "claude-sonnet-4-6"

    def test_agent_factory_role_set(self):
        nx = Ninetrix(provider="anthropic", model="claude-haiku-4-5-20251001")
        agent = nx.agent(
            "analyst",
            description="Handles equities",
            role="You are a senior equity research analyst with 20 years of experience.",
        )
        assert "senior equity research analyst" in agent.config.system_prompt

    def test_agent_factory_description_only_derives_prompt(self):
        nx = Ninetrix(provider="anthropic", model="claude-haiku-4-5-20251001")
        agent = nx.agent("billing", description="Handles invoices and refunds")
        assert "helpful assistant" in agent.config.system_prompt
        assert "Handles invoices and refunds" in agent.config.system_prompt

    def test_team_factory_sets_router(self):
        nx = Ninetrix(provider="anthropic", model="claude-haiku-4-5-20251001")
        agents = [
            nx.agent("billing", description="Billing"),
            nx.agent("tech", description="Tech support"),
        ]
        team = nx.team("support", agents)
        assert team.name == "support"
        assert team._router_provider == "anthropic"
        assert team._router_model == "claude-haiku-4-5-20251001"
        assert len(team._agents) == 2

    def test_team_factory_router_override(self):
        nx = Ninetrix(provider="anthropic", model="claude-haiku-4-5-20251001")
        agents = [nx.agent("a", description="A")]
        team = nx.team("t", agents, router_model="claude-sonnet-4-6")
        assert team._router_model == "claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_workflow_factory_no_parens(self):
        nx = Ninetrix(provider="anthropic", model="claude-haiku-4-5-20251001")

        @nx.workflow
        async def simple(x: int) -> int:
            return x * 2

        r = await simple.arun(5)
        assert r.output == 10

    @pytest.mark.asyncio
    async def test_workflow_factory_with_parens(self):
        nx = Ninetrix(provider="anthropic", model="claude-haiku-4-5-20251001")

        @nx.workflow(durable=True)
        async def durable_wf(x: int) -> int:
            return await Workflow.run_step("double", lambda: asyncio.coroutine(lambda: x * 2)())

        async def double(x):
            return x * 2

        @nx.workflow(durable=True)
        async def durable_wf2(x: int) -> int:
            return await Workflow.run_step("double", lambda: double(x))

        r = await durable_wf2.arun(5)
        assert r.output == 10

    @pytest.mark.asyncio
    async def test_workflow_factory_injects_checkpointer(self):
        cp = InMemoryCheckpointer()
        nx = Ninetrix(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            checkpointer=cp,
        )

        call_count = [0]

        async def work():
            call_count[0] += 1
            return "done"

        @nx.workflow(durable=True)
        async def pipeline() -> str:
            return await Workflow.run_step("work", work)

        r1 = await pipeline.arun(thread_id="nx-inject-1")
        assert r1.output == "done"
        assert call_count[0] == 1

        # Same thread — cached because checkpointer was injected from context
        r2 = await pipeline.arun(thread_id="nx-inject-1")
        assert r2.output == "done"
        assert call_count[0] == 1  # not called again

    def test_context_repr(self):
        nx = Ninetrix(provider="openai", model="gpt-4o")
        r = repr(nx)
        assert "openai" in r
        assert "gpt-4o" in r


# ===========================================================================
# Integration: Ninetrix + run_step + terminate together
# ===========================================================================

class TestV2Integration:

    @pytest.mark.asyncio
    async def test_router_pattern(self):
        """Classify result drives conditional routing."""
        cp = InMemoryCheckpointer()
        nx = Ninetrix(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            checkpointer=cp,
        )

        billing_agent = nx.agent("billing", description="Handles billing")
        tech_agent = nx.agent("tech", description="Handles tech")

        async def classify(query: str) -> str:
            return "billing" if "charged" in query.lower() else "technical"

        async def handle_billing(q: str) -> str:
            return f"billing response: {q}"

        async def handle_tech(q: str) -> str:
            return f"tech response: {q}"

        @nx.workflow(durable=True)
        async def smart_pipeline(query: str) -> str:
            category = await Workflow.run_step(
                "classify",
                lambda: classify(query)
            )
            if category == "billing":
                return await Workflow.run_step(
                    "billing_fix",
                    lambda: handle_billing(query)
                )
            return await Workflow.run_step(
                "tech_fix",
                lambda: handle_tech(query)
            )

        r1 = await smart_pipeline.arun("I was charged twice", thread_id="route-1")
        assert r1.output == "billing response: I was charged twice"
        assert "classify" in r1.completed_steps
        assert "billing_fix" in r1.completed_steps

        r2 = await smart_pipeline.arun("API returns 500 error", thread_id="route-2")
        assert r2.output == "tech response: API returns 500 error"
        assert "tech_fix" in r2.completed_steps

    @pytest.mark.asyncio
    async def test_guard_pattern(self):
        """Guard step blocks on failure, passes on success."""
        async def moderate(text: str) -> str:
            return "FAIL" if "banned" in text else "PASS"

        async def publish(text: str) -> str:
            return f"published: {text}"

        @Workflow
        async def publication_workflow(article: str) -> str:
            verdict = await Workflow.run_step("safety_check", lambda: moderate(article))
            if verdict.strip().upper() != "PASS":
                return Workflow.terminate("Content failed safety check")
            return await Workflow.run_step("publish", lambda: publish(article))

        # Pass case
        r_pass = await publication_workflow.arun("Great article about Python")
        assert r_pass.terminated is False
        assert r_pass.output == "published: Great article about Python"

        # Fail case
        r_fail = await publication_workflow.arun("This is banned content")
        assert r_fail.terminated is True
        assert "safety check" in r_fail.termination_reason.lower()
        assert r_fail.output is None

    @pytest.mark.asyncio
    async def test_retry_loop_pattern(self):
        """Dynamic loop with per-attempt step names."""
        cp = InMemoryCheckpointer()
        attempt_results = ["fail", "fail", "pass"]
        attempt_idx = [0]

        async def run_test(code: str) -> dict:
            idx = attempt_idx[0]
            attempt_idx[0] += 1
            passed = attempt_results[idx] == "pass"
            return {"passed": passed, "error": "" if passed else f"error at attempt {idx}"}

        async def write_code(task: str) -> str:
            return f"def solution(): pass  # {task}"

        async def fix_code(code: str, error: str) -> str:
            return f"{code}  # fixed: {error}"

        @Workflow(durable=True)
        async def coding_pipeline(task: str) -> str:
            code = await Workflow.run_step("write_code", lambda: write_code(task))

            for attempt in range(3):
                test_result = await Workflow.run_step(
                    f"test_{attempt}",
                    lambda: run_test(code)
                )
                if test_result["passed"]:
                    return code
                code = await Workflow.run_step(
                    f"fix_{attempt}",
                    lambda: fix_code(code, test_result["error"])
                )

            return Workflow.terminate("Failed after 3 attempts")

        coding_pipeline.inject_checkpointer(cp)

        r = await coding_pipeline.arun("sort a list", thread_id="code-1")
        assert r.terminated is False
        assert "pass" in r.output or "fixed" in r.output
        # test_0 and test_1 failed, test_2 passed
        assert "test_0" in r.completed_steps
        assert "test_1" in r.completed_steps
        assert "test_2" in r.completed_steps
