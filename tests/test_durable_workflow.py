"""Tests for durable=True workflow — PR 29."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ninetrix import Workflow
from ninetrix.checkpoint.memory import InMemoryCheckpointer
from ninetrix.workflow.context import WorkflowContext, _StepResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_checkpointer(**preloaded_steps) -> InMemoryCheckpointer:
    """Return an InMemoryCheckpointer with pre-populated step results."""
    cp = InMemoryCheckpointer()
    # Directly inject step data so get_completed_steps() returns them
    for step_name, result in preloaded_steps.items():
        cp._steps["resume-thread"] = cp._steps.get("resume-thread", {})
        cp._steps["resume-thread"][step_name] = {
            "result": result,
            "step_index": 0,
            "status": "completed",
            "saved_at": 0.0,
        }
    return cp


# =============================================================================
# _StepResult
# =============================================================================


def test_step_result_fresh():
    sr = _StepResult()
    assert not sr.is_cached
    assert sr.value is None


def test_step_result_cached():
    sr = _StepResult(cached_value="cached_data", is_cached=True)
    assert sr.is_cached
    assert sr.value == "cached_data"


def test_step_result_set():
    sr = _StepResult()
    sr.set("new_value")
    assert sr.value == "new_value"


def test_step_result_set_overrides_cache():
    sr = _StepResult(cached_value="old", is_cached=True)
    sr.set("new")
    assert sr.value == "new"


# =============================================================================
# WorkflowContext.step() — non-durable
# =============================================================================


@pytest.mark.asyncio
async def test_step_yields_step_result():
    ctx = WorkflowContext("t1")
    async with ctx.step("s1") as sr:
        assert isinstance(sr, _StepResult)


@pytest.mark.asyncio
async def test_step_not_cached_on_fresh_run():
    ctx = WorkflowContext("t1")
    async with ctx.step("research") as sr:
        assert not sr.is_cached
        sr.set("result_data")
    assert "research" in ctx._completed_steps


@pytest.mark.asyncio
async def test_step_records_completed_non_durable():
    ctx = WorkflowContext("t1", durable=False)
    async with ctx.step("step_a"):
        pass
    async with ctx.step("step_b"):
        pass
    assert "step_a" in ctx._completed_steps
    assert "step_b" in ctx._completed_steps


# =============================================================================
# WorkflowContext.step() — durable, fresh run
# =============================================================================


@pytest.mark.asyncio
async def test_step_durable_fresh_saves_to_checkpointer():
    cp = InMemoryCheckpointer()
    ctx = WorkflowContext("t1", durable=True, checkpointer=cp)
    async with ctx.step("research") as sr:
        sr.set("research result")

    saved = await cp.get_completed_steps("t1")
    assert "research" in saved
    assert saved["research"] == "research result"


@pytest.mark.asyncio
async def test_step_durable_fresh_not_cached():
    cp = InMemoryCheckpointer()
    ctx = WorkflowContext("t1", durable=True, checkpointer=cp)
    async with ctx.step("fetch") as sr:
        assert not sr.is_cached


@pytest.mark.asyncio
async def test_step_durable_fresh_appears_in_completed():
    cp = InMemoryCheckpointer()
    ctx = WorkflowContext("t1", durable=True, checkpointer=cp)
    async with ctx.step("step1"):
        pass
    assert "step1" in ctx._completed_steps
    assert "step1" not in ctx._skipped_steps


# =============================================================================
# WorkflowContext.step() — durable, resume
# =============================================================================


@pytest.mark.asyncio
async def test_step_durable_resume_is_cached():
    """On resume, a completed step should be marked as cached."""
    cp = _make_checkpointer(research="cached_research_result")
    cached = await cp.get_completed_steps("resume-thread")
    ctx = WorkflowContext(
        "resume-thread", durable=True, checkpointer=cp, cached_steps=cached
    )
    async with ctx.step("research") as sr:
        assert sr.is_cached
        assert sr.value == "cached_research_result"


@pytest.mark.asyncio
async def test_step_durable_resume_appears_in_skipped():
    cp = _make_checkpointer(research="result")
    cached = await cp.get_completed_steps("resume-thread")
    ctx = WorkflowContext(
        "resume-thread", durable=True, checkpointer=cp, cached_steps=cached
    )
    async with ctx.step("research"):
        pass
    assert "research" in ctx._skipped_steps
    assert "research" not in ctx._completed_steps


@pytest.mark.asyncio
async def test_step_durable_resume_does_not_resave():
    """A skipped step should NOT call save_step() again."""
    cp = _make_checkpointer(research="old_result")
    cached = await cp.get_completed_steps("resume-thread")
    ctx = WorkflowContext(
        "resume-thread", durable=True, checkpointer=cp, cached_steps=cached
    )
    initial_count = cp.step_count("resume-thread")
    async with ctx.step("research"):
        pass
    assert cp.step_count("resume-thread") == initial_count  # no new write


@pytest.mark.asyncio
async def test_step_durable_partial_resume():
    """Fresh steps run, completed steps are skipped."""
    cp = _make_checkpointer(step1="result1")
    cached = await cp.get_completed_steps("resume-thread")
    ctx = WorkflowContext(
        "resume-thread", durable=True, checkpointer=cp, cached_steps=cached
    )

    executed = []

    async with ctx.step("step1") as s1:
        executed.append("step1_ran")  # body still runs even for cached steps!
        assert s1.is_cached

    async with ctx.step("step2") as s2:
        executed.append("step2_ran")
        assert not s2.is_cached
        s2.set("result2")

    # step2 should now be saved
    saved = await cp.get_completed_steps("resume-thread")
    assert "step2" in saved

    assert "step2_ran" in executed


# =============================================================================
# requires_approval
# =============================================================================


@pytest.mark.asyncio
async def test_step_requires_approval_saves_pending():
    cp = InMemoryCheckpointer()
    ctx = WorkflowContext("t1", durable=True, checkpointer=cp)
    async with ctx.step("human_review", requires_approval=True) as sr:
        sr.set("review data")

    # The step should be saved with pending_approval status
    steps_raw = cp._steps.get("t1", {})
    assert "human_review" in steps_raw
    assert steps_raw["human_review"]["status"] == "pending_approval"


# =============================================================================
# WorkflowRunner — durable=True integration
# =============================================================================


@pytest.mark.asyncio
async def test_durable_workflow_fresh_run_saves_steps():
    cp = InMemoryCheckpointer()

    @Workflow(durable=True)
    async def pipeline(msg: str) -> str:
        async with Workflow.step("step1") as s:
            s.set(f"processed:{msg}")
        return "done"

    pipeline.inject_checkpointer(cp)
    result = await pipeline.arun("hello", thread_id="t1")

    assert result.output == "done"
    saved = await cp.get_completed_steps("t1")
    assert "step1" in saved
    assert saved["step1"] == "processed:hello"


@pytest.mark.asyncio
async def test_durable_workflow_resume_shows_skipped():
    cp = _make_checkpointer(research="old_data")
    cached = await cp.get_completed_steps("resume-thread")

    calls = []

    @Workflow(durable=True)
    async def pipeline() -> str:
        async with Workflow.step("research") as s:
            if not s.is_cached:
                calls.append("agent_called")
                s.set("new_data")
        async with Workflow.step("write") as s:
            s.set("written")
        return "done"

    pipeline.inject_checkpointer(cp)
    result = await pipeline.arun(thread_id="resume-thread")

    assert "agent_called" not in calls  # agent was NOT called (step cached)
    assert "research" in result.skipped_steps
    assert "write" in result.completed_steps


@pytest.mark.asyncio
async def test_durable_workflow_inject_checkpointer():
    cp = InMemoryCheckpointer()

    @Workflow(durable=True)
    async def wf() -> str:
        return "ok"

    wf.inject_checkpointer(cp)
    assert wf._injected_checkpointer is cp


@pytest.mark.asyncio
async def test_durable_workflow_result_has_step_info():
    cp = InMemoryCheckpointer()

    @Workflow(durable=True)
    async def wf() -> str:
        async with Workflow.step("a"):
            pass
        async with Workflow.step("b"):
            pass
        return "done"

    wf.inject_checkpointer(cp)
    result = await wf.arun(thread_id="wf-t1")
    assert set(result.completed_steps) == {"a", "b"}
    assert result.skipped_steps == []


@pytest.mark.asyncio
async def test_durable_workflow_non_durable_default():
    """Without durable=True, step always runs and no checkpointer is used."""
    @Workflow
    async def wf() -> str:
        async with Workflow.step("s") as sr:
            assert not sr.is_cached
        return "ok"

    result = await wf.arun()
    assert result.output == "ok"
    assert "s" in result.completed_steps


# =============================================================================
# Public: _StepResult importable
# =============================================================================


def test_step_result_importable():
    from ninetrix.workflow.context import _StepResult
    assert _StepResult is not None
