"""Tests for checkpoint/base.py + checkpoint/memory.py — PR 17."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ninetrix.checkpoint.base import Checkpointer
from ninetrix.checkpoint.memory import InMemoryCheckpointer


# =============================================================================
# Checkpointer ABC enforcement
# =============================================================================


def test_checkpointer_is_abstract():
    """Cannot instantiate Checkpointer directly."""
    with pytest.raises(TypeError):
        Checkpointer()  # type: ignore[abstract]


def test_checkpointer_requires_all_methods():
    """Partial implementations raise TypeError at instantiation."""

    class Partial(Checkpointer):
        async def save(self, *, thread_id, agent_id, history, tokens_used,
                       status="completed", metadata=None):
            pass

        # Missing load, delete, save_step, get_completed_steps

    with pytest.raises(TypeError):
        Partial()  # type: ignore[abstract]


def test_inmemory_is_checkpointer_subclass():
    assert issubclass(InMemoryCheckpointer, Checkpointer)


# =============================================================================
# InMemoryCheckpointer — save + load
# =============================================================================


@pytest.mark.asyncio
async def test_save_and_load():
    cp = InMemoryCheckpointer()
    history = [{"role": "user", "content": "hello"}]
    await cp.save(thread_id="t1", agent_id="bot", history=history, tokens_used=42)

    result = await cp.load("t1")
    assert result is not None
    assert result["thread_id"] == "t1"
    assert result["agent_id"] == "bot"
    assert result["tokens_used"] == 42
    assert result["status"] == "completed"
    assert result["history"] == history


@pytest.mark.asyncio
async def test_load_missing_returns_none():
    cp = InMemoryCheckpointer()
    assert await cp.load("nonexistent") is None


@pytest.mark.asyncio
async def test_save_overwrites_previous():
    cp = InMemoryCheckpointer()
    h1 = [{"role": "user", "content": "first"}]
    h2 = [{"role": "user", "content": "second"}]

    await cp.save(thread_id="t1", agent_id="bot", history=h1, tokens_used=10)
    await cp.save(thread_id="t1", agent_id="bot", history=h2, tokens_used=20)

    result = await cp.load("t1")
    assert result["history"] == h2
    assert result["tokens_used"] == 20


@pytest.mark.asyncio
async def test_save_deep_copies_history():
    """Mutating the original list after save should not affect the checkpoint."""
    cp = InMemoryCheckpointer()
    history = [{"role": "user", "content": "hello"}]
    await cp.save(thread_id="t1", agent_id="bot", history=history, tokens_used=5)
    history.append({"role": "assistant", "content": "world"})

    result = await cp.load("t1")
    assert len(result["history"]) == 1


@pytest.mark.asyncio
async def test_load_deep_copies_checkpoint():
    """Mutating the returned checkpoint should not affect stored state."""
    cp = InMemoryCheckpointer()
    await cp.save(thread_id="t1", agent_id="bot",
                  history=[{"role": "user", "content": "hi"}], tokens_used=5)

    r1 = await cp.load("t1")
    r1["history"].append({"role": "assistant", "content": "added"})

    r2 = await cp.load("t1")
    assert len(r2["history"]) == 1


@pytest.mark.asyncio
async def test_save_with_custom_status():
    cp = InMemoryCheckpointer()
    await cp.save(thread_id="t1", agent_id="bot", history=[],
                  tokens_used=0, status="error")
    result = await cp.load("t1")
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_save_with_metadata():
    cp = InMemoryCheckpointer()
    meta = {"run_id": "abc123", "trigger": "webhook"}
    await cp.save(thread_id="t1", agent_id="bot", history=[],
                  tokens_used=0, metadata=meta)
    result = await cp.load("t1")
    assert result["metadata"] == meta


@pytest.mark.asyncio
async def test_save_metadata_deep_copied():
    cp = InMemoryCheckpointer()
    meta = {"key": "value"}
    await cp.save(thread_id="t1", agent_id="bot", history=[],
                  tokens_used=0, metadata=meta)
    meta["key"] = "mutated"
    result = await cp.load("t1")
    assert result["metadata"]["key"] == "value"


@pytest.mark.asyncio
async def test_multiple_threads_independent():
    cp = InMemoryCheckpointer()
    await cp.save(thread_id="a", agent_id="bot", history=[{"role": "user", "content": "a"}], tokens_used=1)
    await cp.save(thread_id="b", agent_id="bot", history=[{"role": "user", "content": "b"}], tokens_used=2)

    ra = await cp.load("a")
    rb = await cp.load("b")
    assert ra["history"][0]["content"] == "a"
    assert rb["history"][0]["content"] == "b"


# =============================================================================
# InMemoryCheckpointer — delete
# =============================================================================


@pytest.mark.asyncio
async def test_delete_removes_checkpoint():
    cp = InMemoryCheckpointer()
    await cp.save(thread_id="t1", agent_id="bot", history=[], tokens_used=0)
    await cp.delete("t1")
    assert await cp.load("t1") is None


@pytest.mark.asyncio
async def test_delete_nonexistent_is_noop():
    cp = InMemoryCheckpointer()
    await cp.delete("does_not_exist")  # should not raise


@pytest.mark.asyncio
async def test_delete_removes_steps_too():
    cp = InMemoryCheckpointer()
    await cp.save(thread_id="t1", agent_id="bot", history=[], tokens_used=0)
    await cp.save_step(thread_id="t1", step_name="step1", step_index=0, result="done")
    await cp.delete("t1")

    completed = await cp.get_completed_steps("t1")
    assert completed == {}


# =============================================================================
# InMemoryCheckpointer — workflow steps
# =============================================================================


@pytest.mark.asyncio
async def test_save_and_get_step():
    cp = InMemoryCheckpointer()
    await cp.save_step(thread_id="wf1", step_name="research", step_index=0, result="found data")
    steps = await cp.get_completed_steps("wf1")
    assert steps == {"research": "found data"}


@pytest.mark.asyncio
async def test_get_completed_steps_empty():
    cp = InMemoryCheckpointer()
    steps = await cp.get_completed_steps("unknown")
    assert steps == {}


@pytest.mark.asyncio
async def test_only_completed_steps_returned():
    cp = InMemoryCheckpointer()
    await cp.save_step(thread_id="wf1", step_name="step1", step_index=0,
                       result="ok", status="completed")
    await cp.save_step(thread_id="wf1", step_name="step2", step_index=1,
                       result="partial", status="in_progress")

    steps = await cp.get_completed_steps("wf1")
    assert "step1" in steps
    assert "step2" not in steps


@pytest.mark.asyncio
async def test_multiple_steps_for_same_thread():
    cp = InMemoryCheckpointer()
    await cp.save_step(thread_id="wf1", step_name="fetch", step_index=0, result={"data": [1, 2]})
    await cp.save_step(thread_id="wf1", step_name="process", step_index=1, result="summary")
    await cp.save_step(thread_id="wf1", step_name="publish", step_index=2, result=True)

    steps = await cp.get_completed_steps("wf1")
    assert len(steps) == 3
    assert steps["fetch"] == {"data": [1, 2]}
    assert steps["process"] == "summary"
    assert steps["publish"] is True


@pytest.mark.asyncio
async def test_step_result_deep_copied():
    cp = InMemoryCheckpointer()
    result = {"items": [1, 2, 3]}
    await cp.save_step(thread_id="wf1", step_name="step1", step_index=0, result=result)
    result["items"].append(4)

    steps = await cp.get_completed_steps("wf1")
    assert steps["step1"]["items"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_step_overwrite():
    cp = InMemoryCheckpointer()
    await cp.save_step(thread_id="wf1", step_name="step1", step_index=0, result="v1")
    await cp.save_step(thread_id="wf1", step_name="step1", step_index=0, result="v2")

    steps = await cp.get_completed_steps("wf1")
    assert steps["step1"] == "v2"


# =============================================================================
# InMemoryCheckpointer — helpers
# =============================================================================


@pytest.mark.asyncio
async def test_all_thread_ids():
    cp = InMemoryCheckpointer()
    await cp.save(thread_id="a", agent_id="bot", history=[], tokens_used=0)
    await cp.save(thread_id="b", agent_id="bot", history=[], tokens_used=0)
    ids = cp.all_thread_ids()
    assert "a" in ids
    assert "b" in ids


@pytest.mark.asyncio
async def test_checkpoint_count():
    cp = InMemoryCheckpointer()
    assert cp.checkpoint_count() == 0
    await cp.save(thread_id="t1", agent_id="bot", history=[], tokens_used=0)
    assert cp.checkpoint_count() == 1
    await cp.save(thread_id="t2", agent_id="bot", history=[], tokens_used=0)
    assert cp.checkpoint_count() == 2


@pytest.mark.asyncio
async def test_step_count():
    cp = InMemoryCheckpointer()
    assert cp.step_count("wf1") == 0
    await cp.save_step(thread_id="wf1", step_name="s1", step_index=0, result="x")
    assert cp.step_count("wf1") == 1
    await cp.save_step(thread_id="wf1", step_name="s2", step_index=1, result="y")
    assert cp.step_count("wf1") == 2


def test_clear():
    cp = InMemoryCheckpointer()
    # Use sync event loop to seed data
    asyncio.run(cp.save(thread_id="t1", agent_id="bot", history=[], tokens_used=0))
    asyncio.run(cp.save_step(thread_id="wf1", step_name="s1", step_index=0, result="x"))
    cp.clear()
    assert cp.checkpoint_count() == 0
    assert cp.step_count("wf1") == 0


def test_repr():
    cp = InMemoryCheckpointer()
    r = repr(cp)
    assert "InMemoryCheckpointer" in r


# =============================================================================
# Integration: runner + InMemoryCheckpointer
# =============================================================================


@pytest.mark.asyncio
async def test_runner_saves_to_checkpointer():
    """AgentRunner saves history to InMemoryCheckpointer after run."""
    from unittest.mock import AsyncMock, MagicMock
    from ninetrix.runtime.runner import AgentRunner, RunnerConfig
    from ninetrix.runtime.dispatcher import ToolDispatcher
    from ninetrix._internals.types import LLMResponse

    provider = MagicMock()
    provider.complete = AsyncMock(return_value=LLMResponse(
        content="42", tool_calls=[], input_tokens=5, output_tokens=3,
    ))
    dispatcher = ToolDispatcher([])
    cp = InMemoryCheckpointer()
    runner = AgentRunner(
        provider=provider,
        dispatcher=dispatcher,
        config=RunnerConfig(name="test-bot"),
        checkpointer=cp,
    )

    result = await runner.run("What is 6 × 7?", thread_id="thread-42")
    saved = await cp.load("thread-42")

    assert saved is not None
    assert saved["agent_id"] == "test-bot"
    assert saved["thread_id"] == "thread-42"
    assert any(m["content"] == "42" for m in saved["history"])


@pytest.mark.asyncio
async def test_runner_restores_from_checkpointer():
    """AgentRunner resumes a prior conversation from InMemoryCheckpointer."""
    from unittest.mock import AsyncMock, MagicMock
    from ninetrix.runtime.runner import AgentRunner, RunnerConfig
    from ninetrix.runtime.dispatcher import ToolDispatcher
    from ninetrix._internals.types import LLMResponse

    prior = [
        {"role": "user", "content": "prior question"},
        {"role": "assistant", "content": "prior answer"},
    ]
    cp = InMemoryCheckpointer()
    await cp.save(thread_id="resume-1", agent_id="bot", history=prior, tokens_used=50)

    provider = MagicMock()
    provider.complete = AsyncMock(return_value=LLMResponse(
        content="follow-up answer", tool_calls=[], input_tokens=10, output_tokens=5,
    ))
    dispatcher = ToolDispatcher([])
    runner = AgentRunner(
        provider=provider,
        dispatcher=dispatcher,
        config=RunnerConfig(name="bot"),
        checkpointer=cp,
    )

    result = await runner.run("follow-up question", thread_id="resume-1")
    saved = await cp.load("resume-1")

    contents = [m["content"] for m in saved["history"]]
    assert "prior question" in contents
    assert "prior answer" in contents
    assert "follow-up question" in contents
    assert "follow-up answer" in contents
