"""
checkpoint/memory.py — in-process checkpoint store for testing and dev.

Layer: L7 (checkpoint) — may import L1 (_internals) + stdlib only.

InMemoryCheckpointer keeps all state in a plain Python dict.  It is
thread-safe for async use (single-process) but is not persistent across
restarts.  Use it in unit tests and during local development.

Example::

    from ninetrix.checkpoint.memory import InMemoryCheckpointer

    cp = InMemoryCheckpointer()
    await cp.save(thread_id="t1", agent_id="bot",
                  history=[{"role": "user", "content": "hi"}],
                  tokens_used=10)
    state = await cp.load("t1")
    assert state["history"][0]["content"] == "hi"
"""

from __future__ import annotations

import copy
import time
from typing import Any

from ninetrix.checkpoint.base import Checkpointer


class InMemoryCheckpointer(Checkpointer):
    """In-process, non-persistent checkpoint store.

    Stores checkpoints and workflow step results as plain Python dicts.
    All data is lost when the process exits.  Intended for unit tests and
    local development — swap for :class:`~ninetrix.checkpoint.postgres.PostgresCheckpointer`
    in production.

    Thread safety:
        Safe for concurrent async use within a single event loop.
        Not safe across OS threads (no lock around dict mutation).
    """

    def __init__(self) -> None:
        # thread_id → latest checkpoint dict
        self._checkpoints: dict[str, dict[str, Any]] = {}
        # thread_id → {step_name: {result, step_index, status, saved_at}}
        self._steps: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Agent history API
    # ------------------------------------------------------------------

    async def save(
        self,
        *,
        thread_id: str,
        agent_id: str,
        history: list[dict],
        tokens_used: int,
        status: str = "completed",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a deep copy of *history* keyed by *thread_id*."""
        self._checkpoints[thread_id] = {
            "thread_id": thread_id,
            "agent_id": agent_id,
            "history": copy.deepcopy(history),
            "tokens_used": tokens_used,
            "status": status,
            "metadata": copy.deepcopy(metadata or {}),
            "saved_at": time.time(),
        }

    async def load(self, thread_id: str) -> dict[str, Any] | None:
        """Return the latest checkpoint for *thread_id*, or ``None``."""
        checkpoint = self._checkpoints.get(thread_id)
        if checkpoint is None:
            return None
        return copy.deepcopy(checkpoint)

    async def delete(self, thread_id: str) -> None:
        """Remove checkpoint and all step data for *thread_id*."""
        self._checkpoints.pop(thread_id, None)
        self._steps.pop(thread_id, None)

    # ------------------------------------------------------------------
    # Workflow step API
    # ------------------------------------------------------------------

    async def save_step(
        self,
        *,
        thread_id: str,
        step_name: str,
        step_index: int,
        result: Any,
        status: str = "completed",
    ) -> None:
        """Persist a single workflow step result.

        If *thread_id* has no prior steps, an empty dict is created first.
        """
        if thread_id not in self._steps:
            self._steps[thread_id] = {}
        self._steps[thread_id][step_name] = {
            "result": copy.deepcopy(result),
            "step_index": step_index,
            "status": status,
            "saved_at": time.time(),
        }

    async def get_completed_steps(self, thread_id: str) -> dict[str, Any]:
        """Return ``{step_name: result}`` for all completed steps.

        Only steps with ``status == "completed"`` are included.
        """
        steps = self._steps.get(thread_id, {})
        return {
            name: data["result"]
            for name, data in steps.items()
            if data.get("status") == "completed"
        }

    # ------------------------------------------------------------------
    # Testing helpers
    # ------------------------------------------------------------------

    def all_thread_ids(self) -> list[str]:
        """Return all thread IDs that have a saved checkpoint."""
        return list(self._checkpoints.keys())

    def checkpoint_count(self) -> int:
        """Return the total number of saved checkpoints."""
        return len(self._checkpoints)

    def step_count(self, thread_id: str) -> int:
        """Return the number of steps saved for *thread_id*."""
        return len(self._steps.get(thread_id, {}))

    def clear(self) -> None:
        """Remove all checkpoints and steps.  Intended for test teardown."""
        self._checkpoints.clear()
        self._steps.clear()

    def __repr__(self) -> str:
        return (
            f"InMemoryCheckpointer("
            f"threads={len(self._checkpoints)}, "
            f"step_threads={len(self._steps)})"
        )
