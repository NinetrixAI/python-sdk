"""
checkpoint/base.py — abstract Checkpointer interface.

Layer: L7 (checkpoint) — may import L1 (_internals) + stdlib only.

All concrete checkpointers must implement this ABC so they are
interchangeable throughout the SDK (runner, workflow, durable steps).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Checkpointer(ABC):
    """Abstract base for checkpoint storage backends.

    The runner calls :meth:`save` and :meth:`load` for per-thread history
    persistence.  Durable workflows additionally use :meth:`save_step` and
    :meth:`get_completed_steps` to checkpoint individual steps.

    All methods are async so implementations can use async I/O (PostgreSQL,
    Redis, S3, …) without blocking the event loop.

    Example::

        cp = InMemoryCheckpointer()
        await cp.save(thread_id="t1", agent_id="bot", history=[...], tokens_used=42)
        state = await cp.load("t1")
        assert state["history"] == [...]
    """

    @abstractmethod
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
        """Persist the full conversation history for *thread_id*.

        Args:
            thread_id:   Unique conversation identifier.
            agent_id:    Name of the agent that ran.
            history:     Message list (system messages excluded by convention).
            tokens_used: Total tokens consumed in this run.
            status:      Run status string (``"completed"``, ``"error"`` …).
            metadata:    Arbitrary key→value pairs stored alongside the checkpoint.
        """

    @abstractmethod
    async def load(self, thread_id: str) -> dict[str, Any] | None:
        """Return the latest checkpoint for *thread_id*, or ``None``.

        The returned dict always contains at least:
            ``history``     — list of message dicts
            ``agent_id``    — agent name
            ``tokens_used`` — integer
            ``status``      — string
        """

    @abstractmethod
    async def delete(self, thread_id: str) -> None:
        """Remove all checkpoint data for *thread_id*."""

    @abstractmethod
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

        Used by durable ``@Workflow(durable=True)`` to checkpoint each
        ``async with Workflow.step("name"):`` block.

        Args:
            thread_id:   Workflow run identifier.
            step_name:   Step label (unique within a workflow run).
            step_index:  Zero-based step position (for ordering).
            result:      Serialisable step output (str, dict, list, …).
            status:      Step status (``"completed"``, ``"skipped"`` …).
        """

    @abstractmethod
    async def get_completed_steps(self, thread_id: str) -> dict[str, Any]:
        """Return ``{step_name: result}`` for all completed steps of *thread_id*.

        Used at workflow resume time to determine which steps can be
        skipped.  Returns an empty dict if no steps have been saved.
        """
