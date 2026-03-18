"""
checkpoint/postgres.py — PostgreSQL-backed checkpointer.

Layer: L7 (checkpoint) — may import L1 (_internals) + stdlib + asyncpg only.

Stores agent history and durable workflow step state in the
``agentfile_checkpoints`` table — the same table used by the CLI and ``api/``
components, so checkpoints are readable across the whole platform.

The ``asyncpg`` library is a soft dependency: it is imported inside
:meth:`connect` so the SDK can be imported without it installed.  Users only
need it when they actually use :class:`PostgresCheckpointer`.

Usage::

    from ninetrix.checkpoint.postgres import PostgresCheckpointer

    cp = PostgresCheckpointer("postgresql://user:pass@localhost/mydb")
    await cp.connect()

    await cp.save(thread_id="t1", agent_id="bot",
                  history=[...], tokens_used=42)

    state = await cp.load("t1")
    await cp.disconnect()

    # Or as an async context manager:
    async with PostgresCheckpointer(db_url) as cp:
        await cp.save(...)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from ninetrix.checkpoint.base import Checkpointer


# ---------------------------------------------------------------------------
# PostgresCheckpointer
# ---------------------------------------------------------------------------


class PostgresCheckpointer(Checkpointer):
    """Persist agent history and workflow steps to PostgreSQL.

    Reuses the ``agentfile_checkpoints`` table (identical schema to the CLI
    and API components — no new migrations needed).

    The JSONB ``checkpoint`` column stores::

        {
            "history": [...],       # message list
            "tokens_used": 42,
            "workflow_steps": {     # durable workflow state
                "research": {"result": "...", "step_index": 0, "status": "completed"},
                ...
            }
        }

    Args:
        db_url: asyncpg-compatible DSN, e.g.
                ``"postgresql://user:pass@localhost/db"``

    Requires ``asyncpg``::

        pip install asyncpg
        # or
        pip install "ninetrix[postgres]"
    """

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url
        self._conn: Any = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the asyncpg connection and ensure the table schema exists.

        Must be called before any read/write operations.

        Raises:
            ImportError: if ``asyncpg`` is not installed.
        """
        try:
            import asyncpg  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "asyncpg is required for PostgresCheckpointer.\n"
                "  Fix: pip install asyncpg"
            ) from exc

        self._conn = await asyncpg.connect(self._db_url)
        await self._ensure_schema()

    async def disconnect(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> "PostgresCheckpointer":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.disconnect()

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
        """Upsert the conversation history for *thread_id*.

        Uses ``SELECT … FOR UPDATE`` inside a transaction to prevent
        concurrent writes from overwriting each other.

        Args:
            thread_id:   Unique conversation identifier.
            agent_id:    Agent name.
            history:     Message list (system messages excluded by convention).
            tokens_used: Total tokens consumed.
            status:      Run status (``"completed"``, ``"error"`` …).
            metadata:    Arbitrary extra data stored alongside the checkpoint.
        """
        self._require_connected()
        checkpoint_data = json.dumps({
            "history": history,
            "tokens_used": tokens_used,
            "workflow_steps": {},
        })
        meta_data = json.dumps(metadata or {})
        trace_id = uuid.uuid4().hex

        async with self._conn.transaction():
            row = await self._conn.fetchrow(
                """
                SELECT id FROM agentfile_checkpoints
                WHERE thread_id = $1 AND agent_id = $2
                ORDER BY step_index DESC
                LIMIT 1
                FOR UPDATE
                """,
                thread_id,
                agent_id,
            )
            if row:
                await self._conn.execute(
                    """
                    UPDATE agentfile_checkpoints
                    SET trace_id = $1,
                        step_index = step_index + 1,
                        status = $2,
                        checkpoint = $3::jsonb,
                        metadata = $4::jsonb,
                        timestamp = NOW()
                    WHERE id = $5
                    """,
                    trace_id,
                    status,
                    checkpoint_data,
                    meta_data,
                    row["id"],
                )
            else:
                await self._conn.execute(
                    """
                    INSERT INTO agentfile_checkpoints
                        (trace_id, thread_id, agent_id, step_index, status,
                         checkpoint, metadata)
                    VALUES ($1, $2, $3, 0, $4, $5::jsonb, $6::jsonb)
                    """,
                    trace_id,
                    thread_id,
                    agent_id,
                    status,
                    checkpoint_data,
                    meta_data,
                )

    async def load(self, thread_id: str) -> dict[str, Any] | None:
        """Return the latest checkpoint for *thread_id*, or ``None``.

        The returned dict always contains ``history``, ``agent_id``,
        ``tokens_used``, and ``status``.
        """
        self._require_connected()
        row = await self._conn.fetchrow(
            """
            SELECT agent_id, status, checkpoint, metadata
            FROM agentfile_checkpoints
            WHERE thread_id = $1
            ORDER BY step_index DESC
            LIMIT 1
            """,
            thread_id,
        )
        if row is None:
            return None

        data = json.loads(row["checkpoint"])
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        return {
            "thread_id": thread_id,
            "agent_id": row["agent_id"],
            "history": data.get("history", []),
            "tokens_used": data.get("tokens_used", 0),
            "status": row["status"],
            "metadata": meta,
        }

    async def delete(self, thread_id: str) -> None:
        """Remove all checkpoint rows for *thread_id*."""
        self._require_connected()
        await self._conn.execute(
            "DELETE FROM agentfile_checkpoints WHERE thread_id = $1",
            thread_id,
        )

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
        """Persist a single workflow step result inside the checkpoint JSONB.

        Uses PostgreSQL ``jsonb_set()`` for an atomic partial update so that
        concurrent step saves do not overwrite each other's data.

        Inserts a new checkpoint row if none exists for *thread_id*.
        """
        self._require_connected()
        step_data = json.dumps({
            "result": result,
            "step_index": step_index,
            "status": status,
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
        })
        trace_id = uuid.uuid4().hex

        async with self._conn.transaction():
            row = await self._conn.fetchrow(
                """
                SELECT id FROM agentfile_checkpoints
                WHERE thread_id = $1
                ORDER BY step_index DESC
                LIMIT 1
                FOR UPDATE
                """,
                thread_id,
            )
            if row:
                # Atomic partial JSONB update — adds/overwrites only this step
                await self._conn.execute(
                    """
                    UPDATE agentfile_checkpoints
                    SET checkpoint = jsonb_set(
                            checkpoint,
                            ARRAY['workflow_steps', $1::text],
                            $2::jsonb,
                            true
                        ),
                        timestamp = NOW()
                    WHERE id = $3
                    """,
                    step_name,
                    step_data,
                    row["id"],
                )
            else:
                # No checkpoint row yet — create one with just this step
                checkpoint_data = json.dumps({
                    "history": [],
                    "tokens_used": 0,
                    "workflow_steps": {step_name: json.loads(step_data)},
                })
                await self._conn.execute(
                    """
                    INSERT INTO agentfile_checkpoints
                        (trace_id, thread_id, agent_id, step_index, status,
                         checkpoint, metadata)
                    VALUES ($1, $2, 'workflow', 0, 'in_progress', $3::jsonb, '{}'::jsonb)
                    """,
                    trace_id,
                    thread_id,
                    checkpoint_data,
                )

    async def get_completed_steps(self, thread_id: str) -> dict[str, Any]:
        """Return ``{step_name: result}`` for all completed steps of *thread_id*.

        Only steps whose ``status`` is ``"completed"`` are returned.
        Returns an empty dict if no checkpoint exists.
        """
        self._require_connected()
        row = await self._conn.fetchrow(
            """
            SELECT checkpoint->'workflow_steps' AS steps
            FROM agentfile_checkpoints
            WHERE thread_id = $1
            ORDER BY step_index DESC
            LIMIT 1
            """,
            thread_id,
        )
        if row is None or row["steps"] is None:
            return {}

        steps = json.loads(row["steps"])
        return {
            name: data["result"]
            for name, data in steps.items()
            if isinstance(data, dict) and data.get("status") == "completed"
        }

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    async def _ensure_schema(self) -> None:
        """Create the checkpoints table and indexes if they do not exist.

        Runs each DDL statement separately (asyncpg requires explicit
        transaction management for DDL).  Idempotent — safe to call on every
        ``connect()``.
        """
        ddl_statements = [
            """
            CREATE TABLE IF NOT EXISTS agentfile_checkpoints (
                id            BIGSERIAL PRIMARY KEY,
                trace_id      TEXT NOT NULL,
                thread_id     TEXT NOT NULL,
                agent_id      TEXT NOT NULL,
                step_index    INTEGER NOT NULL,
                timestamp     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                status        TEXT NOT NULL,
                checkpoint    JSONB NOT NULL,
                metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
                parent_trace_id TEXT
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_af_thread_id ON agentfile_checkpoints(thread_id)",
            "CREATE INDEX IF NOT EXISTS idx_af_trace_id  ON agentfile_checkpoints(trace_id)",
            # Add column idempotently in case table already exists without it
            "ALTER TABLE agentfile_checkpoints ADD COLUMN IF NOT EXISTS parent_trace_id TEXT",
        ]
        for stmt in ddl_statements:
            await self._conn.execute(stmt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if self._conn is None:
            raise RuntimeError(
                "PostgresCheckpointer is not connected.\n"
                "  Fix: call `await cp.connect()` before using it, "
                "or use it as an async context manager: "
                "`async with PostgresCheckpointer(db_url) as cp:`"
            )
