"""Tests for checkpoint/postgres.py — PR 28.

All tests mock asyncpg so no real database is required.
The mocks mirror the asyncpg connection/transaction API surface.
"""

from __future__ import annotations

import json
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ninetrix.checkpoint.postgres import PostgresCheckpointer


# ---------------------------------------------------------------------------
# Helpers — build a minimal asyncpg mock
# ---------------------------------------------------------------------------


def _make_connection(
    fetchrow_result=None,
    execute_result=None,
):
    """Build a mock asyncpg connection + transaction context manager."""
    txn = MagicMock()
    txn.__aenter__ = AsyncMock(return_value=txn)
    txn.__aexit__ = AsyncMock(return_value=False)

    conn = MagicMock()
    conn.execute = AsyncMock(return_value=execute_result)
    conn.fetchrow = AsyncMock(return_value=fetchrow_result)
    conn.transaction = MagicMock(return_value=txn)
    conn.close = AsyncMock()
    return conn


def _make_asyncpg_module(conn):
    """Return a fake asyncpg module whose connect() returns conn."""
    asyncpg = ModuleType("asyncpg")
    asyncpg.connect = AsyncMock(return_value=conn)
    return asyncpg


@pytest.fixture
def simple_conn():
    return _make_connection()


@pytest.fixture
def cp_with_conn(simple_conn):
    """PostgresCheckpointer with a pre-injected mock connection (already connected)."""
    cp = PostgresCheckpointer("postgresql://test/db")
    cp._conn = simple_conn
    return cp, simple_conn


# =============================================================================
# Construction
# =============================================================================


def test_constructs():
    cp = PostgresCheckpointer("postgresql://user:pass@localhost/db")
    assert cp._db_url == "postgresql://user:pass@localhost/db"
    assert cp._conn is None


def test_not_connected_raises():
    cp = PostgresCheckpointer("postgresql://x/y")
    with pytest.raises(RuntimeError, match="not connected"):
        cp._require_connected()


# =============================================================================
# connect() — happy path
# =============================================================================


@pytest.mark.asyncio
async def test_connect_calls_asyncpg():
    conn = _make_connection()
    asyncpg_mod = _make_asyncpg_module(conn)

    cp = PostgresCheckpointer("postgresql://x/y")
    with patch.dict(sys.modules, {"asyncpg": asyncpg_mod}):
        await cp.connect()

    asyncpg_mod.connect.assert_called_once_with("postgresql://x/y")
    assert cp._conn is conn


@pytest.mark.asyncio
async def test_connect_calls_ensure_schema():
    conn = _make_connection()
    asyncpg_mod = _make_asyncpg_module(conn)

    cp = PostgresCheckpointer("postgresql://x/y")
    with patch.dict(sys.modules, {"asyncpg": asyncpg_mod}):
        await cp.connect()

    # _ensure_schema runs 4 DDL statements
    assert conn.execute.call_count >= 4


@pytest.mark.asyncio
async def test_connect_missing_asyncpg_raises():
    cp = PostgresCheckpointer("postgresql://x/y")
    with patch.dict(sys.modules, {"asyncpg": None}):
        with pytest.raises(ImportError, match="asyncpg"):
            await cp.connect()


# =============================================================================
# disconnect()
# =============================================================================


@pytest.mark.asyncio
async def test_disconnect_closes_connection(cp_with_conn):
    cp, conn = cp_with_conn
    await cp.disconnect()
    conn.close.assert_called_once()
    assert cp._conn is None


@pytest.mark.asyncio
async def test_disconnect_noop_when_not_connected():
    cp = PostgresCheckpointer("postgresql://x/y")
    await cp.disconnect()  # should not raise


# =============================================================================
# Async context manager
# =============================================================================


@pytest.mark.asyncio
async def test_context_manager_connects_and_disconnects():
    conn = _make_connection()
    asyncpg_mod = _make_asyncpg_module(conn)

    with patch.dict(sys.modules, {"asyncpg": asyncpg_mod}):
        async with PostgresCheckpointer("postgresql://x/y") as cp:
            assert cp._conn is not None

    conn.close.assert_called_once()
    assert cp._conn is None


# =============================================================================
# save() — no existing row (INSERT path)
# =============================================================================


@pytest.mark.asyncio
async def test_save_inserts_when_no_row(cp_with_conn):
    cp, conn = cp_with_conn
    conn.fetchrow.return_value = None  # no existing row

    await cp.save(
        thread_id="t1",
        agent_id="bot",
        history=[{"role": "user", "content": "hi"}],
        tokens_used=10,
    )

    # Should execute INSERT (execute called at least once after the schema DDL)
    assert conn.execute.called
    # Check INSERT SQL
    last_call_sql = conn.execute.call_args_list[-1][0][0].strip().upper()
    assert "INSERT" in last_call_sql


@pytest.mark.asyncio
async def test_save_updates_when_row_exists(cp_with_conn):
    cp, conn = cp_with_conn
    conn.fetchrow.return_value = {"id": 42}

    await cp.save(
        thread_id="t1",
        agent_id="bot",
        history=[],
        tokens_used=5,
    )

    last_call_sql = conn.execute.call_args_list[-1][0][0].strip().upper()
    assert "UPDATE" in last_call_sql


@pytest.mark.asyncio
async def test_save_uses_transaction(cp_with_conn):
    cp, conn = cp_with_conn
    conn.fetchrow.return_value = None

    await cp.save(thread_id="t1", agent_id="bot", history=[], tokens_used=0)
    conn.transaction.assert_called()


@pytest.mark.asyncio
async def test_save_uses_select_for_update(cp_with_conn):
    cp, conn = cp_with_conn
    conn.fetchrow.return_value = None

    await cp.save(thread_id="t1", agent_id="bot", history=[], tokens_used=0)

    select_sql = conn.fetchrow.call_args[0][0].upper()
    assert "FOR UPDATE" in select_sql


@pytest.mark.asyncio
async def test_save_checkpoint_json_contains_history(cp_with_conn):
    cp, conn = cp_with_conn
    conn.fetchrow.return_value = None
    history = [{"role": "user", "content": "hello"}]

    await cp.save(thread_id="t1", agent_id="bot", history=history, tokens_used=7)

    # Find the INSERT call and inspect the checkpoint JSON argument
    # args: (sql, trace_id, thread_id, agent_id, status, checkpoint_json, meta_json)
    insert_call = conn.execute.call_args_list[-1]
    checkpoint_arg = insert_call[0][5]   # 6th positional arg is checkpoint JSONB
    data = json.loads(checkpoint_arg)
    assert data["history"] == history
    assert data["tokens_used"] == 7


# =============================================================================
# load()
# =============================================================================


@pytest.mark.asyncio
async def test_load_returns_none_when_no_row(cp_with_conn):
    cp, conn = cp_with_conn
    conn.fetchrow.return_value = None

    result = await cp.load("missing")
    assert result is None


@pytest.mark.asyncio
async def test_load_returns_checkpoint(cp_with_conn):
    cp, conn = cp_with_conn
    checkpoint_data = json.dumps({
        "history": [{"role": "user", "content": "hi"}],
        "tokens_used": 15,
        "workflow_steps": {},
    })
    conn.fetchrow.return_value = {
        "agent_id": "bot",
        "status": "completed",
        "checkpoint": checkpoint_data,
        "metadata": "{}",
    }

    result = await cp.load("t1")
    assert result is not None
    assert result["thread_id"] == "t1"
    assert result["agent_id"] == "bot"
    assert result["tokens_used"] == 15
    assert result["history"] == [{"role": "user", "content": "hi"}]
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_load_queries_by_thread_id(cp_with_conn):
    cp, conn = cp_with_conn
    conn.fetchrow.return_value = None

    await cp.load("my-thread")

    sql = conn.fetchrow.call_args[0][0].upper()
    assert "WHERE" in sql
    assert conn.fetchrow.call_args[0][1] == "my-thread"


# =============================================================================
# delete()
# =============================================================================


@pytest.mark.asyncio
async def test_delete_executes_delete(cp_with_conn):
    cp, conn = cp_with_conn
    await cp.delete("t1")

    sql = conn.execute.call_args[0][0].upper()
    assert "DELETE" in sql
    assert conn.execute.call_args[0][1] == "t1"


# =============================================================================
# save_step() — no existing row
# =============================================================================


@pytest.mark.asyncio
async def test_save_step_inserts_when_no_row(cp_with_conn):
    cp, conn = cp_with_conn
    conn.fetchrow.return_value = None

    await cp.save_step(
        thread_id="wf1", step_name="research", step_index=0, result="done"
    )

    last_sql = conn.execute.call_args[0][0].strip().upper()
    assert "INSERT" in last_sql


@pytest.mark.asyncio
async def test_save_step_updates_jsonb_when_row_exists(cp_with_conn):
    cp, conn = cp_with_conn
    conn.fetchrow.return_value = {"id": 7}

    await cp.save_step(
        thread_id="wf1", step_name="analysis", step_index=1, result="result_data"
    )

    last_sql = conn.execute.call_args[0][0].upper()
    assert "UPDATE" in last_sql
    assert "JSONB_SET" in last_sql


@pytest.mark.asyncio
async def test_save_step_uses_select_for_update(cp_with_conn):
    cp, conn = cp_with_conn
    conn.fetchrow.return_value = None

    await cp.save_step(
        thread_id="wf1", step_name="step1", step_index=0, result="ok"
    )

    select_sql = conn.fetchrow.call_args[0][0].upper()
    assert "FOR UPDATE" in select_sql


# =============================================================================
# get_completed_steps()
# =============================================================================


@pytest.mark.asyncio
async def test_get_completed_steps_returns_empty_when_no_row(cp_with_conn):
    cp, conn = cp_with_conn
    conn.fetchrow.return_value = None

    result = await cp.get_completed_steps("t1")
    assert result == {}


@pytest.mark.asyncio
async def test_get_completed_steps_returns_completed_only(cp_with_conn):
    cp, conn = cp_with_conn
    steps_json = json.dumps({
        "research": {"result": "found stuff", "step_index": 0, "status": "completed"},
        "draft":    {"result": None,           "step_index": 1, "status": "in_progress"},
    })
    conn.fetchrow.return_value = {"steps": steps_json}

    result = await cp.get_completed_steps("wf1")
    assert "research" in result
    assert result["research"] == "found stuff"
    assert "draft" not in result  # only completed


@pytest.mark.asyncio
async def test_get_completed_steps_returns_empty_when_steps_null(cp_with_conn):
    cp, conn = cp_with_conn
    conn.fetchrow.return_value = {"steps": None}

    result = await cp.get_completed_steps("t1")
    assert result == {}


# =============================================================================
# Public imports
# =============================================================================


def test_postgres_checkpointer_importable():
    from ninetrix import PostgresCheckpointer
    assert PostgresCheckpointer is not None


def test_postgres_checkpointer_is_checkpointer():
    from ninetrix.checkpoint.base import Checkpointer
    assert issubclass(PostgresCheckpointer, Checkpointer)


def test_postgres_importable_from_checkpoint():
    from ninetrix.checkpoint.postgres import PostgresCheckpointer
    assert PostgresCheckpointer is not None
