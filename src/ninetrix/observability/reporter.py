"""
ninetrix.observability.reporter
================================
L6 (observability) — may import L1 + stdlib + httpx.

RunnerReporter — posts agent / team / workflow lifecycle events to the
Ninetrix local dashboard (ninetrix-oss ``api/``) or SaaS API.

The posted events are identical to those emitted by CLI-generated
``entrypoint.py`` containers, so SDK-run agents appear in the same
dashboard views as Docker-run agents.

Auth resolution order
---------------------
1. ``NINETRIX_API_URL`` + ``NINETRIX_RUNNER_TOKEN`` env vars  (explicit override)
2. ``TenantContext`` (NINETRIX_WORKSPACE_ID + NINETRIX_API_KEY)  → SaaS cloud
3. ``~/.agentfile/.api-secret`` file exists  → local ninetrix-oss dashboard
4. Nothing configured  → ``RunnerReporter.resolve()`` returns ``None``

When the reporter is None the SDK runs with zero overhead — no threads, no
HTTP calls, no side effects.

Usage (automatic — no user code needed)
-----------------------------------------
The reporter is auto-wired by ``Agent.arun()``, ``WorkflowRunner.arun()``,
and ``Team.arun()``.  For explicit control::

    from ninetrix import RunnerReporter
    reporter = RunnerReporter.resolve()          # None if nothing configured
    if reporter:
        print(reporter.config.api_url)
"""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# ReporterConfig
# ---------------------------------------------------------------------------

@dataclass
class ReporterConfig:
    """Connection details resolved once at startup."""

    api_url: str    # base URL, no trailing slash  e.g. "http://localhost:8000"
    token: str      # bearer token
    is_local: bool  # True  → POST /internal/v1/runners/events
                    # False → POST /v1/runners/events


# ---------------------------------------------------------------------------
# RunnerReporter
# ---------------------------------------------------------------------------

class RunnerReporter:
    """Posts lifecycle events to the Ninetrix dashboard.

    Obtained via :meth:`resolve` — never instantiated directly by user code.

    All public methods are async and swallow every exception so a broken
    reporter or unreachable API endpoint can never crash an agent run.
    """

    def __init__(self, config: ReporterConfig) -> None:
        self._config = config

    @property
    def config(self) -> ReporterConfig:
        return self._config

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def resolve(cls) -> "RunnerReporter | None":
        """Try to resolve a reporter from environment / tenant / local secret.

        Returns ``None`` if no reporting target is configured.
        """
        import os

        # Priority 1 — explicit env vars
        api_url = os.environ.get("NINETRIX_API_URL", "").rstrip("/")
        token = os.environ.get("NINETRIX_RUNNER_TOKEN", "")
        if api_url and token:
            is_local = "localhost" in api_url or "127.0.0.1" in api_url
            return cls(ReporterConfig(api_url=api_url, token=token, is_local=is_local))

        # Priority 2 — TenantContext (set by NINETRIX_WORKSPACE_ID + NINETRIX_API_KEY)
        try:
            from ninetrix._internals.tenant import get_tenant
            tenant = get_tenant()
            if tenant and tenant.api_key:
                return cls(ReporterConfig(
                    api_url="https://api.ninetrix.io",
                    token=tenant.api_key,
                    is_local=False,
                ))
        except Exception:
            pass

        # Priority 3 — local machine secret (~/.agentfile/.api-secret)
        try:
            secret_path = Path.home() / ".agentfile" / ".api-secret"
            if secret_path.exists():
                token = secret_path.read_text(encoding="utf-8").strip()
                if token:
                    return cls(ReporterConfig(
                        api_url="http://localhost:8000",
                        token=token,
                        is_local=True,
                    ))
        except Exception:
            pass

        return None

    # ------------------------------------------------------------------
    # Agent events  (called by AgentRunner)
    # ------------------------------------------------------------------

    async def on_run_start(
        self, *, thread_id: str, trace_id: str, parent_trace_id: str,
        agent_id: str, model: str,
    ) -> None:
        await self._post([{
            "type": "thread_started",
            "data": {
                "thread_id": thread_id,
                "trace_id": trace_id,
                "parent_trace_id": parent_trace_id,
                "agent_id": agent_id,
                "model": model,
            },
        }])

    async def on_checkpoint(
        self, *, thread_id: str, trace_id: str, agent_id: str,
        step_index: int, history: list, history_meta: list,
        tokens_in: int, tokens_out: int, model: str,
    ) -> None:
        await self._post([{
            "type": "checkpoint",
            "data": {
                "thread_id": thread_id,
                "trace_id": trace_id,
                "agent_id": agent_id,
                "step_index": step_index,
                "status": "in_progress",
                "history": history,
                "history_meta": history_meta,
                "tokens_used": tokens_in + tokens_out,
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "model": model,
            },
        }])

    async def on_run_complete(
        self, *, thread_id: str, trace_id: str, tokens_used: int, model: str,
    ) -> None:
        await self._post([{
            "type": "thread_completed",
            "data": {
                "thread_id": thread_id,
                "trace_id": trace_id,
                "tokens_used": tokens_used,
                "model": model,
            },
        }])

    async def on_run_error(
        self, *, thread_id: str, trace_id: str, error: str,
    ) -> None:
        await self._post([{
            "type": "thread_error",
            "data": {
                "thread_id": thread_id,
                "trace_id": trace_id,
                "error": error,
            },
        }])

    # ------------------------------------------------------------------
    # Team events
    # ------------------------------------------------------------------

    async def on_team_start(
        self, *, thread_id: str, trace_id: str,
        team_name: str, agent_names: list[str],
    ) -> None:
        await self._post([{
            "type": "team_started",
            "data": {
                "thread_id": thread_id,
                "trace_id": trace_id,
                "team_name": team_name,
                "agents": agent_names,
            },
        }])

    async def on_team_routed(
        self, *, thread_id: str, trace_id: str, routed_to: str,
    ) -> None:
        await self._post([{
            "type": "team_routed",
            "data": {
                "thread_id": thread_id,
                "trace_id": trace_id,
                "routed_to": routed_to,
            },
        }])

    async def on_team_complete(
        self, *, thread_id: str, trace_id: str,
        routed_to: str, tokens_used: int,
    ) -> None:
        await self._post([{
            "type": "team_completed",
            "data": {
                "thread_id": thread_id,
                "trace_id": trace_id,
                "routed_to": routed_to,
                "tokens_used": tokens_used,
            },
        }])

    # ------------------------------------------------------------------
    # Workflow events
    # ------------------------------------------------------------------

    async def on_workflow_start(
        self, *, thread_id: str, trace_id: str, workflow_name: str,
    ) -> None:
        await self._post([{
            "type": "workflow_started",
            "data": {
                "thread_id": thread_id,
                "trace_id": trace_id,
                "workflow_name": workflow_name,
            },
        }])

    async def on_workflow_step_started(
        self, *, thread_id: str, trace_id: str, step_name: str,
    ) -> None:
        await self._post([{
            "type": "workflow_step_started",
            "data": {
                "thread_id": thread_id,
                "trace_id": trace_id,
                "step_name": step_name,
            },
        }])

    async def on_workflow_step_completed(
        self, *, thread_id: str, trace_id: str, step_name: str, cached: bool,
    ) -> None:
        await self._post([{
            "type": "workflow_step_completed",
            "data": {
                "thread_id": thread_id,
                "trace_id": trace_id,
                "step_name": step_name,
                "cached": cached,
            },
        }])

    async def on_workflow_map_started(
        self, *, thread_id: str, trace_id: str,
        step_prefix: str, item_count: int,
    ) -> None:
        await self._post([{
            "type": "workflow_map_started",
            "data": {
                "thread_id": thread_id,
                "trace_id": trace_id,
                "step_prefix": step_prefix,
                "item_count": item_count,
            },
        }])

    async def on_workflow_map_completed(
        self, *, thread_id: str, trace_id: str,
        step_prefix: str, completed: int, cached: int,
    ) -> None:
        await self._post([{
            "type": "workflow_map_completed",
            "data": {
                "thread_id": thread_id,
                "trace_id": trace_id,
                "step_prefix": step_prefix,
                "completed": completed,
                "cached": cached,
            },
        }])

    async def on_workflow_complete(
        self, *, thread_id: str, trace_id: str,
        completed_steps: list[str], skipped_steps: list[str],
        terminated: bool, reason: str,
    ) -> None:
        await self._post([{
            "type": "workflow_completed",
            "data": {
                "thread_id": thread_id,
                "trace_id": trace_id,
                "completed_steps": completed_steps,
                "skipped_steps": skipped_steps,
                "terminated": terminated,
                "termination_reason": reason,
            },
        }])

    # ------------------------------------------------------------------
    # HTTP transport
    # ------------------------------------------------------------------

    @property
    def _events_url(self) -> str:
        path = (
            "/internal/v1/runners/events"
            if self._config.is_local
            else "/v1/runners/events"
        )
        return self._config.api_url + path

    async def _post(self, events: list[dict[str, Any]]) -> None:
        """POST event batch.  Silently swallows all errors."""
        try:
            from ninetrix._internals.http import get_http_client
            client = get_http_client()
            await client.post(
                self._events_url,
                json={"events": events},
                headers={"Authorization": f"Bearer {self._config.token}"},
                timeout=5.0,
            )
        except Exception:
            pass  # reporter must never crash an agent run

    def __repr__(self) -> str:  # pragma: no cover
        mode = "local" if self._config.is_local else "cloud"
        return f"<RunnerReporter mode={mode} url={self._config.api_url!r}>"


# ---------------------------------------------------------------------------
# Timestamp helper used by AgentRunner
# ---------------------------------------------------------------------------

def now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
