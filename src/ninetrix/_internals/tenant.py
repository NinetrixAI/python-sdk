"""
ninetrix._internals.tenant
===========================
L1 kernel — stdlib + _internals only.

``TenantContext`` scopes every agent run to an organization so that multiple
tenants can share a single SDK process without credential or data leakage.

Uses Python ``contextvars`` — asyncio-safe, propagates automatically through
``await`` chains without being threaded through every function argument.

Three-tier usage model
----------------------

1. **Scripts / CLI** (zero setup):
   ``_auto_init_from_env()`` runs at import and reads ``NINETRIX_ORG_ID``
   + ``NINETRIX_API_KEY`` from the environment.  Single-tenant users never
   call ``set_tenant()`` — it just works.

2. **SaaS / FastAPI** (one call per request at the boundary):

   .. code-block:: python

       @app.post("/v1/run")
       async def run_handler(req, user=Depends(auth)):
           async with tenant_scope(TenantContext(
               org_id=user.org_id,
               api_key=user.nxt_api_key,
           )):
               result = await agent.arun(req.message)

3. **Tests** (via ``AgentSandbox(tenant=...)``):
   ``AgentSandbox`` calls ``tenant_scope()`` internally — test code never
   touches the context var directly.

Public exports
--------------
``TenantContext``   — immutable tenant dataclass
``set_tenant``      — set tenant and get a reset token (manual, framework authors)
``get_tenant``      — current tenant or None
``require_tenant``  — current tenant or ConfigurationError (for internal layers)
``tenant_scope``    — async context manager (preferred in application code)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import AsyncGenerator

from ninetrix._internals.types import ConfigurationError

# ---------------------------------------------------------------------------
# TenantContext dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TenantContext:
    """
    Immutable tenant/org scope for a single agent run.

    Set once at the request boundary; propagates automatically through all
    ``await`` chains in the same asyncio task.

    Attributes:
        org_id:       Ninetrix organization identifier (required).
        api_key:      ``nxt_...`` organization token for Ninetrix Cloud API calls.
        region:       Cloud region hint (``"us"`` / ``"eu"``).
        db_schema:    PostgreSQL schema for this tenant's data (multi-tenant DB).
    """

    org_id: str
    api_key: str = field(default="", repr=False)  # never printed; prevents log leakage
    region: str = "us"
    db_schema: str = "public"


# ---------------------------------------------------------------------------
# ContextVar — process-wide, default None
# ---------------------------------------------------------------------------

_current_tenant: ContextVar[TenantContext | None] = ContextVar(
    "_current_tenant", default=None
)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def set_tenant(ctx: TenantContext) -> Token[TenantContext | None]:
    """
    Set the active tenant for the current async task.

    Returns a ``Token`` that can be used to restore the previous value via
    ``_current_tenant.reset(token)``.  Prefer ``tenant_scope()`` which handles
    cleanup automatically.

    Intended for framework/middleware authors.  Application code should use
    ``tenant_scope()``.
    """
    return _current_tenant.set(ctx)


def get_tenant() -> TenantContext | None:
    """Return the current ``TenantContext``, or ``None`` if none is set."""
    return _current_tenant.get()


def require_tenant() -> TenantContext:
    """
    Return the current ``TenantContext``.

    Raises ``ConfigurationError`` with a clear fix message if no tenant is set.
    Used inside checkpointers, MCP tool sources, and API clients — anywhere a
    missing tenant is a programming error, not a user error.
    """
    tenant = _current_tenant.get()
    if tenant is None:
        raise ConfigurationError(
            "No TenantContext is set for this request.\n"
            "  Why: The calling code requires a tenant scope but none was established.\n"
            "  Fix:\n"
            "    → In a FastAPI handler: async with tenant_scope(TenantContext(...)):\n"
            "    → In a script: call set_tenant(TenantContext(org_id=..., api_key=...)) first\n"
            "    → In tests: use AgentSandbox(agent, tenant=TenantContext(org_id='test-org'))"
        )
    return tenant


@asynccontextmanager
async def tenant_scope(
    ctx: TenantContext,
) -> AsyncGenerator[TenantContext, None]:
    """
    Async context manager — sets the active tenant for the duration of the
    block, then restores the previous value.  Nestable.

    .. code-block:: python

        async with tenant_scope(TenantContext(org_id="org-123", api_key="nxt_...")):
            result = await agent.arun("analyse this quarter's results")
    """
    token = _current_tenant.set(ctx)
    try:
        yield ctx
    finally:
        _current_tenant.reset(token)


# ---------------------------------------------------------------------------
# Auto-init from environment (called at module import)
# ---------------------------------------------------------------------------

def _auto_init_from_env() -> None:
    """
    If ``NINETRIX_ORG_ID`` is set in the environment, create a default
    ``TenantContext`` and set it as the process-wide tenant.

    This makes single-tenant scripts work with zero setup::

        export NINETRIX_ORG_ID=org-abc123
        export NINETRIX_API_KEY=nxt_...

    Both env vars must be present; if only one is set, no context is created
    (incomplete configuration is not silently accepted).
    """
    import os

    org_id = os.environ.get("NINETRIX_ORG_ID", "")
    api_key = os.environ.get("NINETRIX_API_KEY", "")

    if org_id and api_key:
        _current_tenant.set(
            TenantContext(org_id=org_id, api_key=api_key)
        )


# Run auto-init once at import time
_auto_init_from_env()
