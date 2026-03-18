"""
client/remote.py — RemoteAgent: Ninetrix Cloud client.

Layer: L9 — may import all layers below.

RemoteAgent calls the Ninetrix Cloud API (``POST /v1/agents/{slug}/invoke``)
and satisfies :class:`~ninetrix._internals.types.AgentProtocol`, making it a
drop-in replacement for a locally-running :class:`~ninetrix.Agent`.

API key resolution (in priority order):
  1. ``api_key=`` kwarg passed to the constructor
  2. :class:`~ninetrix._internals.tenant.TenantContext` in the current
     async context (set by middleware or ``tenant_scope()``)
  3. :class:`~ninetrix._internals.config.NinetrixConfig` — reads
     ``NINETRIX_API_KEY`` env var or ``~/.ninetrix/config.toml``

Example::

    from ninetrix import RemoteAgent

    analyst = RemoteAgent("my-workspace/analyst", api_key="nxt_...")
    result  = analyst.run("Compare AAPL and MSFT")
    print(result.output)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, AsyncIterator, Optional

from ninetrix._internals.config import NinetrixConfig
from ninetrix._internals.http import get_http_client
from ninetrix._internals.tenant import get_tenant
from ninetrix._internals.types import (
    AgentResult,
    CredentialError,
    ProviderError,
    StreamEvent,
)


# Default Ninetrix Cloud API base URL — overridable via env var
_CLOUD_API_URL = "https://api.ninetrix.io"


def _run_in_thread(coro: Any) -> Any:
    """Run *coro* in a new event loop on a background thread."""
    def _run() -> Any:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run).result()


class RemoteAgent:
    """Ninetrix Cloud agent client.

    Calls ``POST /v1/agents/{slug}/invoke`` on the Ninetrix Cloud API.
    Satisfies :class:`~ninetrix._internals.types.AgentProtocol` — swap freely
    with :class:`~ninetrix.Agent` or :class:`~ninetrix.AgentClient` in
    Workflows and Teams.

    HTTP errors (4xx / 5xx) are raised as
    :class:`~ninetrix._internals.types.ProviderError`.  Missing API key raises
    :class:`~ninetrix._internals.types.CredentialError`.

    Args:
        slug:    Agent identifier, either ``"workspace/agent-name"`` or just
                 ``"agent-name"`` (uses the workspace from TenantContext).
        api_key: Ninetrix Cloud API key (``nxt_...``).  When not provided, the
                 key is resolved from :class:`~ninetrix.TenantContext` or the
                 ``NINETRIX_API_KEY`` environment variable.

    Example::

        from ninetrix import RemoteAgent

        agent  = RemoteAgent("acme/data-analyst", api_key="nxt_...")
        result = agent.run("What is the YoY revenue growth?")
        print(result.output)
    """

    def __init__(self, slug: str, *, api_key: str = "") -> None:
        self._slug = slug
        self._explicit_api_key = api_key
        # name is the last segment of the slug
        self.name = slug.split("/")[-1]

    # ------------------------------------------------------------------
    # AgentProtocol implementation
    # ------------------------------------------------------------------

    def run(
        self,
        message: str,
        *,
        thread_id: Optional[str] = None,
    ) -> AgentResult:
        """Run synchronously on a background thread.

        Args:
            message:   User message.
            thread_id: Conversation ID for multi-turn resumption.

        Returns:
            :class:`~ninetrix.AgentResult`

        Raises:
            :class:`~ninetrix.CredentialError`: API key missing.
            :class:`~ninetrix.ProviderError`:   HTTP error from the cloud API.
        """
        return _run_in_thread(self.arun(message, thread_id=thread_id))

    async def arun(
        self,
        message: str,
        *,
        thread_id: Optional[str] = None,
    ) -> AgentResult:
        """Run asynchronously.

        Args:
            message:   User message.
            thread_id: Conversation ID for multi-turn resumption.

        Returns:
            :class:`~ninetrix.AgentResult`

        Raises:
            :class:`~ninetrix.CredentialError`: API key missing.
            :class:`~ninetrix.ProviderError`:   HTTP error from the cloud API.
        """
        api_key = self._resolve_api_key()
        base_url = self._resolve_base_url()
        url = f"{base_url}/v1/agents/{self._slug}/invoke"

        payload: dict[str, Any] = {"message": message}
        if thread_id:
            payload["thread_id"] = thread_id

        try:
            client = get_http_client()
            resp = await client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=300.0,
            )
        except Exception as exc:
            raise ProviderError(
                f"RemoteAgent '{self._slug}' network error.\n"
                f"  URL: {url}\n"
                f"  Why: {type(exc).__name__}: {exc}\n"
                "  Fix: check your network connection and the Ninetrix Cloud status page."
            ) from exc

        if resp.status_code == 401:
            raise CredentialError(
                f"RemoteAgent '{self._slug}': authentication failed (HTTP 401).\n"
                "  Why: the API key was rejected by Ninetrix Cloud.\n"
                "  Fix: verify your api_key= value or regenerate it at app.ninetrix.io."
            )

        if resp.status_code >= 400:
            raise ProviderError(
                f"RemoteAgent '{self._slug}' returned HTTP {resp.status_code}.\n"
                f"  URL: {url}\n"
                f"  Response: {resp.text[:300]}\n"
                "  Fix: check the Ninetrix Cloud dashboard for agent errors."
            )

        try:
            data = resp.json()
        except Exception as exc:
            raise ProviderError(
                f"RemoteAgent '{self._slug}' returned non-JSON response.\n"
                f"  URL: {url}\n"
                f"  Body: {resp.text[:200]}\n"
                f"  Why: {exc}"
            ) from exc

        return AgentResult(
            output=data.get("output", ""),
            thread_id=data.get("thread_id", thread_id or ""),
            tokens_used=data.get("tokens_used", 0),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cost_usd=data.get("cost_usd", 0.0),
            steps=data.get("steps", 0),
            history=data.get("history", []),
        )

    async def stream(
        self,
        message: str,
        *,
        thread_id: Optional[str] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming stub — planned for a future PR.

        Raises:
            :class:`~ninetrix.ProviderError`: always, until streaming is implemented.
        """
        raise ProviderError(
            f"RemoteAgent '{self._slug}' does not support streaming yet.\n"
            "  Why: cloud streaming is planned for a future release.\n"
            "  Fix: use arun() for a single-shot async call instead."
        )
        yield  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_api_key(self) -> str:
        """Resolve the API key from constructor arg → tenant context → config."""
        if self._explicit_api_key:
            return self._explicit_api_key

        tenant = get_tenant()
        if tenant and tenant.api_key:
            return tenant.api_key

        # Fall back to NinetrixConfig (reads NINETRIX_API_KEY env var)
        cfg = NinetrixConfig.load()
        if cfg.api_key:
            return cfg.api_key

        raise CredentialError(
            f"No API key found for RemoteAgent '{self._slug}'.\n"
            "  Why: no api_key= was passed and NINETRIX_API_KEY is not set.\n"
            "  Fix: pass api_key='nxt_...' to RemoteAgent(), or "
            "set the NINETRIX_API_KEY environment variable."
        )

    def _resolve_base_url(self) -> str:
        """Return the cloud API base URL, allowing override via env var."""
        import os
        return os.environ.get("NINETRIX_CLOUD_URL", _CLOUD_API_URL).rstrip("/")

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"RemoteAgent(slug={self._slug!r}, name={self.name!r})"
