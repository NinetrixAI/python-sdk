"""
client/local.py — AgentClient: HTTP wrapper for a running agent container.

Layer: L9 — may import all layers below.

AgentClient wraps a running agent container's ``/invoke`` endpoint and satisfies
:class:`~ninetrix._internals.types.AgentProtocol`, making it a drop-in
replacement for :class:`~ninetrix.agent.agent.Agent` in any Workflow or Team.

Typical usage (agent container started with ``agentfile up``)::

    from ninetrix import AgentClient

    analyst = AgentClient("http://analyst:9000", name="analyst")

    # All three call styles work:
    result = analyst.run("Compare AAPL and MSFT")           # sync
    result = await analyst.arun("...", thread_id="t1")       # async
    async for event in analyst.stream("..."):  pass          # streaming stub
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, AsyncIterator, Optional

from ninetrix._internals.http import get_http_client
from ninetrix._internals.tenant import get_tenant
from ninetrix._internals.types import (
    AgentResult,
    ProviderError,
    StreamEvent,
)


def _run_in_thread(coro: Any) -> Any:
    """Run *coro* in a new event loop on a background thread.

    Safe to call from a running event loop (Jupyter, FastAPI, pytest-asyncio).
    """
    def _run() -> Any:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run).result()


class AgentClient:
    """HTTP wrapper for a running agent container.

    Calls the agent's ``POST /invoke`` endpoint and returns an
    :class:`~ninetrix.AgentResult`.  Satisfies
    :class:`~ninetrix._internals.types.AgentProtocol` — swap freely with
    :class:`~ninetrix.Agent`, :class:`~ninetrix.RemoteAgent`, or any other
    protocol-compatible object in Workflows and Teams.

    HTTP errors (4xx / 5xx) are raised as
    :class:`~ninetrix._internals.types.ProviderError`, never as raw
    ``httpx`` exceptions.

    Args:
        base_url: Base URL of the agent container, e.g.
                  ``"http://analyst:9000"`` or ``"http://localhost:9000"``.
        name:     Human-readable name for this client.  Defaults to
                  *base_url* when not provided.

    Example::

        from ninetrix import AgentClient

        client = AgentClient("http://analyst:9000", name="analyst")
        result = client.run("What is the P/E ratio for AAPL?")
        print(result.output)
    """

    def __init__(self, base_url: str, *, name: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.name = name or base_url

    # ------------------------------------------------------------------
    # AgentProtocol implementation
    # ------------------------------------------------------------------

    def run(
        self,
        message: str,
        *,
        thread_id: Optional[str] = None,
    ) -> AgentResult:
        """Run synchronously using a background thread + event loop.

        Safe in Jupyter notebooks and FastAPI handlers where a loop is already
        running.

        Args:
            message:   User message.
            thread_id: Conversation ID for multi-turn resumption.

        Returns:
            :class:`~ninetrix.AgentResult`

        Raises:
            :class:`~ninetrix.ProviderError`: HTTP 4xx / 5xx from the agent.
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
            :class:`~ninetrix.ProviderError`: HTTP 4xx / 5xx from the agent.
        """
        headers: dict[str, str] = {}
        tenant = get_tenant()
        if tenant and tenant.api_key:
            headers["Authorization"] = f"Bearer {tenant.api_key}"

        url = f"{self.base_url}/invoke"
        payload: dict[str, Any] = {"message": message}
        if thread_id:
            payload["thread_id"] = thread_id

        try:
            client = get_http_client()
            resp = await client.post(url, json=payload, headers=headers, timeout=300.0)
        except Exception as exc:
            raise ProviderError(
                f"AgentClient '{self.name}' network error.\n"
                f"  URL: {url}\n"
                f"  Why: {type(exc).__name__}: {exc}\n"
                "  Fix: check that the agent container is running and reachable."
            ) from exc

        if resp.status_code >= 400:
            raise ProviderError(
                f"AgentClient '{self.name}' returned HTTP {resp.status_code}.\n"
                f"  URL: {url}\n"
                f"  Response: {resp.text[:300]}\n"
                "  Fix: check the agent container logs for errors."
            )

        try:
            data = resp.json()
        except Exception as exc:
            raise ProviderError(
                f"AgentClient '{self.name}' returned non-JSON response.\n"
                f"  URL: {url}\n"
                f"  Body: {resp.text[:200]}\n"
                f"  Why: {exc}"
            ) from exc

        # Normalise — the /invoke contract returns AgentResult fields
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
        """Streaming stub — SSE support planned for a future PR.

        Raises:
            :class:`~ninetrix.ProviderError`: always, until SSE is implemented.
        """
        raise ProviderError(
            f"AgentClient '{self.name}' does not support streaming yet.\n"
            "  Why: SSE support is planned for a future release.\n"
            "  Fix: use arun() for a single-shot async call instead."
        )
        # Required to make this an async generator function (never reached)
        yield  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"AgentClient(base_url={self.base_url!r}, name={self.name!r})"
