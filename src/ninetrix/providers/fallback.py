"""
ninetrix.providers.fallback
===========================

``FallbackProviderAdapter`` — wraps a chain of ``LLMProviderAdapter`` instances.
Tries each in order on ``ProviderError``.  Raises only when the chain is exhausted.

Usage::

    from ninetrix.providers.anthropic import AnthropicAdapter
    from ninetrix.providers.openai import OpenAIAdapter
    from ninetrix.providers.fallback import FallbackConfig, FallbackProviderAdapter

    primary = AnthropicAdapter(api_key="sk-ant-...", model="claude-sonnet-4-6")
    backup  = OpenAIAdapter(api_key="sk-...",       model="gpt-4o")

    adapter = FallbackProviderAdapter(
        primary=primary,
        fallbacks=[
            (backup, FallbackConfig("openai", "gpt-4o")),
        ],
    )

    # Used the same way as any single adapter:
    result = await adapter.complete(messages, tools)

``FallbackConfig`` selects which error codes/types should trigger a fallback.
By default ``[429, 500, 503, 529]`` and ``["overloaded", "rate_limit"]`` do.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import AsyncIterator

from ninetrix._internals.types import (
    Attachment,
    LLMChunk,
    LLMProviderAdapter,
    LLMResponse,
    ProviderConfig,
    ProviderError,
)

logger = logging.getLogger(__name__)


@dataclass
class FallbackConfig:
    """
    Describes one fallback provider in a ``FallbackProviderAdapter`` chain.

    Attributes:
        provider:         Human-readable provider name (used in log messages).
        model:            Model name used by this fallback adapter.
        on_status_codes:  HTTP status codes that should trigger fallback.
        on_error_types:   Substrings of the exception message that trigger fallback.
    """

    provider: str
    model: str
    on_status_codes: list[int] = field(
        default_factory=lambda: [429, 500, 503, 529]
    )
    on_error_types: list[str] = field(
        default_factory=lambda: ["overloaded", "rate_limit"]
    )


class FallbackProviderAdapter(LLMProviderAdapter):
    """
    Wraps a primary adapter and a list of fallback (adapter, config) pairs.

    On ``ProviderError`` from the primary, it checks whether the error matches
    the *next* fallback's ``FallbackConfig`` and tries it.  If the error does
    not match, the error is re-raised immediately (e.g. credential errors should
    not trigger a fallback to the same provider with the same key).

    The primary always attempts first; ``FallbackConfig`` for the primary is
    implicitly "try the next one on any ProviderError".
    """

    provider_name = "fallback"

    def __init__(
        self,
        primary: LLMProviderAdapter,
        fallbacks: list[tuple[LLMProviderAdapter, FallbackConfig]],
    ) -> None:
        self._primary = primary
        self._fallbacks = fallbacks

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        attachments: list[Attachment] | None = None,
        output_schema: dict | None = None,
        config: ProviderConfig | None = None,
    ) -> LLMResponse:
        last_exc: ProviderError | None = None

        # Build a flat list: (adapter, config_or_None) — None = primary
        chain: list[tuple[LLMProviderAdapter, FallbackConfig | None]] = [
            (self._primary, None)
        ] + list(self._fallbacks)

        for i, (adapter, fb_cfg) in enumerate(chain):
            try:
                return await adapter.complete(
                    messages,
                    tools,
                    attachments=attachments,
                    output_schema=output_schema,
                    config=config,
                )
            except ProviderError as exc:
                last_exc = exc
                # Determine whether we should try the next adapter
                next_cfg = chain[i + 1][1] if i + 1 < len(chain) else None
                if not self._should_fallback(exc, next_cfg):
                    raise
                provider_name = getattr(adapter, "provider_name", str(adapter))
                logger.warning(
                    "Provider fallback triggered",
                    extra={
                        "from_provider": provider_name,
                        "error": str(exc),
                        "status_code": exc.status_code,
                    },
                )
                continue

        if last_exc is not None:
            raise last_exc
        raise ProviderError(
            "FallbackProviderAdapter: chain exhausted with no response.",
            provider="fallback",
        )

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        attachments: list[Attachment] | None = None,
        config: ProviderConfig | None = None,
    ) -> AsyncIterator[LLMChunk]:
        """
        Streaming fallback: delegates to the primary stream.
        Fallback on stream errors is not supported — use complete() for reliability.
        """
        return await self._primary.stream(
            messages,
            tools,
            attachments=attachments,
            config=config,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _should_fallback(
        self, exc: ProviderError, next_cfg: FallbackConfig | None
    ) -> bool:
        """
        Return True if this error should trigger a fallback to the next adapter.

        If ``next_cfg`` is None (no more adapters), always return False.
        """
        if next_cfg is None:
            return False
        status_match = (
            exc.status_code is not None
            and exc.status_code in next_cfg.on_status_codes
        )
        msg_match = any(t in str(exc).lower() for t in next_cfg.on_error_types)
        return status_match or msg_match
