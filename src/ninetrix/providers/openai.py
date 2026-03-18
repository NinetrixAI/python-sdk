"""
ninetrix.providers.openai
=========================

``OpenAIAdapter`` — wraps the ``openai`` Python SDK.

Install::

    pip install openai

Structured output
-----------------
When ``output_schema`` is passed, the adapter uses OpenAI's native
``response_format={"type": "json_schema", "json_schema": {...}}`` to
constrain the model's output.  Raw JSON is returned in
``LLMResponse.content``; the runner owns the parse-and-retry loop.

Attachments
-----------
``ImageAttachment`` objects are passed as ``image_url`` content parts
in the last user message.  ``DocumentAttachment`` is embedded as a
``text`` content part (base64 data URI).
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, AsyncIterator

from ninetrix._internals.types import (
    Attachment,
    ConfigurationError,
    CredentialError,
    DocumentAttachment,
    ImageAttachment,
    LLMChunk,
    LLMProviderAdapter,
    LLMResponse,
    ProviderConfig,
    ProviderError,
    ToolCall,
)

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {429, 500, 502, 503}


class OpenAIAdapter(LLMProviderAdapter):
    """
    Adapter for the OpenAI Chat Completions API (and compatible endpoints).

    All ``openai`` SDK exceptions are caught and re-raised as
    ``CredentialError`` or ``ProviderError`` — nothing raw leaks out.
    """

    provider_name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        *,
        base_url: str | None = None,
        _client: Any = None,  # injectable for tests
    ) -> None:
        try:
            import openai as _sdk
        except ImportError as exc:
            raise ConfigurationError(
                "The 'openai' package is required to use OpenAIAdapter.\n"
                "  Fix: pip install openai"
            ) from exc
        self._sdk = _sdk
        self._model = model
        self._client = _client or _sdk.AsyncOpenAI(
            api_key=api_key,
            **({"base_url": base_url} if base_url else {}),
        )

    # ------------------------------------------------------------------
    # Public Protocol methods
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        attachments: list[Attachment] | None = None,
        output_schema: dict | None = None,
        config: ProviderConfig | None = None,
    ) -> LLMResponse:
        cfg = config or ProviderConfig()
        msgs = self._build_messages(messages, attachments)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": msgs,
            "temperature": cfg.temperature,
        }
        if cfg.max_tokens is not None:
            kwargs["max_tokens"] = cfg.max_tokens
        if cfg.top_p is not None:
            kwargs["top_p"] = cfg.top_p
        if cfg.stop_sequences:
            kwargs["stop"] = cfg.stop_sequences
        if cfg.presence_penalty is not None:
            kwargs["presence_penalty"] = cfg.presence_penalty
        if cfg.frequency_penalty is not None:
            kwargs["frequency_penalty"] = cfg.frequency_penalty
        if tools:
            kwargs["tools"] = self._build_tools(tools)
        if output_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": True,
                    "schema": output_schema,
                },
            }

        try:
            response = await self._client.chat.completions.create(**kwargs)
            return self._normalize(response)
        except self._sdk.AuthenticationError as exc:
            raise CredentialError(
                "OpenAI API key is invalid.\n"
                f"  Why: {exc}\n"
                "  Fix: Check OPENAI_API_KEY or run: ninetrix auth login"
            ) from exc
        except self._sdk.RateLimitError as exc:
            raise ProviderError(
                "OpenAI rate limit hit.\n"
                f"  Why: {exc}\n"
                "  Fix: Reduce concurrency or add RetryMiddleware with backoff.",
                status_code=429,
                provider="openai",
                retryable=True,
            ) from exc
        except self._sdk.APIStatusError as exc:
            raise ProviderError(
                f"OpenAI API error (status {exc.status_code}).\n"
                f"  Why: {exc.message}\n"
                "  Fix: Check the OpenAI status page or reduce request complexity.",
                status_code=exc.status_code,
                provider="openai",
                retryable=exc.status_code in _RETRYABLE_STATUS,
            ) from exc
        except (CredentialError, ProviderError):
            raise
        except Exception as exc:
            raise ProviderError(
                f"Unexpected error from OpenAI provider.\n"
                f"  Why: {type(exc).__name__}: {exc}\n"
                "  Fix: Report this at github.com/Ninetrix-ai/ninetrix/issues",
                provider="openai",
            ) from exc

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        attachments: list[Attachment] | None = None,
        config: ProviderConfig | None = None,
    ) -> AsyncIterator[LLMChunk]:
        return self._stream_impl(messages, tools, attachments=attachments, config=config)

    async def _stream_impl(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        attachments: list[Attachment] | None = None,
        config: ProviderConfig | None = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        cfg = config or ProviderConfig()
        msgs = self._build_messages(messages, attachments)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": msgs,
            "temperature": cfg.temperature,
            "stream": True,
        }
        if cfg.max_tokens is not None:
            kwargs["max_tokens"] = cfg.max_tokens
        if cfg.top_p is not None:
            kwargs["top_p"] = cfg.top_p
        if cfg.stop_sequences:
            kwargs["stop"] = cfg.stop_sequences
        if cfg.presence_penalty is not None:
            kwargs["presence_penalty"] = cfg.presence_penalty
        if cfg.frequency_penalty is not None:
            kwargs["frequency_penalty"] = cfg.frequency_penalty
        if tools:
            kwargs["tools"] = self._build_tools(tools)

        try:
            async for chunk in await self._client.chat.completions.create(**kwargs):
                normalized = self._normalize_chunk(chunk)
                if normalized is not None:
                    yield normalized
        except self._sdk.AuthenticationError as exc:
            raise CredentialError(
                "OpenAI API key is invalid.\n"
                f"  Why: {exc}\n"
                "  Fix: Check OPENAI_API_KEY or run: ninetrix auth login"
            ) from exc
        except self._sdk.RateLimitError as exc:
            raise ProviderError(
                "OpenAI rate limit hit.\n"
                f"  Why: {exc}\n"
                "  Fix: Reduce concurrency or add RetryMiddleware with backoff.",
                status_code=429,
                provider="openai",
                retryable=True,
            ) from exc
        except self._sdk.APIStatusError as exc:
            raise ProviderError(
                f"OpenAI API error (status {exc.status_code}).\n"
                f"  Why: {exc.message}\n"
                "  Fix: Check OpenAI status page.",
                status_code=exc.status_code,
                provider="openai",
                retryable=exc.status_code in _RETRYABLE_STATUS,
            ) from exc
        except (CredentialError, ProviderError):
            raise
        except Exception as exc:
            raise ProviderError(
                f"Unexpected streaming error from OpenAI.\n"
                f"  Why: {type(exc).__name__}: {exc}\n"
                "  Fix: Report this at github.com/Ninetrix-ai/ninetrix/issues",
                provider="openai",
            ) from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_tools(self, tools: list[dict]) -> list[dict]:
        """Convert ninetrix tool defs to OpenAI function format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                },
            }
            for t in tools
        ]

    def _build_messages(
        self, messages: list[dict], attachments: list[Attachment] | None
    ) -> list[dict]:
        """Inject attachment content parts into the last user message."""
        if not attachments:
            return messages
        msgs = list(messages)
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                existing = msgs[i].get("content", "")
                content: list[dict] = []
                if isinstance(existing, str):
                    content.append({"type": "text", "text": existing})
                elif isinstance(existing, list):
                    content.extend(existing)
                for att in attachments:
                    content.append(self._format_attachment(att))
                msgs[i] = {**msgs[i], "content": content}
                break
        return msgs

    def _format_attachment(self, att: Attachment) -> dict:
        if isinstance(att, ImageAttachment):
            if att.url is not None:
                return {
                    "type": "image_url",
                    "image_url": {"url": att.url},
                }
            data_uri = f"data:{att.mime_type};base64,{att.to_base64()}"
            return {
                "type": "image_url",
                "image_url": {"url": data_uri},
            }
        # DocumentAttachment — embed as text (base64 data URI)
        data_uri = f"data:{att.mime_type};base64,{att.to_base64()}"
        return {
            "type": "text",
            "text": f"[Document: {att.filename}]\n{data_uri}",
        }

    def _normalize(self, response: Any) -> LLMResponse:
        """Convert OpenAI ChatCompletion to LLMResponse."""
        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        tool_calls: list[ToolCall] = []

        if message.tool_calls:
            import json as _json

            for tc in message.tool_calls:
                try:
                    args = _json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                )

        usage = response.usage
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            stop_reason=choice.finish_reason or "",
            raw=response,
        )

    def _normalize_chunk(self, chunk: Any) -> LLMChunk | None:
        """Convert an OpenAI streaming chunk to LLMChunk, or None to skip."""
        if not chunk.choices:
            return None
        delta = chunk.choices[0].delta
        finish_reason = chunk.choices[0].finish_reason

        if finish_reason == "stop":
            return LLMChunk(type="done")

        if delta.content:
            return LLMChunk(type="token", content=delta.content)

        if delta.tool_calls:
            tc_delta = delta.tool_calls[0]
            return LLMChunk(
                type="tool_call_delta",
                tool_name=getattr(tc_delta.function, "name", None),
                tool_call_id=tc_delta.id or "",
                tool_arg_delta=getattr(tc_delta.function, "arguments", None),
            )
        return None
