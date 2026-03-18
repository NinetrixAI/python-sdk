"""
ninetrix.providers.litellm
==========================

``LiteLLMAdapter`` — wraps ``litellm.acompletion()`` as a generic fallback.

Install::

    pip install litellm

Any model string supported by LiteLLM works (``openai/gpt-4o``,
``anthropic/claude-sonnet-4-6``, ``ollama/llama3``, etc.).

Structured output
-----------------
When ``output_schema`` is passed, the adapter appends a system-level
instruction asking the model to respond with valid JSON matching the schema.
LiteLLM's underlying provider may have native JSON-mode support (e.g. OpenAI
via LiteLLM) — the system-prompt approach ensures compatibility across
all models.

Attachments
-----------
``ImageAttachment`` objects are injected as ``image_url`` content parts
(same format as OpenAI).  ``DocumentAttachment`` is embedded as text.
"""

from __future__ import annotations

import json
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

_JSON_SYSTEM_SUFFIX = (
    "\n\nIMPORTANT: Respond ONLY with valid JSON. "
    "Do not include any text outside the JSON object."
)


class LiteLLMAdapter(LLMProviderAdapter):
    """
    Generic provider adapter via LiteLLM.

    Accepts any model string that LiteLLM supports.  All exceptions are
    caught and re-raised as ``CredentialError`` or ``ProviderError``.
    """

    provider_name = "litellm"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        _acompletion: Any = None,  # injectable for tests
    ) -> None:
        try:
            import litellm as _sdk
        except ImportError as exc:
            raise ConfigurationError(
                "The 'litellm' package is required to use LiteLLMAdapter.\n"
                "  Fix: pip install litellm"
            ) from exc
        self._sdk = _sdk
        self._model = model
        self._api_key = api_key
        self._acompletion = _acompletion or _sdk.acompletion

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
        msgs = self._build_messages(messages, attachments, output_schema)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": msgs,
            "temperature": cfg.temperature,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
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
            response = await self._acompletion(**kwargs)
            return self._normalize(response)
        except Exception as exc:
            self._wrap_exception(exc)
            raise  # unreachable

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
        msgs = self._build_messages(messages, attachments, output_schema=None)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": msgs,
            "temperature": cfg.temperature,
            "stream": True,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if cfg.max_tokens is not None:
            kwargs["max_tokens"] = cfg.max_tokens
        if cfg.top_p is not None:
            kwargs["top_p"] = cfg.top_p
        if cfg.stop_sequences:
            kwargs["stop"] = cfg.stop_sequences
        if tools:
            kwargs["tools"] = self._build_tools(tools)

        try:
            async for chunk in await self._acompletion(**kwargs):
                normalized = self._normalize_chunk(chunk)
                if normalized is not None:
                    yield normalized
        except Exception as exc:
            self._wrap_exception(exc)
            raise  # unreachable

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_tools(self, tools: list[dict]) -> list[dict]:
        """Convert ninetrix tool defs to OpenAI-compatible function format."""
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
        self,
        messages: list[dict],
        attachments: list[Attachment] | None,
        output_schema: dict | None,
    ) -> list[dict]:
        """Inject attachments into last user message + schema instruction if needed."""
        msgs = list(messages)

        # Append JSON instruction to first system message (or prepend one)
        if output_schema is not None:
            schema_str = json.dumps(output_schema, indent=2)
            instruction = (
                f"{_JSON_SYSTEM_SUFFIX}\n\nExpected JSON schema:\n{schema_str}"
            )
            injected = False
            for msg in msgs:
                if msg.get("role") == "system":
                    msg = {**msg, "content": msg["content"] + instruction}
                    injected = True
                    break
            if not injected:
                msgs.insert(0, {"role": "system", "content": instruction.lstrip()})

        if not attachments:
            return msgs

        # Inject attachment parts into last user message
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
                return {"type": "image_url", "image_url": {"url": att.url}}
            data_uri = f"data:{att.mime_type};base64,{att.to_base64()}"
            return {"type": "image_url", "image_url": {"url": data_uri}}
        # DocumentAttachment
        data_uri = f"data:{att.mime_type};base64,{att.to_base64()}"
        return {"type": "text", "text": f"[Document: {att.filename}]\n{data_uri}"}

    def _normalize(self, response: Any) -> LLMResponse:
        """Convert LiteLLM response (OpenAI-compatible) to LLMResponse."""
        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        tool_calls: list[ToolCall] = []

        if getattr(message, "tool_calls", None):
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                )

        usage = getattr(response, "usage", None)
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            stop_reason=choice.finish_reason or "",
            raw=response,
        )

    def _normalize_chunk(self, chunk: Any) -> LLMChunk | None:
        """Convert a LiteLLM streaming chunk to LLMChunk, or None to skip."""
        if not getattr(chunk, "choices", None):
            return None
        delta = chunk.choices[0].delta
        finish_reason = chunk.choices[0].finish_reason

        if finish_reason == "stop":
            return LLMChunk(type="done")
        if delta.content:
            return LLMChunk(type="token", content=delta.content)
        if getattr(delta, "tool_calls", None):
            tc_delta = delta.tool_calls[0]
            return LLMChunk(
                type="tool_call_delta",
                tool_name=getattr(tc_delta.function, "name", None),
                tool_call_id=tc_delta.id or "",
                tool_arg_delta=getattr(tc_delta.function, "arguments", None),
            )
        return None

    def _wrap_exception(self, exc: Exception) -> None:
        """Classify and re-raise a litellm exception as NinetrixError."""
        if isinstance(exc, (CredentialError, ProviderError)):
            raise

        exc_type = type(exc).__name__
        exc_str = str(exc).lower()

        # LiteLLM exception type names are predictable
        if "AuthenticationError" in exc_type or "invalid api key" in exc_str:
            raise CredentialError(
                "LiteLLM authentication failed.\n"
                f"  Why: {exc}\n"
                "  Fix: Check the API key for your configured provider."
            ) from exc
        if "RateLimitError" in exc_type or "rate limit" in exc_str:
            raise ProviderError(
                "LiteLLM rate limit hit.\n"
                f"  Why: {exc}\n"
                "  Fix: Reduce concurrency or add RetryMiddleware with backoff.",
                status_code=429,
                provider=self._model.split("/")[0] if "/" in self._model else "litellm",
                retryable=True,
            ) from exc
        if "ServiceUnavailableError" in exc_type or "503" in exc_str:
            raise ProviderError(
                "LiteLLM provider is temporarily unavailable.\n"
                f"  Why: {exc}\n"
                "  Fix: Retry after a brief delay.",
                status_code=503,
                provider="litellm",
                retryable=True,
            ) from exc

        status_code: int | None = getattr(exc, "status_code", None)
        raise ProviderError(
            f"Unexpected error from LiteLLM provider.\n"
            f"  Why: {type(exc).__name__}: {exc}\n"
            "  Fix: Report this at github.com/Ninetrix-ai/ninetrix/issues",
            status_code=status_code,
            provider="litellm",
            retryable=(status_code in _RETRYABLE_STATUS) if status_code else False,
        ) from exc
