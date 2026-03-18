"""
ninetrix.providers.anthropic
============================

``AnthropicAdapter`` — wraps the ``anthropic`` Python SDK.

Install::

    pip install anthropic

Structured output
-----------------
When ``output_schema`` is passed to ``complete()``, the adapter injects a
hidden tool ``__structured_output__`` with the schema as its ``input_schema``
and forces ``tool_choice`` to use it.  The runner receives raw JSON in
``LLMResponse.content`` and owns the parse-and-retry loop.

Attachments
-----------
``ImageAttachment`` and ``DocumentAttachment`` objects are appended to the
last user message as Anthropic content blocks (``image`` / ``document``).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, AsyncIterator

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

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {429, 500, 502, 503, 529}
_SCHEMA_TOOL_NAME = "__structured_output__"


class AnthropicAdapter(LLMProviderAdapter):
    """
    Adapter for the Anthropic Messages API.

    All ``anthropic`` SDK exceptions are caught and re-raised as
    ``CredentialError`` or ``ProviderError`` — nothing raw leaks out.
    """

    provider_name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        *,
        _client: Any = None,  # injectable for tests
    ) -> None:
        try:
            import anthropic as _sdk
        except ImportError as exc:
            raise ConfigurationError(
                "The 'anthropic' package is required to use AnthropicAdapter.\n"
                "  Fix: pip install anthropic"
            ) from exc
        self._sdk = _sdk
        self._model = model
        self._client = _client or _sdk.AsyncAnthropic(api_key=api_key)

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
        anthropic_tools, schema_tool_name = self._build_tools(tools, output_schema)
        msgs = self._build_messages(messages, attachments)

        # Anthropic requires system prompt as top-level param, not in messages array
        system_parts = [m["content"] for m in msgs if m.get("role") == "system"]
        msgs = [m for m in msgs if m.get("role") != "system"]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": msgs,
            "max_tokens": cfg.max_tokens or 4096,
            "temperature": cfg.temperature,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        if cfg.top_p is not None:
            kwargs["top_p"] = cfg.top_p
        if cfg.stop_sequences:
            kwargs["stop_sequences"] = cfg.stop_sequences
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        if schema_tool_name is not None:
            kwargs["tool_choice"] = {"type": "tool", "name": schema_tool_name}

        try:
            response = await self._client.messages.create(**kwargs)
            return self._normalize(response, output_schema=output_schema)
        except self._sdk.AuthenticationError as exc:
            raise CredentialError(
                "Anthropic API key is invalid.\n"
                f"  Why: {exc}\n"
                "  Fix: Check ANTHROPIC_API_KEY or run: ninetrix auth login"
            ) from exc
        except self._sdk.RateLimitError as exc:
            raise ProviderError(
                "Anthropic rate limit hit.\n"
                f"  Why: {exc}\n"
                "  Fix: Reduce concurrency or add RetryMiddleware with backoff.",
                status_code=429,
                provider="anthropic",
                retryable=True,
            ) from exc
        except self._sdk.APIStatusError as exc:
            raise ProviderError(
                f"Anthropic API error (status {exc.status_code}).\n"
                f"  Why: {exc.message}\n"
                "  Fix: Check the Anthropic status page or reduce request complexity.",
                status_code=exc.status_code,
                provider="anthropic",
                retryable=exc.status_code in _RETRYABLE_STATUS,
            ) from exc
        except (CredentialError, ProviderError):
            raise
        except Exception as exc:
            raise ProviderError(
                f"Unexpected error from Anthropic provider.\n"
                f"  Why: {type(exc).__name__}: {exc}\n"
                "  Fix: Report this at github.com/Ninetrix-ai/ninetrix/issues",
                provider="anthropic",
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
        anthropic_tools, _ = self._build_tools(tools, output_schema=None)
        msgs = self._build_messages(messages, attachments)

        system_parts = [m["content"] for m in msgs if m.get("role") == "system"]
        msgs = [m for m in msgs if m.get("role") != "system"]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": msgs,
            "max_tokens": cfg.max_tokens or 4096,
            "temperature": cfg.temperature,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        if cfg.top_p is not None:
            kwargs["top_p"] = cfg.top_p
        if cfg.stop_sequences:
            kwargs["stop_sequences"] = cfg.stop_sequences
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        try:
            async with self._client.messages.stream(**kwargs) as stream_ctx:
                async for event in stream_ctx:
                    chunk = self._normalize_stream_event(event)
                    if chunk is not None:
                        yield chunk
        except self._sdk.AuthenticationError as exc:
            raise CredentialError(
                "Anthropic API key is invalid.\n"
                f"  Why: {exc}\n"
                "  Fix: Check ANTHROPIC_API_KEY or run: ninetrix auth login"
            ) from exc
        except self._sdk.RateLimitError as exc:
            raise ProviderError(
                "Anthropic rate limit hit.\n"
                f"  Why: {exc}\n"
                "  Fix: Reduce concurrency or add RetryMiddleware with backoff.",
                status_code=429,
                provider="anthropic",
                retryable=True,
            ) from exc
        except self._sdk.APIStatusError as exc:
            raise ProviderError(
                f"Anthropic API error (status {exc.status_code}).\n"
                f"  Why: {exc.message}\n"
                "  Fix: Check Anthropic status page.",
                status_code=exc.status_code,
                provider="anthropic",
                retryable=exc.status_code in _RETRYABLE_STATUS,
            ) from exc
        except (CredentialError, ProviderError):
            raise
        except Exception as exc:
            raise ProviderError(
                f"Unexpected streaming error from Anthropic.\n"
                f"  Why: {type(exc).__name__}: {exc}\n"
                "  Fix: Report this at github.com/Ninetrix-ai/ninetrix/issues",
                provider="anthropic",
            ) from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_tools(
        self, tools: list[dict], output_schema: dict | None
    ) -> tuple[list[dict], str | None]:
        """Convert ninetrix tool defs to Anthropic format + optional schema tool."""
        anthropic_tools = []
        for t in tools:
            # Unwrap OpenAI-compatible {"type": "function", "function": {...}} wrapper
            fn = t.get("function", t)
            anthropic_tools.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            })
        schema_tool_name: str | None = None
        if output_schema is not None:
            schema_tool_name = _SCHEMA_TOOL_NAME
            anthropic_tools.append(
                {
                    "name": schema_tool_name,
                    "description": (
                        "Return your response as structured JSON matching this schema."
                    ),
                    "input_schema": output_schema,
                }
            )
        return anthropic_tools, schema_tool_name

    def _build_messages(
        self, messages: list[dict], attachments: list[Attachment] | None
    ) -> list[dict]:
        """Convert OpenAI-compatible message history to Anthropic format.

        Handles:
        - Assistant messages with tool_calls → content blocks with type=tool_use
        - role=tool results → role=user messages with type=tool_result blocks
        - Attachment injection into the last user message
        """
        msgs: list[dict] = []
        pending_tool_results: list[dict] = []

        for m in messages:
            role = m.get("role")

            if role == "system":
                # Passed separately as top-level system param — keep as-is for extraction
                msgs.append(m)

            elif role == "tool":
                # Buffer tool results; they'll be emitted as a single user message
                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id", ""),
                    "content": m.get("content", ""),
                })

            else:
                # Flush any buffered tool results before the next assistant/user turn
                if pending_tool_results:
                    msgs.append({"role": "user", "content": pending_tool_results})
                    pending_tool_results = []

                if role == "assistant":
                    tool_calls = m.get("tool_calls") or []
                    if tool_calls:
                        # Build content blocks: optional text + tool_use blocks
                        content: list[dict] = []
                        text = m.get("content") or ""
                        if text:
                            content.append({"type": "text", "text": text})
                        for tc in tool_calls:
                            fn = tc.get("function", {})
                            raw_args = fn.get("arguments", "{}")
                            try:
                                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                            except (json.JSONDecodeError, TypeError):
                                args = {}
                            content.append({
                                "type": "tool_use",
                                "id": tc.get("id", ""),
                                "name": fn.get("name", ""),
                                "input": args,
                            })
                        msgs.append({"role": "assistant", "content": content})
                    else:
                        msgs.append({"role": "assistant", "content": m.get("content", "")})

                else:
                    # user message — handle attachments below
                    msgs.append(m)

        # Flush any trailing tool results
        if pending_tool_results:
            msgs.append({"role": "user", "content": pending_tool_results})

        # Inject attachments into the last user message
        if attachments:
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i].get("role") == "user":
                    existing = msgs[i].get("content", "")
                    content_blocks: list[dict] = []
                    if isinstance(existing, str):
                        content_blocks.append({"type": "text", "text": existing})
                    elif isinstance(existing, list):
                        content_blocks.extend(existing)
                    for att in attachments:
                        content_blocks.append(self._format_attachment(att))
                    msgs[i] = {**msgs[i], "content": content_blocks}
                    break

        return msgs

    def _format_attachment(self, att: Attachment) -> dict:
        if isinstance(att, ImageAttachment):
            if att.url is not None:
                return {"type": "image", "source": {"type": "url", "url": att.url}}
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": att.mime_type,
                    "data": att.to_base64(),
                },
            }
        # DocumentAttachment
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": att.mime_type,
                "data": att.to_base64(),
            },
            "title": att.filename,
        }

    def _normalize(
        self, response: Any, *, output_schema: dict | None = None
    ) -> LLMResponse:
        """Convert Anthropic Messages response to LLMResponse."""
        content_text = ""
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                if output_schema is not None and block.name == _SCHEMA_TOOL_NAME:
                    # Structured output: serialize tool input back to JSON string
                    content_text = json.dumps(block.input)
                else:
                    tool_calls.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=(
                                block.input
                                if isinstance(block.input, dict)
                                else {}
                            ),
                        )
                    )

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason or "",
            raw=response,
        )

    def _normalize_stream_event(self, event: Any) -> LLMChunk | None:
        """Convert an Anthropic streaming event to LLMChunk, or None to skip."""
        event_type = getattr(event, "type", None)
        if event_type == "content_block_delta":
            delta = event.delta
            if delta.type == "text_delta":
                return LLMChunk(type="token", content=delta.text)
            if delta.type == "input_json_delta":
                return LLMChunk(
                    type="tool_call_delta",
                    tool_arg_delta=delta.partial_json,
                    tool_call_id=str(getattr(event, "index", "")),
                )
        if event_type == "content_block_start":
            block = event.content_block
            if block.type == "tool_use":
                return LLMChunk(
                    type="tool_call_delta",
                    tool_name=block.name,
                    tool_call_id=block.id,
                )
        if event_type == "message_stop":
            return LLMChunk(type="done")
        return None
