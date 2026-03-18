"""
ninetrix.providers.google
=========================

``GoogleAdapter`` — wraps the ``google-generativeai`` Python SDK.

Install::

    pip install google-generativeai

Structured output
-----------------
When ``output_schema`` is passed, the adapter sets
``generation_config.response_mime_type="application/json"`` and passes the
sanitized schema as ``response_schema``.  Raw JSON is returned in
``LLMResponse.content``; the runner owns the parse-and-retry loop.

Schema sanitization
-------------------
Gemini's schema validator is stricter than JSON Schema and rejects several
standard fields (``$schema``, ``$defs``, ``definitions``, ``$ref``,
``additionalProperties``, ``default``).  ``_sanitize_schema_for_gemini()``
strips these recursively before any tool or output-schema call.

Attachments
-----------
``ImageAttachment`` objects are embedded as ``inline_data`` parts in the
last user message.  ``DocumentAttachment`` is not currently supported by
the SDK; it is passed as a text block with the filename.
"""

from __future__ import annotations

import copy
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

# Fields Gemini rejects in JSON schemas
_GEMINI_UNSUPPORTED_FIELDS = frozenset(
    {
        "$schema",
        "$defs",
        "definitions",
        "$ref",
        "additionalProperties",
        "default",
        "$id",
        "$comment",
        "examples",
    }
)


def _sanitize_schema_for_gemini(schema: dict) -> dict:
    """
    Recursively strip fields that Gemini's schema validator rejects.

    Returns a new dict — the original is not modified.
    """
    result = {}
    for key, value in schema.items():
        if key in _GEMINI_UNSUPPORTED_FIELDS:
            continue
        if isinstance(value, dict):
            result[key] = _sanitize_schema_for_gemini(value)
        elif isinstance(value, list):
            result[key] = [
                _sanitize_schema_for_gemini(v) if isinstance(v, dict) else v
                for v in value
            ]
        else:
            result[key] = value
    return result


class GoogleAdapter(LLMProviderAdapter):
    """
    Adapter for the Google Generative AI API (Gemini).

    All ``google.generativeai`` SDK exceptions are caught and re-raised as
    ``CredentialError`` or ``ProviderError`` — nothing raw leaks out.
    """

    provider_name = "google"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-pro",
        *,
        _model_instance: Any = None,  # injectable for tests
    ) -> None:
        try:
            import google.generativeai as _sdk
        except ImportError as exc:
            raise ConfigurationError(
                "The 'google-generativeai' package is required to use GoogleAdapter.\n"
                "  Fix: pip install google-generativeai"
            ) from exc
        self._sdk = _sdk
        self._model_name = model
        self._api_key = api_key
        if _model_instance is not None:
            self._model_instance = _model_instance
        else:
            _sdk.configure(api_key=api_key)
            self._model_instance = _sdk.GenerativeModel(model)

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
        contents = self._build_contents(messages, attachments)

        gen_config: dict[str, Any] = {
            "temperature": cfg.temperature,
        }
        if cfg.max_tokens is not None:
            gen_config["max_output_tokens"] = cfg.max_tokens
        if cfg.top_p is not None:
            gen_config["top_p"] = cfg.top_p
        if cfg.stop_sequences:
            gen_config["stop_sequences"] = cfg.stop_sequences
        if output_schema is not None:
            gen_config["response_mime_type"] = "application/json"
            gen_config["response_schema"] = _sanitize_schema_for_gemini(
                copy.deepcopy(output_schema)
            )

        gemini_tools = self._build_tools(tools) if tools else None

        try:
            response = await self._model_instance.generate_content_async(
                contents,
                generation_config=gen_config,
                **({"tools": gemini_tools} if gemini_tools else {}),
            )
            return self._normalize(response)
        except Exception as exc:
            self._wrap_exception(exc)
            raise  # unreachable — _wrap_exception always raises

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
        contents = self._build_contents(messages, attachments)

        gen_config: dict[str, Any] = {"temperature": cfg.temperature}
        if cfg.max_tokens is not None:
            gen_config["max_output_tokens"] = cfg.max_tokens
        if cfg.top_p is not None:
            gen_config["top_p"] = cfg.top_p
        if cfg.stop_sequences:
            gen_config["stop_sequences"] = cfg.stop_sequences

        gemini_tools = self._build_tools(tools) if tools else None

        try:
            async for chunk in await self._model_instance.generate_content_async(
                contents,
                generation_config=gen_config,
                stream=True,
                **({"tools": gemini_tools} if gemini_tools else {}),
            ):
                normalized = self._normalize_chunk(chunk)
                if normalized is not None:
                    yield normalized
        except Exception as exc:
            self._wrap_exception(exc)
            raise  # unreachable

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_tools(self, tools: list[dict]) -> list[Any]:
        """Convert ninetrix tool defs to Gemini FunctionDeclaration list."""
        declarations = []
        for t in tools:
            schema = _sanitize_schema_for_gemini(
                t.get("parameters", {"type": "object", "properties": {}})
            )
            declarations.append(
                self._sdk.protos.FunctionDeclaration(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters=schema,
                )
            )
        return [self._sdk.Tool(function_declarations=declarations)]

    def _build_contents(
        self, messages: list[dict], attachments: list[Attachment] | None
    ) -> list[dict]:
        """
        Convert OpenAI-style message list to Gemini contents format.
        Gemini uses ``model`` / ``user`` roles (not ``assistant``).
        System messages are prepended as a user turn.
        """
        contents: list[dict] = []
        system_parts: list[dict] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_parts.append({"text": content})
                continue

            gemini_role = "model" if role == "assistant" else "user"
            parts: list[dict] = []
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for block in content:
                    if block.get("type") == "text":
                        parts.append({"text": block["text"]})

            contents.append({"role": gemini_role, "parts": parts})

        # Inject system prompt as first user message if present
        if system_parts:
            system_turn = {"role": "user", "parts": system_parts}
            contents.insert(0, system_turn)

        # Inject attachments into last user message
        if attachments:
            for i in range(len(contents) - 1, -1, -1):
                if contents[i]["role"] == "user":
                    for att in attachments:
                        contents[i]["parts"].append(
                            self._format_attachment(att)
                        )
                    break

        return contents

    def _format_attachment(self, att: Attachment) -> dict:
        if isinstance(att, ImageAttachment):
            if att.data is not None:
                return {
                    "inline_data": {
                        "mime_type": att.mime_type,
                        "data": att.to_base64(),
                    }
                }
            # URL-based — embed as text reference (Gemini doesn't support URL images directly)
            return {"text": f"[Image: {att.url}]"}
        # DocumentAttachment — not natively supported, embed as text
        return {"text": f"[Document: {att.filename} (base64 data omitted)]"}

    def _normalize(self, response: Any) -> LLMResponse:
        """Convert Gemini GenerateContentResponse to LLMResponse."""
        content_text = ""
        tool_calls: list[ToolCall] = []
        input_tokens = 0
        output_tokens = 0

        if response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
            output_tokens = getattr(
                response.usage_metadata, "candidates_token_count", 0
            )

        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "text"):
                    content_text += part.text
                elif hasattr(part, "function_call"):
                    fc = part.function_call
                    args = dict(fc.args) if fc.args else {}
                    tool_calls.append(
                        ToolCall(id=fc.name, name=fc.name, arguments=args)
                    )

        stop_reason = ""
        if response.candidates:
            finish = response.candidates[0].finish_reason
            stop_reason = str(finish) if finish is not None else ""

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
            raw=response,
        )

    def _normalize_chunk(self, chunk: Any) -> LLMChunk | None:
        """Convert a Gemini streaming chunk to LLMChunk, or None to skip."""
        try:
            if chunk.text:
                return LLMChunk(type="token", content=chunk.text)
        except Exception:
            pass

        # Check for function call parts
        for candidate in getattr(chunk, "candidates", []):
            for part in candidate.content.parts:
                if hasattr(part, "function_call"):
                    return LLMChunk(
                        type="tool_call_delta",
                        tool_name=part.function_call.name,
                    )

        # Check done
        if getattr(chunk, "done", False):
            return LLMChunk(type="done")
        return None

    def _wrap_exception(self, exc: Exception) -> None:
        """Classify and re-raise a google.generativeai exception as NinetrixError."""
        exc_type = type(exc).__name__
        exc_module = type(exc).__module__

        if (
            "PermissionDenied" in exc_type
            or "Unauthenticated" in exc_type
            or ("google" in exc_module and "permission" in str(exc).lower())
        ):
            raise CredentialError(
                "Google API key is invalid or lacks permission.\n"
                f"  Why: {exc}\n"
                "  Fix: Check GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS."
            ) from exc
        if "ResourceExhausted" in exc_type or "quota" in str(exc).lower():
            raise ProviderError(
                "Google API quota exceeded.\n"
                f"  Why: {exc}\n"
                "  Fix: Reduce concurrency or check your Google Cloud quota.",
                status_code=429,
                provider="google",
                retryable=True,
            ) from exc
        if "ServiceUnavailable" in exc_type or "InternalServerError" in exc_type:
            raise ProviderError(
                "Google API is temporarily unavailable.\n"
                f"  Why: {exc}\n"
                "  Fix: Retry after a brief delay.",
                status_code=503,
                provider="google",
                retryable=True,
            ) from exc
        if isinstance(exc, (CredentialError, ProviderError)):
            raise

        raise ProviderError(
            f"Unexpected error from Google provider.\n"
            f"  Why: {type(exc).__name__}: {exc}\n"
            "  Fix: Report this at github.com/Ninetrix-ai/ninetrix/issues",
            provider="google",
        ) from exc
