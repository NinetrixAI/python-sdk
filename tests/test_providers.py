"""
tests/test_providers.py
=======================

Unit tests for ninetrix.providers.*.

All provider SDK calls are mocked — no real API keys required.
Mocking strategy: inject fake SDK modules into sys.modules before import,
then use the _client / _model_instance / _acompletion injection hooks for
per-test control.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ninetrix._internals.types import (
    ConfigurationError,
    CredentialError,
    ImageAttachment,
    DocumentAttachment,
    LLMChunk,
    LLMProviderAdapter,
    LLMResponse,
    ProviderConfig,
    ProviderError,
    ToolCall,
)
from ninetrix.providers.base import LLMProviderAdapter as BaseProviderAdapter
from ninetrix.providers.fallback import FallbackConfig, FallbackProviderAdapter


# ===========================================================================
# Helpers — fake SDK factories
# ===========================================================================

def _make_anthropic_response(
    text: str = "Hello",
    tool_calls: list[dict] | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
    stop_reason: str = "end_turn",
) -> MagicMock:
    """Build a fake anthropic.Message-like object."""
    resp = MagicMock()
    resp.usage.input_tokens = input_tokens
    resp.usage.output_tokens = output_tokens
    resp.stop_reason = stop_reason

    blocks = []
    if text:
        block = MagicMock()
        block.type = "text"
        block.text = text
        blocks.append(block)
    for tc in (tool_calls or []):
        block = MagicMock()
        block.type = "tool_use"
        block.id = tc["id"]
        block.name = tc["name"]
        block.input = tc["arguments"]
        blocks.append(block)
    resp.content = blocks
    return resp


def _make_openai_response(
    text: str = "Hello",
    tool_calls: list[dict] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    finish_reason: str = "stop",
) -> MagicMock:
    """Build a fake openai.ChatCompletion-like object."""
    resp = MagicMock()
    message = MagicMock()
    message.content = text
    message.tool_calls = None

    if tool_calls:
        tcs = []
        for tc in tool_calls:
            tc_mock = MagicMock()
            tc_mock.id = tc["id"]
            tc_mock.function.name = tc["name"]
            tc_mock.function.arguments = json.dumps(tc["arguments"])
            tcs.append(tc_mock)
        message.tool_calls = tcs

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason
    resp.choices = [choice]
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


# ===========================================================================
# Fake SDK module setup helpers
# ===========================================================================

def _fake_anthropic_sdk() -> types.ModuleType:
    """Return a fake 'anthropic' module with the exception hierarchy."""
    sdk = types.ModuleType("anthropic")

    class APIError(Exception):
        def __init__(self, msg="", *, status_code=500, message=""):
            super().__init__(msg)
            self.status_code = status_code
            self.message = message or msg

    class APIStatusError(APIError):
        pass

    class AuthenticationError(APIStatusError):
        def __init__(self, msg=""):
            super().__init__(msg, status_code=401)

    class RateLimitError(APIStatusError):
        def __init__(self, msg=""):
            super().__init__(msg, status_code=429)

    sdk.APIError = APIError
    sdk.APIStatusError = APIStatusError
    sdk.AuthenticationError = AuthenticationError
    sdk.RateLimitError = RateLimitError

    client = MagicMock()
    sdk.AsyncAnthropic = MagicMock(return_value=client)
    return sdk


def _fake_openai_sdk() -> types.ModuleType:
    """Return a fake 'openai' module with the exception hierarchy."""
    sdk = types.ModuleType("openai")

    class APIError(Exception):
        pass

    class APIStatusError(APIError):
        def __init__(self, msg="", *, status_code=500, message=""):
            super().__init__(msg)
            self.status_code = status_code
            self.message = message or msg

    class AuthenticationError(APIStatusError):
        def __init__(self, msg=""):
            super().__init__(msg, status_code=401)

    class RateLimitError(APIStatusError):
        def __init__(self, msg=""):
            super().__init__(msg, status_code=429)

    sdk.APIError = APIError
    sdk.APIStatusError = APIStatusError
    sdk.AuthenticationError = AuthenticationError
    sdk.RateLimitError = RateLimitError

    client = MagicMock()
    sdk.AsyncOpenAI = MagicMock(return_value=client)
    return sdk


# ===========================================================================
# Provider base — protocol re-export
# ===========================================================================

class TestProviderBase:
    def test_base_exports_protocol(self):
        assert BaseProviderAdapter is LLMProviderAdapter

    def test_base_and_types_are_same(self):
        from ninetrix.providers.base import LLMProviderAdapter as A
        from ninetrix._internals.types import LLMProviderAdapter as B
        assert A is B


# ===========================================================================
# AnthropicAdapter
# ===========================================================================

class TestAnthropicAdapter:
    """Tests for AnthropicAdapter using a mocked anthropic SDK."""

    def _make_adapter(self, mock_client: Any) -> Any:
        """Create an AnthropicAdapter with an injected mock client."""
        fake_sdk = _fake_anthropic_sdk()
        with patch.dict(sys.modules, {"anthropic": fake_sdk}):
            from ninetrix.providers.anthropic import AnthropicAdapter
            adapter = AnthropicAdapter("sk-test", model="claude-test", _client=mock_client)
            adapter._sdk = fake_sdk  # so exception types match
        return adapter

    @pytest.mark.asyncio
    async def test_complete_returns_llm_response(self):
        client = MagicMock()
        resp = _make_anthropic_response(text="Paris")
        client.messages.create = AsyncMock(return_value=resp)
        adapter = self._make_adapter(client)

        result = await adapter.complete([{"role": "user", "content": "Capital?"}], [])

        assert isinstance(result, LLMResponse)
        assert result.content == "Paris"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_complete_with_tool_calls(self):
        client = MagicMock()
        resp = _make_anthropic_response(
            text="",
            tool_calls=[{"id": "tc1", "name": "search", "arguments": {"q": "Paris"}}],
        )
        client.messages.create = AsyncMock(return_value=resp)
        adapter = self._make_adapter(client)

        result = await adapter.complete([{"role": "user", "content": "Search"}], [])

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"q": "Paris"}

    @pytest.mark.asyncio
    async def test_complete_builds_correct_anthropic_tools(self):
        client = MagicMock()
        client.messages.create = AsyncMock(
            return_value=_make_anthropic_response("ok")
        )
        adapter = self._make_adapter(client)

        tools = [{"name": "calc", "description": "Add numbers",
                  "parameters": {"type": "object", "properties": {"a": {"type": "number"}}}}]
        await adapter.complete([{"role": "user", "content": "hi"}], tools)

        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs["tools"][0]["name"] == "calc"
        assert call_kwargs["tools"][0]["input_schema"]["properties"]["a"]["type"] == "number"

    @pytest.mark.asyncio
    async def test_structured_output_injects_schema_tool(self):
        client = MagicMock()
        # Return a tool_use block for __structured_output__
        resp = MagicMock()
        resp.usage.input_tokens = 5
        resp.usage.output_tokens = 3
        resp.stop_reason = "tool_use"
        block = MagicMock()
        block.type = "tool_use"
        block.name = "__structured_output__"
        block.input = {"answer": 42}
        block.id = "tc1"
        resp.content = [block]
        client.messages.create = AsyncMock(return_value=resp)
        adapter = self._make_adapter(client)

        schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}
        result = await adapter.complete(
            [{"role": "user", "content": "What is 6*7?"}], [], output_schema=schema
        )

        # The JSON of the tool input should be in content
        assert json.loads(result.content) == {"answer": 42}
        assert result.tool_calls == []

        call_kwargs = client.messages.create.call_args.kwargs
        tool_names = [t["name"] for t in call_kwargs["tools"]]
        assert "__structured_output__" in tool_names
        assert call_kwargs["tool_choice"]["name"] == "__structured_output__"

    @pytest.mark.asyncio
    async def test_complete_with_image_attachment_url(self):
        client = MagicMock()
        client.messages.create = AsyncMock(
            return_value=_make_anthropic_response("ok")
        )
        adapter = self._make_adapter(client)

        att = ImageAttachment(url="https://example.com/img.png")
        await adapter.complete(
            [{"role": "user", "content": "Describe this"}], [], attachments=[att]
        )

        call_kwargs = client.messages.create.call_args.kwargs
        last_msg = call_kwargs["messages"][-1]
        content = last_msg["content"]
        image_block = next(b for b in content if b.get("type") == "image")
        assert image_block["source"]["type"] == "url"
        assert image_block["source"]["url"] == "https://example.com/img.png"

    @pytest.mark.asyncio
    async def test_complete_with_image_attachment_data(self):
        client = MagicMock()
        client.messages.create = AsyncMock(
            return_value=_make_anthropic_response("ok")
        )
        adapter = self._make_adapter(client)

        att = ImageAttachment(data=b"fake-image-bytes", mime_type="image/png")
        await adapter.complete(
            [{"role": "user", "content": "Describe"}], [], attachments=[att]
        )

        call_kwargs = client.messages.create.call_args.kwargs
        last_msg = call_kwargs["messages"][-1]
        content = last_msg["content"]
        image_block = next(b for b in content if b.get("type") == "image")
        assert image_block["source"]["type"] == "base64"
        assert image_block["source"]["media_type"] == "image/png"

    @pytest.mark.asyncio
    async def test_complete_with_document_attachment(self):
        client = MagicMock()
        client.messages.create = AsyncMock(
            return_value=_make_anthropic_response("ok")
        )
        adapter = self._make_adapter(client)

        att = DocumentAttachment(data=b"pdf-bytes", filename="report.pdf")
        await adapter.complete(
            [{"role": "user", "content": "Summarize"}], [], attachments=[att]
        )

        call_kwargs = client.messages.create.call_args.kwargs
        last_msg = call_kwargs["messages"][-1]
        content = last_msg["content"]
        doc_block = next(b for b in content if b.get("type") == "document")
        assert doc_block["title"] == "report.pdf"

    @pytest.mark.asyncio
    async def test_complete_passes_provider_config(self):
        client = MagicMock()
        client.messages.create = AsyncMock(
            return_value=_make_anthropic_response("ok")
        )
        adapter = self._make_adapter(client)

        cfg = ProviderConfig(temperature=0.7, max_tokens=512, top_p=0.9)
        await adapter.complete([{"role": "user", "content": "hi"}], [], config=cfg)

        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 512
        assert call_kwargs["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_authentication_error_raises_credential_error(self):
        fake_sdk = _fake_anthropic_sdk()
        client = MagicMock()
        client.messages.create = AsyncMock(
            side_effect=fake_sdk.AuthenticationError("bad key")
        )
        with patch.dict(sys.modules, {"anthropic": fake_sdk}):
            from ninetrix.providers.anthropic import AnthropicAdapter
            adapter = AnthropicAdapter("bad", _client=client)
            adapter._sdk = fake_sdk

        with pytest.raises(CredentialError) as exc_info:
            await adapter.complete([{"role": "user", "content": "hi"}], [])
        assert "Anthropic API key" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_rate_limit_error_raises_retryable_provider_error(self):
        fake_sdk = _fake_anthropic_sdk()
        client = MagicMock()
        client.messages.create = AsyncMock(
            side_effect=fake_sdk.RateLimitError("rate limited")
        )
        with patch.dict(sys.modules, {"anthropic": fake_sdk}):
            from ninetrix.providers.anthropic import AnthropicAdapter
            adapter = AnthropicAdapter("key", _client=client)
            adapter._sdk = fake_sdk

        with pytest.raises(ProviderError) as exc_info:
            await adapter.complete([{"role": "user", "content": "hi"}], [])
        assert exc_info.value.retryable is True
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_api_status_error_wraps_to_provider_error(self):
        fake_sdk = _fake_anthropic_sdk()
        client = MagicMock()
        client.messages.create = AsyncMock(
            side_effect=fake_sdk.APIStatusError("server error", status_code=500, message="Internal")
        )
        with patch.dict(sys.modules, {"anthropic": fake_sdk}):
            from ninetrix.providers.anthropic import AnthropicAdapter
            adapter = AnthropicAdapter("key", _client=client)
            adapter._sdk = fake_sdk

        with pytest.raises(ProviderError) as exc_info:
            await adapter.complete([{"role": "user", "content": "hi"}], [])
        assert exc_info.value.status_code == 500
        assert exc_info.value.retryable is True

    @pytest.mark.asyncio
    async def test_unexpected_error_wraps_to_provider_error(self):
        fake_sdk = _fake_anthropic_sdk()
        client = MagicMock()
        client.messages.create = AsyncMock(side_effect=RuntimeError("connection reset"))
        with patch.dict(sys.modules, {"anthropic": fake_sdk}):
            from ninetrix.providers.anthropic import AnthropicAdapter
            adapter = AnthropicAdapter("key", _client=client)
            adapter._sdk = fake_sdk

        with pytest.raises(ProviderError) as exc_info:
            await adapter.complete([{"role": "user", "content": "hi"}], [])
        assert "Unexpected error" in str(exc_info.value)

    def test_import_error_raises_configuration_error(self):
        # Remove anthropic from sys.modules to force ImportError
        with patch.dict(sys.modules, {"anthropic": None}):  # type: ignore[dict-item]
            # Need to re-import the module for the try/except to trigger
            if "ninetrix.providers.anthropic" in sys.modules:
                del sys.modules["ninetrix.providers.anthropic"]
            from ninetrix.providers.anthropic import AnthropicAdapter
            with pytest.raises(ConfigurationError) as exc_info:
                AnthropicAdapter("key")
        assert "pip install anthropic" in str(exc_info.value)

    def test_issubclass_protocol_check(self):
        fake_sdk = _fake_anthropic_sdk()
        with patch.dict(sys.modules, {"anthropic": fake_sdk}):
            if "ninetrix.providers.anthropic" in sys.modules:
                del sys.modules["ninetrix.providers.anthropic"]
            from ninetrix.providers.anthropic import AnthropicAdapter
        assert issubclass(AnthropicAdapter, LLMProviderAdapter)


# ===========================================================================
# OpenAIAdapter
# ===========================================================================

class TestOpenAIAdapter:
    def _make_adapter(self, mock_client: Any) -> Any:
        fake_sdk = _fake_openai_sdk()
        with patch.dict(sys.modules, {"openai": fake_sdk}):
            if "ninetrix.providers.openai" in sys.modules:
                del sys.modules["ninetrix.providers.openai"]
            from ninetrix.providers.openai import OpenAIAdapter
            adapter = OpenAIAdapter("sk-test", model="gpt-test", _client=mock_client)
            adapter._sdk = fake_sdk
        return adapter

    @pytest.mark.asyncio
    async def test_complete_returns_llm_response(self):
        client = MagicMock()
        resp = _make_openai_response("Hello")
        client.chat.completions.create = AsyncMock(return_value=resp)
        adapter = self._make_adapter(client)

        result = await adapter.complete([{"role": "user", "content": "hi"}], [])

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello"
        assert result.input_tokens == 10
        assert result.output_tokens == 5

    @pytest.mark.asyncio
    async def test_complete_with_tool_calls(self):
        client = MagicMock()
        resp = _make_openai_response(
            text="",
            tool_calls=[{"id": "c1", "name": "add", "arguments": {"a": 1, "b": 2}}],
        )
        client.chat.completions.create = AsyncMock(return_value=resp)
        adapter = self._make_adapter(client)

        result = await adapter.complete([{"role": "user", "content": "add"}], [])

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "add"
        assert result.tool_calls[0].arguments == {"a": 1, "b": 2}

    @pytest.mark.asyncio
    async def test_structured_output_adds_response_format(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_make_openai_response('{"x": 1}')
        )
        adapter = self._make_adapter(client)

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        result = await adapter.complete(
            [{"role": "user", "content": "give x"}], [], output_schema=schema
        )

        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"]["type"] == "json_schema"
        assert call_kwargs["response_format"]["json_schema"]["strict"] is True

    @pytest.mark.asyncio
    async def test_builds_openai_tool_format(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_make_openai_response("ok")
        )
        adapter = self._make_adapter(client)

        tools = [{"name": "search", "description": "Search web",
                  "parameters": {"type": "object", "properties": {}}}]
        await adapter.complete([{"role": "user", "content": "search"}], tools)

        call_kwargs = client.chat.completions.create.call_args.kwargs
        t = call_kwargs["tools"][0]
        assert t["type"] == "function"
        assert t["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_image_attachment_url(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_make_openai_response("ok")
        )
        adapter = self._make_adapter(client)

        att = ImageAttachment(url="https://example.com/cat.jpg")
        await adapter.complete(
            [{"role": "user", "content": "describe"}], [], attachments=[att]
        )

        call_kwargs = client.chat.completions.create.call_args.kwargs
        last_msg = call_kwargs["messages"][-1]
        parts = last_msg["content"]
        img_part = next(p for p in parts if p["type"] == "image_url")
        assert img_part["image_url"]["url"] == "https://example.com/cat.jpg"

    @pytest.mark.asyncio
    async def test_authentication_error_raises_credential_error(self):
        fake_sdk = _fake_openai_sdk()
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=fake_sdk.AuthenticationError("bad key")
        )
        with patch.dict(sys.modules, {"openai": fake_sdk}):
            if "ninetrix.providers.openai" in sys.modules:
                del sys.modules["ninetrix.providers.openai"]
            from ninetrix.providers.openai import OpenAIAdapter
            adapter = OpenAIAdapter("bad", _client=client)
            adapter._sdk = fake_sdk

        with pytest.raises(CredentialError):
            await adapter.complete([{"role": "user", "content": "hi"}], [])

    @pytest.mark.asyncio
    async def test_rate_limit_error_is_retryable(self):
        fake_sdk = _fake_openai_sdk()
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=fake_sdk.RateLimitError("rate limited")
        )
        with patch.dict(sys.modules, {"openai": fake_sdk}):
            if "ninetrix.providers.openai" in sys.modules:
                del sys.modules["ninetrix.providers.openai"]
            from ninetrix.providers.openai import OpenAIAdapter
            adapter = OpenAIAdapter("key", _client=client)
            adapter._sdk = fake_sdk

        with pytest.raises(ProviderError) as exc_info:
            await adapter.complete([{"role": "user", "content": "hi"}], [])
        assert exc_info.value.retryable is True

    @pytest.mark.asyncio
    async def test_passes_provider_config_options(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_make_openai_response("ok")
        )
        adapter = self._make_adapter(client)

        cfg = ProviderConfig(
            temperature=0.5,
            max_tokens=256,
            presence_penalty=0.1,
            frequency_penalty=0.2,
        )
        await adapter.complete([{"role": "user", "content": "hi"}], [], config=cfg)

        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 256
        assert call_kwargs["presence_penalty"] == 0.1
        assert call_kwargs["frequency_penalty"] == 0.2

    def test_issubclass_protocol_check(self):
        fake_sdk = _fake_openai_sdk()
        with patch.dict(sys.modules, {"openai": fake_sdk}):
            if "ninetrix.providers.openai" in sys.modules:
                del sys.modules["ninetrix.providers.openai"]
            from ninetrix.providers.openai import OpenAIAdapter
        assert issubclass(OpenAIAdapter, LLMProviderAdapter)

    def test_import_error_raises_configuration_error(self):
        with patch.dict(sys.modules, {"openai": None}):  # type: ignore[dict-item]
            if "ninetrix.providers.openai" in sys.modules:
                del sys.modules["ninetrix.providers.openai"]
            from ninetrix.providers.openai import OpenAIAdapter
            with pytest.raises(ConfigurationError) as exc_info:
                OpenAIAdapter("key")
        assert "pip install openai" in str(exc_info.value)


# ===========================================================================
# GoogleAdapter — _sanitize_schema_for_gemini
# ===========================================================================

class TestGoogleSanitizeSchema:
    """Test the schema sanitizer independently — no SDK needed."""

    def test_strips_unsupported_fields(self):
        from ninetrix.providers.google import _sanitize_schema_for_gemini

        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "additionalProperties": False,
            "default": {},
            "properties": {
                "name": {
                    "type": "string",
                    "$comment": "The user's name",
                    "default": "Alice",
                }
            },
        }
        result = _sanitize_schema_for_gemini(schema)

        assert "$schema" not in result
        assert "additionalProperties" not in result
        assert "default" not in result
        assert "type" in result
        assert "properties" in result
        assert "$comment" not in result["properties"]["name"]
        assert "default" not in result["properties"]["name"]
        assert "type" in result["properties"]["name"]

    def test_strips_defs_and_ref(self):
        from ninetrix.providers.google import _sanitize_schema_for_gemini

        schema = {
            "type": "object",
            "$defs": {"Foo": {"type": "string"}},
            "definitions": {"Bar": {"type": "integer"}},
            "properties": {
                "x": {"$ref": "#/$defs/Foo"},
            },
        }
        result = _sanitize_schema_for_gemini(schema)

        assert "$defs" not in result
        assert "definitions" not in result
        assert "$ref" not in result["properties"]["x"]

    def test_does_not_modify_original(self):
        from ninetrix.providers.google import _sanitize_schema_for_gemini

        original = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"x": {"type": "string", "default": "y"}},
        }
        result = _sanitize_schema_for_gemini(original)

        # Original unchanged
        assert "additionalProperties" in original
        assert "default" in original["properties"]["x"]
        # Result cleaned
        assert "additionalProperties" not in result

    def test_handles_nested_arrays(self):
        from ninetrix.providers.google import _sanitize_schema_for_gemini

        schema = {
            "type": "array",
            "items": [
                {"type": "string", "$comment": "item", "default": "x"},
                {"type": "integer"},
            ],
        }
        result = _sanitize_schema_for_gemini(schema)

        assert "$comment" not in result["items"][0]
        assert "default" not in result["items"][0]
        assert result["items"][1]["type"] == "integer"


class TestGoogleAdapter:
    def _make_adapter(self, mock_model_instance: Any) -> Any:
        """Create a GoogleAdapter with injected mock model instance."""
        fake_google = types.ModuleType("google")
        fake_genai = types.ModuleType("google.generativeai")
        fake_genai.configure = MagicMock()
        fake_genai.GenerativeModel = MagicMock(return_value=mock_model_instance)
        fake_genai.protos = MagicMock()
        fake_genai.Tool = MagicMock()
        fake_google.generativeai = fake_genai

        with patch.dict(
            sys.modules,
            {"google": fake_google, "google.generativeai": fake_genai},
        ):
            if "ninetrix.providers.google" in sys.modules:
                del sys.modules["ninetrix.providers.google"]
            from ninetrix.providers.google import GoogleAdapter
            adapter = GoogleAdapter(
                "api-key",
                model="gemini-test",
                _model_instance=mock_model_instance,
            )
            adapter._sdk = fake_genai
        return adapter

    def _make_google_response(
        self, text: str = "Hello", input_tokens: int = 10, output_tokens: int = 5
    ) -> MagicMock:
        resp = MagicMock()
        resp.usage_metadata.prompt_token_count = input_tokens
        resp.usage_metadata.candidates_token_count = output_tokens
        candidate = MagicMock()
        part = MagicMock(spec=["text"])
        part.text = text
        candidate.content.parts = [part]
        candidate.finish_reason = "STOP"
        resp.candidates = [candidate]
        return resp

    @pytest.mark.asyncio
    async def test_complete_returns_llm_response(self):
        model_inst = MagicMock()
        resp = self._make_google_response("Paris")
        model_inst.generate_content_async = AsyncMock(return_value=resp)
        adapter = self._make_adapter(model_inst)

        result = await adapter.complete([{"role": "user", "content": "Capital?"}], [])

        assert isinstance(result, LLMResponse)
        assert result.content == "Paris"
        assert result.input_tokens == 10
        assert result.output_tokens == 5

    @pytest.mark.asyncio
    async def test_structured_output_sets_response_mime_type(self):
        model_inst = MagicMock()
        model_inst.generate_content_async = AsyncMock(
            return_value=self._make_google_response('{"x": 1}')
        )
        adapter = self._make_adapter(model_inst)

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        await adapter.complete(
            [{"role": "user", "content": "give x"}], [], output_schema=schema
        )

        call_kwargs = model_inst.generate_content_async.call_args.kwargs
        gen_config = call_kwargs["generation_config"]
        assert gen_config["response_mime_type"] == "application/json"
        assert "$schema" not in str(gen_config["response_schema"])

    @pytest.mark.asyncio
    async def test_image_attachment_inline_data(self):
        model_inst = MagicMock()
        model_inst.generate_content_async = AsyncMock(
            return_value=self._make_google_response("ok")
        )
        adapter = self._make_adapter(model_inst)

        att = ImageAttachment(data=b"fake-png-bytes", mime_type="image/png")
        await adapter.complete(
            [{"role": "user", "content": "Describe"}], [], attachments=[att]
        )

        call_args = model_inst.generate_content_async.call_args
        contents = call_args.args[0]
        last_user = next(c for c in reversed(contents) if c["role"] == "user")
        inline_parts = [p for p in last_user["parts"] if "inline_data" in p]
        assert len(inline_parts) == 1
        assert inline_parts[0]["inline_data"]["mime_type"] == "image/png"

    @pytest.mark.asyncio
    async def test_converts_system_message_to_user_turn(self):
        model_inst = MagicMock()
        model_inst.generate_content_async = AsyncMock(
            return_value=self._make_google_response("ok")
        )
        adapter = self._make_adapter(model_inst)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        await adapter.complete(messages, [])

        call_args = model_inst.generate_content_async.call_args
        contents = call_args.args[0]
        # System message should be converted to a user turn at the start
        assert contents[0]["role"] == "user"
        assert any(
            "You are helpful." in str(p.get("text", ""))
            for p in contents[0]["parts"]
        )

    def test_issubclass_protocol_check(self):
        fake_google = types.ModuleType("google")
        fake_genai = types.ModuleType("google.generativeai")
        fake_genai.configure = MagicMock()
        fake_genai.GenerativeModel = MagicMock()
        fake_genai.protos = MagicMock()
        fake_genai.Tool = MagicMock()
        fake_google.generativeai = fake_genai

        with patch.dict(
            sys.modules,
            {"google": fake_google, "google.generativeai": fake_genai},
        ):
            if "ninetrix.providers.google" in sys.modules:
                del sys.modules["ninetrix.providers.google"]
            from ninetrix.providers.google import GoogleAdapter
        assert issubclass(GoogleAdapter, LLMProviderAdapter)


# ===========================================================================
# LiteLLMAdapter
# ===========================================================================

class TestLiteLLMAdapter:
    def _make_adapter(self, mock_acompletion: Any) -> Any:
        fake_litellm = types.ModuleType("litellm")
        fake_litellm.acompletion = mock_acompletion

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            if "ninetrix.providers.litellm" in sys.modules:
                del sys.modules["ninetrix.providers.litellm"]
            from ninetrix.providers.litellm import LiteLLMAdapter
            adapter = LiteLLMAdapter(
                "openai/gpt-4o", api_key="sk-test",
                _acompletion=mock_acompletion,
            )
            adapter._sdk = fake_litellm
        return adapter

    def _make_llm_response(
        self, text: str = "Hello", prompt_tokens: int = 10, completion_tokens: int = 5
    ) -> MagicMock:
        resp = MagicMock()
        choice = MagicMock()
        choice.message.content = text
        choice.message.tool_calls = None
        choice.finish_reason = "stop"
        resp.choices = [choice]
        resp.usage.prompt_tokens = prompt_tokens
        resp.usage.completion_tokens = completion_tokens
        return resp

    @pytest.mark.asyncio
    async def test_complete_returns_llm_response(self):
        mock_acomp = AsyncMock(return_value=self._make_llm_response("Hello"))
        adapter = self._make_adapter(mock_acomp)

        result = await adapter.complete([{"role": "user", "content": "hi"}], [])

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello"

    @pytest.mark.asyncio
    async def test_structured_output_injects_system_message(self):
        mock_acomp = AsyncMock(return_value=self._make_llm_response('{"x":1}'))
        adapter = self._make_adapter(mock_acomp)

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        await adapter.complete(
            [{"role": "user", "content": "give x"}], [], output_schema=schema
        )

        call_kwargs = mock_acomp.call_args.kwargs
        msgs = call_kwargs["messages"]
        system_msgs = [m for m in msgs if m["role"] == "system"]
        assert system_msgs, "Expected a system message with JSON instruction"
        assert "JSON" in system_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_image_attachment_url(self):
        mock_acomp = AsyncMock(return_value=self._make_llm_response("ok"))
        adapter = self._make_adapter(mock_acomp)

        att = ImageAttachment(url="https://example.com/img.jpg")
        await adapter.complete(
            [{"role": "user", "content": "describe"}], [], attachments=[att]
        )

        call_kwargs = mock_acomp.call_args.kwargs
        last_msg = call_kwargs["messages"][-1]
        assert isinstance(last_msg["content"], list)
        img_parts = [p for p in last_msg["content"] if p.get("type") == "image_url"]
        assert img_parts[0]["image_url"]["url"] == "https://example.com/img.jpg"

    @pytest.mark.asyncio
    async def test_rate_limit_error_is_retryable(self):
        class FakeRateLimitError(Exception):
            pass

        mock_acomp = AsyncMock(side_effect=FakeRateLimitError("RateLimitError exceeded"))
        adapter = self._make_adapter(mock_acomp)

        with pytest.raises(ProviderError) as exc_info:
            await adapter.complete([{"role": "user", "content": "hi"}], [])
        assert exc_info.value.retryable is True

    @pytest.mark.asyncio
    async def test_unknown_error_wraps_to_provider_error(self):
        mock_acomp = AsyncMock(side_effect=RuntimeError("network timeout"))
        adapter = self._make_adapter(mock_acomp)

        with pytest.raises(ProviderError) as exc_info:
            await adapter.complete([{"role": "user", "content": "hi"}], [])
        assert "Unexpected error" in str(exc_info.value)

    def test_issubclass_protocol_check(self):
        fake_litellm = types.ModuleType("litellm")
        fake_litellm.acompletion = AsyncMock()
        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            if "ninetrix.providers.litellm" in sys.modules:
                del sys.modules["ninetrix.providers.litellm"]
            from ninetrix.providers.litellm import LiteLLMAdapter
        assert issubclass(LiteLLMAdapter, LLMProviderAdapter)


# ===========================================================================
# FallbackProviderAdapter
# ===========================================================================

def _make_mock_adapter(
    name: str = "mock",
    response: LLMResponse | None = None,
    side_effect: Exception | None = None,
) -> MagicMock:
    adapter = MagicMock(spec=LLMProviderAdapter)
    adapter.provider_name = name
    if side_effect is not None:
        adapter.complete = AsyncMock(side_effect=side_effect)
    else:
        r = response or LLMResponse(
            content=f"ok from {name}",
            tool_calls=[],
            input_tokens=5,
            output_tokens=3,
            stop_reason="end_turn",
        )
        adapter.complete = AsyncMock(return_value=r)
    return adapter


class TestFallbackProviderAdapter:
    @pytest.mark.asyncio
    async def test_returns_primary_when_ok(self):
        primary = _make_mock_adapter("primary")
        fallback_adapter = _make_mock_adapter("fallback")
        fb_cfg = FallbackConfig("fallback", "model-b", on_status_codes=[429])

        adapter = FallbackProviderAdapter(
            primary=primary, fallbacks=[(fallback_adapter, fb_cfg)]
        )
        result = await adapter.complete([], [])

        assert "primary" in result.content
        fallback_adapter.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_on_matching_status_code(self):
        primary = _make_mock_adapter(
            "primary",
            side_effect=ProviderError("overloaded", status_code=429, provider="primary"),
        )
        fallback_adapter = _make_mock_adapter("fallback")
        fb_cfg = FallbackConfig("fallback", "model-b", on_status_codes=[429])

        adapter = FallbackProviderAdapter(
            primary=primary, fallbacks=[(fallback_adapter, fb_cfg)]
        )
        result = await adapter.complete([], [])

        assert "fallback" in result.content
        fallback_adapter.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_falls_back_on_matching_error_string(self):
        primary = _make_mock_adapter(
            "primary",
            side_effect=ProviderError(
                "model is overloaded", status_code=529, provider="primary"
            ),
        )
        fallback_adapter = _make_mock_adapter("fallback")
        fb_cfg = FallbackConfig(
            "fallback", "model-b",
            on_status_codes=[],
            on_error_types=["overloaded"],
        )

        adapter = FallbackProviderAdapter(
            primary=primary, fallbacks=[(fallback_adapter, fb_cfg)]
        )
        result = await adapter.complete([], [])

        assert "fallback" in result.content

    @pytest.mark.asyncio
    async def test_raises_if_chain_exhausted(self):
        exc = ProviderError("rate limit", status_code=429, provider="p")
        primary = _make_mock_adapter("p", side_effect=exc)
        fallback_a = _make_mock_adapter("fa", side_effect=ProviderError("also fail", status_code=429))
        fb_cfg = FallbackConfig("fa", "model-x", on_status_codes=[429])

        adapter = FallbackProviderAdapter(
            primary=primary, fallbacks=[(fallback_a, fb_cfg)]
        )

        with pytest.raises(ProviderError):
            await adapter.complete([], [])

    @pytest.mark.asyncio
    async def test_does_not_fallback_on_non_matching_status(self):
        # 400 is not in the fallback's on_status_codes
        primary = _make_mock_adapter(
            "primary",
            side_effect=ProviderError("bad request", status_code=400, provider="p"),
        )
        fallback_adapter = _make_mock_adapter("fallback")
        fb_cfg = FallbackConfig("fallback", "model-b", on_status_codes=[429, 500])

        adapter = FallbackProviderAdapter(
            primary=primary, fallbacks=[(fallback_adapter, fb_cfg)]
        )

        with pytest.raises(ProviderError) as exc_info:
            await adapter.complete([], [])
        # Should not have fallen back
        fallback_adapter.complete.assert_not_called()
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_passes_kwargs_to_adapters(self):
        primary = _make_mock_adapter("primary")
        adapter = FallbackProviderAdapter(primary=primary, fallbacks=[])

        schema = {"type": "object"}
        cfg = ProviderConfig(temperature=0.5)
        await adapter.complete(
            [{"role": "user", "content": "hi"}],
            [],
            output_schema=schema,
            config=cfg,
        )

        primary.complete.assert_called_once()
        call_kwargs = primary.complete.call_args.kwargs
        assert call_kwargs["output_schema"] == schema
        assert call_kwargs["config"] == cfg

    @pytest.mark.asyncio
    async def test_stream_delegates_to_primary(self):
        primary = MagicMock(spec=LLMProviderAdapter)
        primary.provider_name = "primary"

        async def fake_stream_iter():
            yield LLMChunk(type="token", content="hi")

        primary.stream = AsyncMock(return_value=fake_stream_iter())

        adapter = FallbackProviderAdapter(primary=primary, fallbacks=[])
        stream = await adapter.stream([], [])
        chunks = [c async for c in stream]

        assert len(chunks) == 1
        assert chunks[0].content == "hi"

    def test_fallback_config_defaults(self):
        cfg = FallbackConfig("openai", "gpt-4o")
        assert 429 in cfg.on_status_codes
        assert 500 in cfg.on_status_codes
        assert "rate_limit" in cfg.on_error_types


# ===========================================================================
# __init__.py re-exports
# ===========================================================================

class TestProviderInitReExports:
    def test_top_level_exports_fallback_config(self):
        import ninetrix
        assert hasattr(ninetrix, "FallbackConfig")
        assert hasattr(ninetrix, "FallbackProviderAdapter")

    def test_providers_package_exports(self):
        from ninetrix.providers import FallbackConfig, FallbackProviderAdapter, LLMProviderAdapter
        assert FallbackConfig is not None
        assert FallbackProviderAdapter is not None
        assert LLMProviderAdapter is not None
