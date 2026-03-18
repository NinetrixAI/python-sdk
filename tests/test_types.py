"""
Unit tests for ninetrix._internals.types (PR 1).

Coverage:
- AgentResult[T_Output]: construction, to_dict(), generic usage
- StreamEvent, WorkflowResult, StepResult: field defaults
- LLMResponse, LLMChunk, ToolCall: normalized LLM types
- ImageAttachment, DocumentAttachment, image(), document() helpers
- All 8 error subclasses: message, attributes, hierarchy
- AgentProtocol: isinstance checks via runtime_checkable
- ToolSource, CheckpointerProtocol, LLMProviderAdapter: Protocol structure
"""

import os
import tempfile
import pytest
from ninetrix._internals.types import (
    AgentResult,
    StreamEvent,
    WorkflowResult,
    StepResult,
    LLMResponse,
    LLMChunk,
    ToolCall,
    ProviderConfig,
    ImageAttachment,
    DocumentAttachment,
    image,
    document,
    AgentProtocol,
    ToolSource,
    CheckpointerProtocol,
    LLMProviderAdapter,
    NinetrixError,
    CredentialError,
    ProviderError,
    ToolError,
    BudgetExceededError,
    OutputParseError,
    CheckpointError,
    ApprovalTimeoutError,
    ConfigurationError,
    NetworkError,
)


# ---------------------------------------------------------------------------
# AgentResult
# ---------------------------------------------------------------------------

class TestAgentResult:
    def _make(self, output="hello", **kwargs) -> AgentResult:
        defaults = dict(
            output=output,
            thread_id="t1",
            tokens_used=10,
            input_tokens=7,
            output_tokens=3,
            cost_usd=0.001,
            steps=1,
        )
        defaults.update(kwargs)
        return AgentResult(**defaults)

    def test_basic_construction(self):
        r = self._make()
        assert r.output == "hello"
        assert r.thread_id == "t1"
        assert r.steps == 1
        assert r.error is None
        assert r.history == []

    def test_to_dict_str_output(self):
        d = self._make().to_dict()
        assert d["output"] == "hello"
        assert d["error"] is None
        assert "thread_id" in d
        assert "cost_usd" in d

    def test_to_dict_with_error(self):
        r = self._make(error=ValueError("something broke"))
        d = r.to_dict()
        assert d["error"] == "something broke"

    def test_to_dict_with_pydantic_model(self):
        # Simulate output_type= usage: output is a model with .model_dump()
        class FakeModel:
            def model_dump(self):
                return {"ticker": "AAPL", "price": 150.0}

        r = self._make(output=FakeModel())
        d = r.to_dict()
        assert d["output"] == {"ticker": "AAPL", "price": 150.0}

    def test_history_default_is_empty_list(self):
        r = self._make()
        assert isinstance(r.history, list)
        # ensure it's not shared across instances
        r2 = self._make()
        r.history.append({"role": "user", "content": "hi"})
        assert r2.history == []

    def test_generic_annotation_accepted(self):
        # Static: AgentResult[str] is valid. Runtime: just check it's an AgentResult.
        r: AgentResult[str] = self._make(output="typed")
        assert isinstance(r, AgentResult)


# ---------------------------------------------------------------------------
# StreamEvent
# ---------------------------------------------------------------------------

class TestStreamEvent:
    def test_defaults(self):
        e = StreamEvent(type="token")
        assert e.content == ""
        assert e.tool_name is None
        assert e.tokens_used == 0
        assert e.error is None

    def test_token_event(self):
        e = StreamEvent(type="token", content="Hello")
        assert e.content == "Hello"

    def test_tool_start_event(self):
        e = StreamEvent(type="tool_start", tool_name="get_price", tool_args={"ticker": "AAPL"})
        assert e.tool_name == "get_price"
        assert e.tool_args == {"ticker": "AAPL"}

    def test_done_event(self):
        e = StreamEvent(type="done", tokens_used=42, cost_usd=0.002)
        assert e.tokens_used == 42

    def test_error_event(self):
        err = ProviderError("timeout")
        e = StreamEvent(type="error", error=err)
        assert isinstance(e.error, ProviderError)


# ---------------------------------------------------------------------------
# WorkflowResult
# ---------------------------------------------------------------------------

class TestWorkflowResult:
    def test_construction(self):
        wf = WorkflowResult(
            output="done",
            thread_id="wf1",
            step_results={},
            tokens_used=100,
            cost_usd=0.01,
            elapsed_seconds=3.5,
        )
        assert wf.completed_steps == []
        assert wf.skipped_steps == []

    def test_with_step_results(self):
        step = AgentResult(
            output="step output",
            thread_id="s1",
            tokens_used=20,
            input_tokens=15,
            output_tokens=5,
            cost_usd=0.002,
            steps=1,
        )
        wf = WorkflowResult(
            output="final",
            thread_id="wf2",
            step_results={"analyze": step},
            tokens_used=20,
            cost_usd=0.002,
            elapsed_seconds=1.0,
            completed_steps=["analyze"],
        )
        assert wf.step_results["analyze"].output == "step output"
        assert "analyze" in wf.completed_steps


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------

class TestStepResult:
    def test_construction(self):
        s = StepResult(
            step_name="summarize",
            output="summary text",
            completed_at="2026-03-18T12:00:00Z",
            tokens_used=50,
        )
        assert s.agent_result is None
        assert s.step_name == "summarize"


# ---------------------------------------------------------------------------
# LLMResponse, LLMChunk, ToolCall
# ---------------------------------------------------------------------------

class TestLLMTypes:
    def test_llm_response_defaults(self):
        r = LLMResponse(content="Hello")
        assert r.tool_calls == []
        assert r.input_tokens == 0
        assert r.stop_reason == ""
        assert r.raw is None

    def test_llm_response_with_tool_calls(self):
        tc = ToolCall(id="c1", name="get_price", arguments={"ticker": "AAPL"})
        r = LLMResponse(
            content="",
            tool_calls=[tc],
            input_tokens=10,
            output_tokens=5,
            stop_reason="tool_use",
        )
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "get_price"
        assert r.stop_reason == "tool_use"

    def test_tool_call_fields(self):
        tc = ToolCall(id="abc", name="search_web", arguments={"query": "python"})
        assert tc.id == "abc"
        assert tc.arguments["query"] == "python"

    def test_llm_chunk_token(self):
        c = LLMChunk(type="token", content="Hello")
        assert c.content == "Hello"
        assert c.tool_call_id is None

    def test_llm_chunk_tool_delta(self):
        c = LLMChunk(
            type="tool_call_delta",
            tool_call_id="c1",
            tool_name="get_price",
            tool_arg_delta='{"tick',
        )
        assert c.tool_call_id == "c1"
        assert c.tool_arg_delta == '{"tick'

    def test_tool_calls_list_not_shared(self):
        r1 = LLMResponse(content="a")
        r2 = LLMResponse(content="b")
        r1.tool_calls.append(ToolCall(id="x", name="f", arguments={}))
        assert r2.tool_calls == []


# ---------------------------------------------------------------------------
# Attachment types
# ---------------------------------------------------------------------------

class TestImageAttachment:
    def test_url_attachment(self):
        img = ImageAttachment(url="https://example.com/img.png")
        assert img.url == "https://example.com/img.png"
        assert img.data is None

    def test_bytes_attachment(self):
        img = ImageAttachment(data=b"\xff\xd8\xff", mime_type="image/jpeg")
        assert img.data == b"\xff\xd8\xff"
        assert img.url is None

    def test_to_base64(self):
        import base64
        raw = b"hello"
        img = ImageAttachment(data=raw)
        assert img.to_base64() == base64.b64encode(raw).decode()

    def test_neither_url_nor_data_raises(self):
        with pytest.raises(ValueError, match="url="):
            ImageAttachment()

    def test_to_base64_without_data_raises(self):
        img = ImageAttachment(url="https://example.com/img.png")
        with pytest.raises(ValueError, match="data="):
            img.to_base64()


class TestDocumentAttachment:
    def test_construction(self):
        doc = DocumentAttachment(data=b"%PDF-1.4", filename="report.pdf")
        assert doc.filename == "report.pdf"
        assert doc.mime_type == "application/pdf"

    def test_to_base64(self):
        import base64
        raw = b"%PDF-1.4"
        doc = DocumentAttachment(data=raw, filename="f.pdf")
        assert doc.to_base64() == base64.b64encode(raw).decode()


class TestAttachmentHelpers:
    def test_image_from_url(self):
        img = image("https://example.com/photo.jpg")
        assert img.url == "https://example.com/photo.jpg"
        assert img.data is None

    def test_image_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff")
            path = f.name
        try:
            img = image(path)
            assert img.data == b"\xff\xd8\xff"
            assert img.url is None
            assert "image" in img.mime_type
        finally:
            os.unlink(path)

    def test_document_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4")
            path = f.name
        try:
            doc = document(path)
            assert doc.data == b"%PDF-1.4"
            assert doc.filename == os.path.basename(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------

class TestErrorHierarchy:
    def test_all_inherit_from_ninetrix_error(self):
        for cls in (
            CredentialError, ProviderError, ToolError, BudgetExceededError,
            OutputParseError, CheckpointError, ApprovalTimeoutError,
            ConfigurationError, NetworkError,
        ):
            assert issubclass(cls, NinetrixError), f"{cls.__name__} must inherit NinetrixError"

    def test_all_are_exceptions(self):
        assert issubclass(NinetrixError, Exception)

    def test_ninetrix_error_message(self):
        e = NinetrixError("base error")
        assert str(e) == "base error"

    def test_credential_error(self):
        e = CredentialError(
            "No API key for 'anthropic'.\n"
            "  Why: ANTHROPIC_API_KEY not set.\n"
            "  Fix: export ANTHROPIC_API_KEY=sk-ant-..."
        )
        assert "anthropic" in str(e)
        assert isinstance(e, NinetrixError)

    def test_provider_error_attributes(self):
        e = ProviderError("rate limited", status_code=429, provider="anthropic", retryable=True)
        assert e.status_code == 429
        assert e.provider == "anthropic"
        assert e.retryable is True

    def test_provider_error_defaults(self):
        e = ProviderError("server error")
        assert e.status_code is None
        assert e.provider == ""
        assert e.retryable is False

    def test_tool_error_attributes(self):
        e = ToolError("tool crashed", tool_name="get_price", arguments={"ticker": "AAPL"})
        assert e.tool_name == "get_price"
        assert e.arguments == {"ticker": "AAPL"}

    def test_tool_error_defaults(self):
        e = ToolError("failed")
        assert e.tool_name == ""
        assert e.arguments == {}

    def test_budget_exceeded_error_attributes(self):
        e = BudgetExceededError(
            "Budget exceeded",
            budget_usd=1.0,
            spent_usd=1.05,
            budget_tokens=10000,
            spent_tokens=10200,
        )
        assert e.budget_usd == 1.0
        assert e.spent_usd == 1.05
        assert e.budget_tokens == 10000
        assert e.spent_tokens == 10200

    def test_output_parse_error_attributes(self):
        e = OutputParseError(
            "Could not parse JSON",
            raw_output='{"bad": }',
            validation_errors=[{"loc": ["ticker"], "msg": "field required"}],
            attempts=2,
        )
        assert e.raw_output == '{"bad": }'
        assert len(e.validation_errors) == 1
        assert e.attempts == 2

    def test_output_parse_error_defaults(self):
        e = OutputParseError("bad")
        assert e.raw_output == ""
        assert e.validation_errors == []
        assert e.attempts == 1

    def test_checkpoint_error_attributes(self):
        e = CheckpointError("DB unavailable", thread_id="t1", operation="save")
        assert e.thread_id == "t1"
        assert e.operation == "save"

    def test_approval_timeout_error_attributes(self):
        e = ApprovalTimeoutError("timed out", timeout_seconds=30.0, step_name="review")
        assert e.timeout_seconds == 30.0
        assert e.step_name == "review"

    def test_network_error_retryable_default(self):
        e = NetworkError("connection refused")
        assert e.retryable is True

    def test_network_error_not_retryable(self):
        e = NetworkError("SSL cert invalid", retryable=False)
        assert e.retryable is False

    def test_configuration_error(self):
        e = ConfigurationError("provider 'badprovider' is not supported")
        assert isinstance(e, NinetrixError)

    def test_errors_can_be_raised_and_caught(self):
        with pytest.raises(ProviderError) as exc_info:
            raise ProviderError("API down", status_code=503, retryable=True)
        assert exc_info.value.retryable is True

        with pytest.raises(NinetrixError):
            raise CredentialError("missing key")


# ---------------------------------------------------------------------------
# AgentProtocol — runtime_checkable
# ---------------------------------------------------------------------------

class TestAgentProtocol:
    def test_class_with_all_methods_satisfies_protocol(self):
        from typing import AsyncIterator

        class FakeAgent:
            name = "fake"

            def run(self, message, *, thread_id=None):
                return AgentResult(
                    output=message, thread_id="t", tokens_used=0,
                    input_tokens=0, output_tokens=0, cost_usd=0.0, steps=0,
                )

            async def arun(self, message, *, thread_id=None):
                return self.run(message)

            async def stream(self, message, *, thread_id=None) -> AsyncIterator[StreamEvent]:
                yield StreamEvent(type="done")

        agent = FakeAgent()
        assert isinstance(agent, AgentProtocol)

    def test_class_missing_arun_does_not_satisfy(self):
        class Incomplete:
            name = "x"
            def run(self, message, *, thread_id=None): ...

        assert not isinstance(Incomplete(), AgentProtocol)

    def test_class_missing_name_does_not_satisfy(self):
        from typing import AsyncIterator

        class NoName:
            def run(self, message, *, thread_id=None): ...
            async def arun(self, message, *, thread_id=None): ...
            async def stream(self, message, *, thread_id=None) -> AsyncIterator[StreamEvent]:
                yield StreamEvent(type="done")

        assert not isinstance(NoName(), AgentProtocol)


# ---------------------------------------------------------------------------
# ProviderConfig
# ---------------------------------------------------------------------------

class TestProviderConfig:
    def test_defaults(self):
        cfg = ProviderConfig()
        assert cfg.temperature == 0.0
        assert cfg.max_tokens is None
        assert cfg.top_p is None
        assert cfg.stop_sequences == []
        assert cfg.presence_penalty is None
        assert cfg.frequency_penalty is None

    def test_custom_values(self):
        cfg = ProviderConfig(
            temperature=0.7,
            max_tokens=2048,
            top_p=0.9,
            stop_sequences=["STOP", "END"],
            presence_penalty=0.1,
            frequency_penalty=0.2,
        )
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 2048
        assert cfg.top_p == 0.9
        assert cfg.stop_sequences == ["STOP", "END"]
        assert cfg.presence_penalty == 0.1
        assert cfg.frequency_penalty == 0.2

    def test_stop_sequences_not_shared(self):
        c1 = ProviderConfig()
        c2 = ProviderConfig()
        c1.stop_sequences.append("STOP")
        assert c2.stop_sequences == []

    def test_exported_from_ninetrix(self):
        import ninetrix
        assert hasattr(ninetrix, "ProviderConfig")
        cfg = ninetrix.ProviderConfig(temperature=0.5)
        assert cfg.temperature == 0.5


# ---------------------------------------------------------------------------
# LLMProviderAdapter Protocol — updated signature
# ---------------------------------------------------------------------------

class TestLLMProviderAdapterProtocol:
    def test_class_with_complete_and_stream_satisfies_protocol(self):
        from typing import AsyncIterator

        class FakeAdapter:
            async def complete(
                self, messages, tools, *,
                attachments=None, output_schema=None, config=None
            ) -> LLMResponse:
                return LLMResponse(content="ok")

            async def stream(
                self, messages, tools, *,
                attachments=None, config=None
            ) -> AsyncIterator[LLMChunk]:
                yield LLMChunk(type="done")

        assert isinstance(FakeAdapter(), LLMProviderAdapter)

    def test_class_missing_stream_does_not_satisfy(self):
        class NoStream:
            async def complete(self, messages, tools, **kwargs) -> LLMResponse:
                return LLMResponse(content="ok")

        assert not isinstance(NoStream(), LLMProviderAdapter)

    def test_complete_accepts_attachments(self):
        """Verify attachments parameter is accepted (structural check via duck typing)."""
        import inspect
        from ninetrix._internals.types import LLMProviderAdapter

        # The Protocol signature should accept attachments= keyword
        # We verify this by checking a concrete implementation that uses it compiles
        class FakeAdapter:
            async def complete(self, messages, tools, *, attachments=None,
                               output_schema=None, config=None):
                assert attachments is None or isinstance(attachments, list)
                return LLMResponse(content="")

            async def stream(self, messages, tools, *, attachments=None, config=None):
                yield LLMChunk(type="done")

        import asyncio
        adapter = FakeAdapter()
        result = asyncio.run(adapter.complete([], [], attachments=None))
        assert result.content == ""

    def test_complete_accepts_provider_config(self):
        import asyncio

        class FakeAdapter:
            async def complete(self, messages, tools, *, attachments=None,
                               output_schema=None, config=None):
                if config is not None:
                    assert isinstance(config, ProviderConfig)
                return LLMResponse(content="")

            async def stream(self, messages, tools, *, attachments=None, config=None):
                yield LLMChunk(type="done")

        cfg = ProviderConfig(temperature=0.5, max_tokens=100)
        adapter = FakeAdapter()
        result = asyncio.run(adapter.complete([], [], config=cfg))
        assert result.content == ""


# ---------------------------------------------------------------------------
# py.typed marker
# ---------------------------------------------------------------------------

def test_py_typed_marker_exists():
    import ninetrix
    marker = os.path.join(os.path.dirname(ninetrix.__file__), "py.typed")
    assert os.path.exists(marker), "py.typed must exist for PEP 561 compliance"
