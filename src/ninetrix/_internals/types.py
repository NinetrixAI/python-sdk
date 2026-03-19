"""
ninetrix._internals.types
=========================
L1 kernel — stdlib only, no ninetrix imports.

Contains:
- Public result / event types (AgentResult, StreamEvent, WorkflowResult, StepResult)
- Internal normalized LLM types (LLMResponse, LLMChunk, ToolCall)
- Attachment types (ImageAttachment, DocumentAttachment)
- Internal structural Protocols (ToolSource, CheckpointerProtocol, LLMProviderAdapter)
- Full error hierarchy
- AgentProtocol — the @runtime_checkable Protocol satisfied by Agent, AgentClient, RemoteAgent
"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    runtime_checkable,
)

# ---------------------------------------------------------------------------
# Type aliases / literals
# ---------------------------------------------------------------------------

Provider = Literal["anthropic", "openai", "google", "groq", "mistral", "litellm"]
ExecutionMode = Literal["direct", "planned"]
OnStepFailure = Literal["continue", "abort", "retry_once"]

# T_Output: str by default; a Pydantic BaseModel subclass when output_type= is set.
# We keep it unbound here (L1 has no pydantic dependency) — callers narrow it.
T_Output = TypeVar("T_Output")


# ---------------------------------------------------------------------------
# Attachment types
# ---------------------------------------------------------------------------

@dataclass
class ImageAttachment:
    """Image passed as part of a user message."""
    url: str | None = None          # public URL — mutually exclusive with data
    data: bytes | None = None       # raw bytes — mutually exclusive with url
    mime_type: str = "image/jpeg"

    def __post_init__(self) -> None:
        if self.url is None and self.data is None:
            raise ValueError(
                "ImageAttachment requires either url= or data=.\n"
                "  → Pass url='https://...' for a public image, or\n"
                "  → Pass data=bytes + mime_type= for inline image data."
            )

    def to_base64(self) -> str:
        if self.data is None:
            raise ValueError("ImageAttachment.to_base64() called but data= is None.")
        return base64.b64encode(self.data).decode()


@dataclass
class DocumentAttachment:
    """Document (PDF, text, etc.) passed as part of a user message."""
    data: bytes
    filename: str
    mime_type: str = "application/pdf"

    def to_base64(self) -> str:
        return base64.b64encode(self.data).decode()


Attachment = ImageAttachment | DocumentAttachment


def image(path_or_url: str) -> ImageAttachment:
    """Convenience constructor: path → inline bytes, URL → url= field."""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return ImageAttachment(url=path_or_url)
    data = Path(path_or_url).read_bytes()
    mime, _ = mimetypes.guess_type(path_or_url)
    return ImageAttachment(data=data, mime_type=mime or "image/jpeg")


def document(path: str) -> DocumentAttachment:
    """Convenience constructor: file path → DocumentAttachment."""
    p = Path(path)
    mime, _ = mimetypes.guess_type(path)
    return DocumentAttachment(
        data=p.read_bytes(),
        filename=p.name,
        mime_type=mime or "application/octet-stream",
    )


# ---------------------------------------------------------------------------
# Normalized LLM types (provider-agnostic)
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A single tool call requested by the LLM (normalized across providers)."""
    id: str                 # provider-assigned call ID (used to match tool results)
    name: str               # tool function name
    arguments: dict         # parsed JSON arguments


@dataclass
class LLMChunk:
    """One streamed chunk from the LLM."""
    type: Literal["token", "tool_call_delta", "done"]
    content: str = ""
    tool_call_id: str | None = None     # present on tool_call_delta chunks
    tool_name: str | None = None        # present on first chunk of a tool call
    tool_arg_delta: str | None = None   # raw JSON fragment


@dataclass
class LLMResponse:
    """
    Normalized response from any LLM provider.
    Providers produce this; the runner consumes it.
    """
    content: str                        # text response (empty if pure tool-call turn)
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""               # "end_turn", "tool_use", "max_tokens", etc.
    raw: Any = None                     # original provider response (for debugging)


# ---------------------------------------------------------------------------
# Public result / event types
# ---------------------------------------------------------------------------

@dataclass
class AgentResult(Generic[T_Output]):
    """
    Result of a single agent.run() / agent.arun() call.

    output is T_Output:
    - str when no output_type= was set (default)
    - a validated Pydantic model instance when output_type=MyModel was set
    """
    output: T_Output                        # final answer (str or parsed model)
    thread_id: str
    tokens_used: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    steps: int                              # number of LLM turns taken
    history: list[dict] = field(default_factory=list)   # full message list
    error: Exception | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict (no raw Exception objects)."""
        return {
            "output": self.output if not hasattr(self.output, "model_dump")
                      else self.output.model_dump(),  # type: ignore[union-attr]
            "thread_id": self.thread_id,
            "tokens_used": self.tokens_used,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "steps": self.steps,
            "history": self.history,
            "error": str(self.error) if self.error else None,
        }


@dataclass
class StreamEvent:
    """Single event yielded by agent.stream()."""
    type: Literal["token", "tool_start", "tool_end", "turn_end", "done", "error"]
    content: str = ""
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_result: str | None = None
    tokens_used: int = 0
    cost_usd: float = 0.0
    error: Exception | None = None
    structured_output: Any = None   # set on "done" when output_type= is used


@dataclass
class WorkflowResult:
    """Result of a workflow.run() call."""
    output: Any
    thread_id: str
    step_results: dict[str, AgentResult]    # step_name → result
    tokens_used: int
    cost_usd: float
    elapsed_seconds: float
    completed_steps: list[str] = field(default_factory=list)
    skipped_steps: list[str] = field(default_factory=list)   # resumed steps
    budget_remaining_usd: float = 0.0    # how much budget was left (0 if no limit)
    budget_limit_usd: float = 0.0        # what the limit was (0 = no limit)
    terminated: bool = False             # True when Workflow.terminate() was called
    termination_reason: str = ""         # reason passed to Workflow.terminate()


@dataclass
class StepResult:
    """Persisted result of one Workflow.step() block."""
    step_name: str
    output: Any
    completed_at: str                       # ISO timestamp
    tokens_used: int
    agent_result: AgentResult | None = None


# ---------------------------------------------------------------------------
# Internal structural Protocols
# Used by runtime/, checkpoint/, providers/ — defined here to keep L1 self-contained.
# ---------------------------------------------------------------------------

@runtime_checkable
class ToolSource(Protocol):
    """
    Provides tool definitions and dispatches calls.
    Implemented by LocalToolSource and RegistryToolSource (and future MCPToolSource).
    """

    async def initialize(self) -> None:
        """Fetch/warm tool schemas. Called once before first run."""
        ...

    async def list_tools(self) -> list[dict]:
        """Return list of tool schemas in the format expected by the provider."""
        ...

    async def call(self, name: str, arguments: dict) -> str:
        """
        Invoke a tool by name with parsed arguments.
        Returns the tool result as a string (JSON-serialized if needed).
        """
        ...


@runtime_checkable
class CheckpointerProtocol(Protocol):
    """
    Persists and restores agent conversation state.
    Implemented by InMemoryCheckpointer and PostgresCheckpointer.
    """

    async def save(
        self,
        thread_id: str,
        agent_id: str,
        history: list[dict],
        tokens_used: int,
    ) -> None: ...

    async def load(self, thread_id: str) -> dict[str, Any] | None: ...

    async def delete(self, thread_id: str) -> None: ...


@dataclass
class ProviderConfig:
    """
    Normalized, provider-agnostic parameters passed to LLMProviderAdapter.complete()
    and LLMProviderAdapter.stream().

    Each adapter extracts the fields it supports and ignores the rest.
    All fields are optional — adapters fall back to their own defaults when None.

    How it maps to provider APIs:
        temperature       → all providers
        max_tokens        → Anthropic max_tokens, OpenAI max_tokens, Google maxOutputTokens
        top_p             → Anthropic top_p, OpenAI top_p, Google topP
        stop_sequences    → Anthropic stop_sequences, OpenAI stop, Google stopSequences
        presence_penalty  → OpenAI only (ignored by Anthropic / Google)
        frequency_penalty → OpenAI only (ignored by Anthropic / Google)
    """
    temperature: float = 0.0
    max_tokens: int | None = None
    top_p: float | None = None
    stop_sequences: list[str] = field(default_factory=list)
    presence_penalty: float | None = None      # OpenAI only
    frequency_penalty: float | None = None     # OpenAI only


@runtime_checkable
class LLMProviderAdapter(Protocol):
    """
    Normalizes calls to any LLM provider into LLMResponse.
    Implemented by AnthropicAdapter, OpenAIAdapter, GoogleAdapter, LiteLLMAdapter.

    Structured output contract:
        If output_schema is set, the provider instructs the LLM to respond with
        valid JSON matching that schema (using native JSON mode or tool-use trick).
        The provider returns the raw JSON string in LLMResponse.content.
        The runner owns the parse+retry loop — providers do NOT parse the JSON.

    Attachments contract:
        Attachments are processed by the provider into provider-native content blocks
        (Anthropic content list, OpenAI message parts, etc.) before the API call.
        They are appended to the last user message in the messages list.

    force_json_schema (NOT on this Protocol):
        Each adapter has a private _force_json_schema(schema) method.
        It is called internally when output_schema is set — not by the runner.
    """

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        attachments: "list[Attachment] | None" = None,
        output_schema: dict | None = None,
        config: ProviderConfig | None = None,
    ) -> LLMResponse: ...

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        attachments: "list[Attachment] | None" = None,
        config: ProviderConfig | None = None,
    ) -> AsyncIterator[LLMChunk]: ...


# ---------------------------------------------------------------------------
# AgentProtocol — public, @runtime_checkable
# Satisfied by Agent, AgentClient, and RemoteAgent.
# Workflow and Team accept AgentProtocol — never Agent specifically.
# ---------------------------------------------------------------------------

@runtime_checkable
class AgentProtocol(Protocol):
    """
    Structural protocol satisfied by Agent, AgentClient, and RemoteAgent.

    Any object with these three methods and a name attribute can be used
    wherever an agent is expected — in Workflow, Team, or user code.
    The developer never needs to know if the agent runs locally or in the cloud.
    """

    name: str

    def run(self, message: str, *, thread_id: str | None = None) -> AgentResult: ...

    async def arun(
        self, message: str, *, thread_id: str | None = None
    ) -> AgentResult: ...

    async def stream(
        self, message: str, *, thread_id: str | None = None
    ) -> AsyncIterator[StreamEvent]: ...


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------

class NinetrixError(Exception):
    """
    Base class for all ninetrix SDK errors.

    Every subclass must be raised with a message that includes:
    - What failed
    - Why it failed
    - How to fix it

    Example:
        raise CredentialError(
            "No API key found for provider 'anthropic'.\n"
            "  Why: ANTHROPIC_API_KEY is not set and no key was found in "
            "~/.ninetrix/credentials.toml.\n"
            "  Fix: export ANTHROPIC_API_KEY=sk-ant-... or run: ninetrix config set-key anthropic"
        )
    """


class CredentialError(NinetrixError):
    """
    Raised when an API key or credential is missing, expired, or invalid.

    What:  Which provider or service lacks a credential.
    Why:   Where the SDK looked and found nothing.
    Fix:   How to supply the credential (env var name, config command, etc.)
    """


class ProviderError(NinetrixError):
    """
    Raised when an LLM provider call fails (network, rate limit, server error).
    Raw provider exceptions (anthropic.APIError, openai.APIError, etc.) are
    always wrapped in this — nothing from third-party SDKs leaks out.

    Attributes:
        status_code: HTTP status code if available, else None.
        provider:    Which provider raised (e.g. "anthropic").
        retryable:   Whether retrying after a backoff might succeed.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider: str = "",
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider
        self.retryable = retryable


class ToolError(NinetrixError):
    """
    Raised when a tool call fails (function raised, timed out, or returned bad output).

    Attributes:
        tool_name:   Which tool failed.
        arguments:   The arguments that were passed (for debugging).
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str = "",
        arguments: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.arguments = arguments or {}


class BudgetExceededError(NinetrixError):
    """
    Raised when a run exceeds its token or cost budget.

    Attributes:
        budget_usd:   The configured budget limit (USD).
        spent_usd:    How much was spent before the limit was hit.
        budget_tokens: The configured token limit (if applicable).
        spent_tokens:  Tokens consumed before the limit was hit.
    """

    def __init__(
        self,
        message: str,
        *,
        budget_usd: float | None = None,
        spent_usd: float | None = None,
        budget_tokens: int | None = None,
        spent_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.budget_usd = budget_usd
        self.spent_usd = spent_usd
        self.budget_tokens = budget_tokens
        self.spent_tokens = spent_tokens


class OutputParseError(NinetrixError):
    """
    Raised when the LLM response cannot be parsed into the requested output_type
    after all retries are exhausted.

    Attributes:
        raw_output:        The raw string the LLM returned.
        validation_errors: Pydantic validation error detail (as list of dicts).
        attempts:          How many parse attempts were made.
    """

    def __init__(
        self,
        message: str,
        *,
        raw_output: str = "",
        validation_errors: list[dict] | None = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.raw_output = raw_output
        self.validation_errors = validation_errors or []
        self.attempts = attempts


class CheckpointError(NinetrixError):
    """
    Raised when saving or loading a checkpoint fails.

    Attributes:
        thread_id: The thread being checkpointed.
        operation: "save" or "load".
    """

    def __init__(
        self,
        message: str,
        *,
        thread_id: str = "",
        operation: Literal["save", "load", "delete"] = "save",
    ) -> None:
        super().__init__(message)
        self.thread_id = thread_id
        self.operation = operation


class ApprovalTimeoutError(NinetrixError):
    """
    Raised when a human-in-the-loop approval step times out.

    Attributes:
        timeout_seconds: How long the SDK waited before timing out.
        step_name:       The workflow step waiting for approval.
    """

    def __init__(
        self,
        message: str,
        *,
        timeout_seconds: float = 0.0,
        step_name: str = "",
    ) -> None:
        super().__init__(message)
        self.timeout_seconds = timeout_seconds
        self.step_name = step_name


class ConfigurationError(NinetrixError):
    """
    Raised when the SDK detects an invalid configuration at startup or agent creation.
    (Wrong types, conflicting options, unsupported provider+model combo, etc.)
    """


class NetworkError(NinetrixError):
    """
    Raised on connection failures, DNS errors, or timeouts when reaching
    external services (providers, MCP gateway, Ninetrix Cloud).

    Attributes:
        retryable: True for transient failures (timeout, 503), False for permanent ones.
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable
