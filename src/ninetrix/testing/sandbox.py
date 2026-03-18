"""
ninetrix.testing.sandbox — AgentSandbox: isolated test environment for agents.

Layer: L9 (testing) — may import L8 (agent) and all below.

AgentSandbox replaces the agent's underlying provider with a mock provider
and the dispatcher with a sandboxed one that only dispatches to registered
MockTools.  Real LLM API calls are never made unless a real provider is
explicitly passed.

Usage::

    from ninetrix.testing import AgentSandbox, MockTool

    async with AgentSandbox(agent, script=["The weather is 72°F."]) as sandbox:
        sandbox.add_mock(MockTool("get_weather", return_value={"temp": 72}))
        result = await sandbox.run("What is the weather?")
        assert result.success
        sandbox.assert_tool_called("get_weather")
        sandbox.assert_tool_call_count("get_weather", 1)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from ninetrix._internals.types import (
    AgentResult,
    LLMChunk,
    LLMResponse,
    ProviderConfig,
    ToolCall,
)
from ninetrix.testing.mock_tool import MockTool


# ---------------------------------------------------------------------------
# SandboxResult — extends AgentResult with a convenience .success property
# ---------------------------------------------------------------------------


@dataclass
class SandboxResult:
    """Result of an AgentSandbox.run() call.

    Wraps :class:`~ninetrix._internals.types.AgentResult` and adds a
    ``success`` convenience property.

    Attributes:
        output:        Final answer string (or parsed Pydantic model).
        thread_id:     Conversation ID.
        tokens_used:   Tokens consumed during the run (always 0 for mock).
        input_tokens:  Input tokens (always 0 for mock).
        output_tokens: Output tokens (always 0 for mock).
        cost_usd:      Cost (always 0.0 for mock).
        steps:         Number of LLM turns taken.
        history:       Full message list (system excluded).
        error:         Exception if the run failed, else ``None``.
        success:       ``True`` when ``error is None``.
    """

    output: Any
    thread_id: str
    tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    steps: int = 0
    history: list[dict] = field(default_factory=list)
    error: Optional[Exception] = None

    @property
    def success(self) -> bool:
        """True when the run completed without error."""
        return self.error is None

    @classmethod
    def from_agent_result(cls, result: AgentResult) -> "SandboxResult":
        """Wrap an AgentResult."""
        return cls(
            output=result.output,
            thread_id=result.thread_id,
            tokens_used=result.tokens_used,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
            steps=result.steps,
            history=result.history,
            error=result.error,
        )


# ---------------------------------------------------------------------------
# MockProvider — canned LLM responses for testing
# ---------------------------------------------------------------------------


class MockProvider:
    """A fake LLM provider that cycles through a pre-defined script.

    Each ``complete()`` call advances through the script.  When the script
    is exhausted the last message is repeated.

    Args:
        script:            List of plain-text assistant messages.
        tool_calls_script: Richer script with optional tool_calls per turn.
                           Each entry is a dict with ``content`` and optional
                           ``tool_calls`` list.  ``tool_calls`` items:
                           ``{"name": str, "arguments": dict}``.
    """

    def __init__(
        self,
        script: Optional[list[str]] = None,
        tool_calls_script: Optional[list[dict]] = None,
    ) -> None:
        self._index = 0

        if tool_calls_script is not None:
            self._turns: list[dict] = tool_calls_script
        elif script:
            self._turns = [{"content": s, "tool_calls": []} for s in script]
        else:
            # Default single-turn: "I have completed the task."
            self._turns = [{"content": "I have completed the task.", "tool_calls": []}]

    def _next_turn(self) -> dict:
        if self._index < len(self._turns):
            turn = self._turns[self._index]
            self._index += 1
        else:
            turn = self._turns[-1] if self._turns else {
                "content": "I have completed the task.",
                "tool_calls": [],
            }
        return turn

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        attachments: Any = None,
        output_schema: Any = None,
        config: ProviderConfig | None = None,
    ) -> LLMResponse:
        turn = self._next_turn()
        content = turn.get("content", "")
        raw_tool_calls = turn.get("tool_calls") or []

        tool_calls: list[ToolCall] = []
        for i, tc in enumerate(raw_tool_calls):
            tool_calls.append(ToolCall(
                id=f"mock_tc_{self._index}_{i}",
                name=tc["name"],
                arguments=tc.get("arguments", {}),
            ))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=0,
            output_tokens=0,
            stop_reason="end_turn" if not tool_calls else "tool_use",
        )

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        attachments: Any = None,
        config: ProviderConfig | None = None,
    ):
        """Yield a single token chunk from the current script entry."""
        response = await self.complete(messages, tools)
        if response.content:
            yield LLMChunk(type="token", content=response.content)
        yield LLMChunk(type="done")

    def reset(self) -> None:
        """Reset the script index to the beginning."""
        self._index = 0


# ---------------------------------------------------------------------------
# SandboxedToolSource — dispatches ONLY to registered MockTools
# ---------------------------------------------------------------------------


class SandboxedToolSource:
    """A ToolSource that routes calls exclusively to registered MockTools.

    Any call to an unregistered tool returns an error string rather than
    failing hard, so the agent can continue (and the sandbox can report it).

    Args:
        mocks: Dict of ``{tool_name: MockTool}``.
    """

    def __init__(self, mocks: dict[str, MockTool]) -> None:
        self._mocks = mocks

    def tool_definitions(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Mock tool: {name}",
                    "parameters": mock._schema,
                },
            }
            for name, mock in self._mocks.items()
        ]

    def handles(self, tool_name: str) -> bool:
        return tool_name in self._mocks

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if tool_name not in self._mocks:
            return f"Error: tool '{tool_name}' is not registered in sandbox."
        mock = self._mocks[tool_name]
        result = await mock._dispatch(**arguments)
        if result is None:
            return "(done)"
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(result)


# ---------------------------------------------------------------------------
# SandboxedDispatcher
# ---------------------------------------------------------------------------


class SandboxedDispatcher:
    """A ToolDispatcher replacement that uses only SandboxedToolSource.

    Tracks all tool calls with their arguments and results for later
    assertion via AgentSandbox.

    Args:
        source: The single sandboxed tool source.
    """

    def __init__(self, source: SandboxedToolSource) -> None:
        self._source = source
        self._call_log: list[dict[str, Any]] = []

    async def initialize(self) -> None:
        """No-op; mocks are pre-registered."""

    def all_tool_definitions(self) -> list[dict]:
        return self._source.tool_definitions()

    async def call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        **kwargs: Any,
    ) -> str:
        result = await self._source.call(tool_name, arguments)
        self._call_log.append({
            "tool": tool_name,
            "args": dict(arguments),
            "result": result,
        })
        return result

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return list(self._call_log)


# ---------------------------------------------------------------------------
# AgentSandbox
# ---------------------------------------------------------------------------


class AgentSandbox:
    """Isolated test environment for running an agent without real LLM calls.

    ``AgentSandbox`` replaces the agent's provider with a :class:`MockProvider`
    and its dispatcher with a :class:`SandboxedDispatcher` that routes tool
    calls only to registered :class:`MockTool` instances.

    Usage::

        async with AgentSandbox(agent, script=["Paris is the capital of France."]) as sandbox:
            result = await sandbox.run("What is the capital of France?")
            assert result.success
            assert "Paris" in result.output

        # With mock tools and tool_calls_script:
        async with AgentSandbox(
            agent,
            tool_calls_script=[
                {
                    "content": "Let me search.",
                    "tool_calls": [{"name": "search", "arguments": {"q": "Paris"}}],
                },
                {"content": "Paris is in France."},
            ],
        ) as sandbox:
            sandbox.add_mock(MockTool("search", return_value=["Paris, France"]))
            result = await sandbox.run("Tell me about Paris")
            sandbox.assert_tool_called("search")

    Args:
        agent:              :class:`~ninetrix.agent.agent.Agent` instance to run.
        script:             List of plain-text assistant responses the mock
                            provider will cycle through.  Last entry repeats.
        tool_calls_script:  Richer script list; each entry is a dict with
                            ``content`` (str) and optional ``tool_calls``
                            (list of ``{"name": str, "arguments": dict}``).
        provider:           Optional custom provider.  Pass ``None`` to use the
                            default :class:`MockProvider`.  Pass a real adapter
                            for integration tests.

    .. note::
        ``script`` and ``tool_calls_script`` are mutually exclusive.  If both
        are provided ``tool_calls_script`` takes precedence.
    """

    def __init__(
        self,
        agent: Any,                         # Agent — avoid circular import at module level
        *,
        script: Optional[list[str]] = None,
        tool_calls_script: Optional[list[dict]] = None,
        provider: Any = None,
    ) -> None:
        self._agent = agent
        self._script = script
        self._tool_calls_script = tool_calls_script
        self._custom_provider = provider

        # Mutable state — populated between __aenter__ and __aexit__
        self._mocks: dict[str, MockTool] = {}
        self._dispatcher: Optional[SandboxedDispatcher] = None
        self._mock_provider: Optional[MockProvider] = None

        # Saved runner for teardown
        self._saved_runner: Any = None

    # ------------------------------------------------------------------
    # Mock registration
    # ------------------------------------------------------------------

    def add_mock(self, mock: MockTool) -> None:
        """Register a :class:`MockTool` for use during this sandbox run.

        Must be called before ``sandbox.run()`` or inside the ``async with``
        block before the first run.

        Args:
            mock: The mock tool to register.
        """
        self._mocks[mock.name] = mock
        # If dispatcher is already built, update it in place
        if self._dispatcher is not None:
            self._dispatcher._source._mocks[mock.name] = mock

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(
        self,
        message: str,
        *,
        thread_id: Optional[str] = None,
    ) -> SandboxResult:
        """Run the agent with the given message.

        Args:
            message:   User message to send to the agent.
            thread_id: Optional conversation ID for multi-turn tests.

        Returns:
            :class:`SandboxResult` with the final output and call log.
        """
        if self._dispatcher is None:
            await self._install()

        try:
            result = await self._agent.arun(message, thread_id=thread_id)
            return SandboxResult.from_agent_result(result)
        except Exception as exc:
            return SandboxResult(
                output="",
                thread_id=thread_id or "",
                error=exc,
            )

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        """All tool calls made during this sandbox session, in order.

        Each entry is ``{"tool": str, "args": dict, "result": str}``.
        """
        if self._dispatcher is None:
            return []
        return self._dispatcher.tool_calls

    @property
    def turn_count(self) -> int:
        """Number of LLM turns taken across all sandbox runs."""
        if self._mock_provider is None:
            return 0
        return self._mock_provider._index

    def assert_tool_called(self, name: str) -> None:
        """Assert that the named tool was called at least once.

        Args:
            name: Tool name.

        Raises:
            AssertionError: If the tool was never called.
        """
        calls = [c for c in self.tool_calls if c["tool"] == name]
        if not calls:
            all_called = [c["tool"] for c in self.tool_calls]
            raise AssertionError(
                f"Tool '{name}' was never called during the sandbox run.\n"
                f"  Tools called: {all_called}"
            )

    def assert_tool_call_count(self, name: str, n: int) -> None:
        """Assert that the named tool was called exactly *n* times.

        Args:
            name: Tool name.
            n:    Expected call count.

        Raises:
            AssertionError: If the actual call count differs from *n*.
        """
        calls = [c for c in self.tool_calls if c["tool"] == name]
        if len(calls) != n:
            raise AssertionError(
                f"Tool '{name}' was expected to be called {n} time(s), "
                f"but was called {len(calls)} time(s).\n"
                f"  Calls: {calls}"
            )

    def assert_no_tool_called(self, name: str) -> None:
        """Assert that the named tool was NOT called.

        Args:
            name: Tool name.

        Raises:
            AssertionError: If the tool was called.
        """
        calls = [c for c in self.tool_calls if c["tool"] == name]
        if calls:
            raise AssertionError(
                f"Tool '{name}' was expected NOT to be called, but was called "
                f"{len(calls)} time(s).\n"
                f"  Calls: {calls}"
            )

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "AgentSandbox":
        await self._install()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self._restore()

    # ------------------------------------------------------------------
    # Install / restore
    # ------------------------------------------------------------------

    async def _install(self) -> None:
        """Inject mock provider and dispatcher into the agent's runner."""
        # Build mock provider
        if self._custom_provider is not None:
            self._mock_provider = self._custom_provider
        else:
            self._mock_provider = MockProvider(
                script=self._script,
                tool_calls_script=self._tool_calls_script,
            )

        # Build sandboxed dispatcher
        source = SandboxedToolSource(dict(self._mocks))
        self._dispatcher = SandboxedDispatcher(source)

        # Ensure agent has a runner built so we can swap its internals
        runner = await self._agent._get_runner()
        self._saved_runner = runner

        # Inject our mock provider and dispatcher into the runner
        runner.provider = self._mock_provider
        runner.dispatcher = self._dispatcher

    async def _restore(self) -> None:
        """Restore the original runner state after the test."""
        if self._saved_runner is not None:
            # Invalidate the runner so it gets rebuilt fresh next time.
            # We don't try to restore the original provider/dispatcher because
            # they may be stateful (e.g. they hold API connections).
            self._agent.invalidate_runner()

    def __repr__(self) -> str:
        return (
            f"AgentSandbox(agent={self._agent.name!r}, "
            f"mocks={list(self._mocks.keys())}, "
            f"tool_calls={len(self.tool_calls)})"
        )
