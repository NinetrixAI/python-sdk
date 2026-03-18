"""
runtime/streaming.py — agentic loop as AsyncIterator[StreamEvent].

Layer: L3 (runtime) — may import L1 (_internals) + L2 (tools) + stdlib only.

StreamingRunner drives the same agentic loop as AgentRunner but yields
StreamEvent objects progressively instead of returning a single AgentResult.

Uses provider.complete() for LLM calls.  True token-level streaming via
provider.stream() will be wired when all provider adapters implement it.
"""

from __future__ import annotations

import uuid
from typing import Any, AsyncIterator, Optional

from ninetrix._internals.types import (
    BudgetExceededError,
    LLMProviderAdapter,
    StreamEvent,
)
from ninetrix.runtime.budget import BudgetTracker
from ninetrix.runtime.dispatcher import ToolDispatcher
from ninetrix.runtime.history import MessageHistory
from ninetrix.runtime.runner import (
    RunnerConfig,
    _build_assistant_message,
    _dispatch_tool,
    _get_output_schema,
)


class StreamingRunner:
    """Agentic loop variant that yields :class:`~ninetrix.StreamEvent` objects.

    Shares the same config + dependency model as :class:`~ninetrix.AgentRunner`
    but exposes an async-generator interface instead of returning a single
    :class:`~ninetrix.AgentResult`.

    Event sequence for a single-turn, no-tool run::

        token(content="Hello!")
        turn_end(tokens_used=12, cost_usd=0.00003)
        done(content="Hello!", tokens_used=12, cost_usd=0.00003)

    Event sequence for a run with one tool call::

        tool_start(tool_name="search", tool_args={...})
        tool_end(tool_name="search", tool_result="...")
        token(content="Based on the results...")
        turn_end(...)
        done(...)

    On any unhandled exception a single ``error`` event is yielded and the
    generator terminates normally (no re-raise).

    Args:
        provider:   Injected LLM adapter.
        dispatcher: Tool dispatcher with all sources initialised.
        config:     Runner configuration (same :class:`RunnerConfig` as AgentRunner).
        event_bus:  Optional :class:`~ninetrix.EventBus` — lifecycle events are
                    also emitted here in addition to being yielded.
    """

    def __init__(
        self,
        provider: LLMProviderAdapter,
        dispatcher: ToolDispatcher,
        config: RunnerConfig,
        event_bus: Any = None,
    ) -> None:
        self.provider = provider
        self.dispatcher = dispatcher
        self.config = config
        self.event_bus = event_bus

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def stream(
        self,
        message: str,
        *,
        thread_id: Optional[str] = None,
        attachments: Optional[list] = None,
        tool_context: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Run the agent and yield :class:`~ninetrix.StreamEvent` objects.

        Args:
            message:      The user's input message.
            thread_id:    Conversation ID; generated when ``None``.
            attachments:  Image / document attachments for the first turn.
            tool_context: Per-request context dict injected into tool calls.

        Yields:
            :class:`~ninetrix.StreamEvent` instances in order:
            ``token`` → ``tool_start`` / ``tool_end`` → ``turn_end`` → ``done``
            (or ``error`` on failure).
        """
        thread_id = thread_id or uuid.uuid4().hex[:16]
        budget = BudgetTracker(max_usd=self.config.max_budget_usd)

        history = MessageHistory(max_tokens=self.config.history_max_tokens)
        if self.config.system_prompt:
            history.append({"role": "system", "content": self.config.system_prompt})
        history.append({"role": "user", "content": message})

        tools = self.dispatcher.all_tool_definitions()
        output_schema = _get_output_schema(self.config.output_type)
        pconfig = self.config.effective_provider_config()

        final_content = ""
        steps = 0

        try:
            for turn in range(self.config.max_turns):
                trimmed = history.trim()

                response = await self.provider.complete(
                    trimmed,
                    tools,
                    attachments=attachments if turn == 0 else None,
                    output_schema=output_schema,
                    config=pconfig,
                )
                steps += 1

                budget.charge(
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    model=self.config.model,
                )

                history.append(_build_assistant_message(response))

                # ── No tool calls — final turn ─────────────────────────
                if not response.tool_calls:
                    final_content = response.content or ""
                    if final_content:
                        yield StreamEvent(
                            type="token",
                            content=final_content,
                            tokens_used=response.output_tokens,
                        )
                    usage = budget.usage()
                    yield StreamEvent(
                        type="turn_end",
                        tokens_used=usage.total_tokens,
                        cost_usd=usage.cost_usd,
                    )
                    break

                # ── Tool calls ─────────────────────────────────────────
                for tc in response.tool_calls:
                    yield StreamEvent(
                        type="tool_start",
                        tool_name=tc.name,
                        tool_args=tc.arguments,
                    )
                    result_str = await _dispatch_tool(
                        dispatcher=self.dispatcher,
                        tool_call=tc,
                        thread_id=thread_id,
                        agent_name=self.config.name,
                        turn_index=turn,
                        extra_context=tool_context,
                        timeout=self.config.tool_timeout,
                    )
                    yield StreamEvent(
                        type="tool_end",
                        tool_name=tc.name,
                        tool_result=result_str,
                    )
                    history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    })

            else:
                # Reached max_turns without a final text response
                final_content = "[Agent reached maximum turn limit without a final answer.]"

        except Exception as exc:
            yield StreamEvent(type="error", error=exc)
            return

        # ── Structured output parse (best-effort for streaming) ────────
        structured_output: Any = None
        if self.config.output_type is not None and final_content:
            try:
                structured_output = self.config.output_type.model_validate_json(
                    final_content
                )
            except Exception:
                pass  # streaming never raises on parse failure

        usage = budget.usage()
        yield StreamEvent(
            type="done",
            content=final_content,
            tokens_used=usage.total_tokens,
            cost_usd=usage.cost_usd,
            structured_output=structured_output,
        )
