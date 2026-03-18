"""
runtime/runner.py — the agentic loop.

Layer: L3 (runtime) — may import L1 (_internals) + L2 (tools) + stdlib only.

AgentRunner faithfully mirrors the agentic loop in the CLI's
``entrypoint.py.j2`` / ``_run_turn.py.j2`` templates so that SDK-run agents
and Docker-run agents behave identically.

Supports:
  - Direct execution mode  (multi-turn tool-use loop)
  - Structured output (Pydantic parse + retry, owner of the retry loop)
  - Per-run token budget enforcement
  - Checkpoint save/restore (via CheckpointerProtocol)
  - Per-call tool context injection (forwarded to ToolDispatcher)
  - Graceful max-turns exit
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from ninetrix._internals.types import (
    AgentResult,
    CheckpointerProtocol,
    LLMProviderAdapter,
    OutputParseError,
    ProviderConfig,
)
from ninetrix.runtime.dispatcher import ToolDispatcher
from ninetrix.runtime.history import MessageHistory
from ninetrix.runtime.budget import BudgetTracker


# ---------------------------------------------------------------------------
# RunnerConfig — pure data, no behavior
# ---------------------------------------------------------------------------

@dataclass
class RunnerConfig:
    """Configuration consumed by :class:`AgentRunner`.

    This is the *minimal* config the runner needs.  The full ``AgentConfig``
    (PR 18) will wrap or build from this.

    Args:
        name:            Agent name (used in ToolContext + checkpointer key).
        system_prompt:   System message prepended to every conversation.
        model:           Model identifier used for cost tracking.
        provider_name:   Provider name (informational; actual calls go via
                         the injected ``LLMProviderAdapter``).
        max_turns:       Maximum tool-use iterations before aborting.
        max_tokens:      Max completion tokens per LLM call.
        temperature:     Sampling temperature.
        tool_timeout:    Seconds before a tool call is cancelled.
        max_budget_usd:  Hard USD ceiling; ``None`` for unlimited.
        output_type:     Pydantic ``BaseModel`` subclass for structured output.
                         ``None`` → raw string output.
        output_retries:  Number of structured-output correction attempts.
        provider_config: Provider-specific parameters.  When ``None`` a
                         default is built from *temperature* + *max_tokens*.
        history_max_tokens: Token budget for the sliding message window.
    """

    name: str = "agent"
    system_prompt: str = ""
    model: str = "claude-sonnet-4-6"
    provider_name: str = "anthropic"
    max_turns: int = 20
    max_tokens: int = 8192
    temperature: float = 0.0
    tool_timeout: float = 30.0
    max_budget_usd: Optional[float] = None
    output_type: Any = None                 # Pydantic BaseModel subclass
    output_retries: int = 1
    provider_config: Optional[ProviderConfig] = None
    history_max_tokens: int = 128_000

    def effective_provider_config(self) -> ProviderConfig:
        """Return the provider config to use, building a default if not set."""
        if self.provider_config is not None:
            return self.provider_config
        return ProviderConfig(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


# ---------------------------------------------------------------------------
# AgentRunner
# ---------------------------------------------------------------------------

class AgentRunner:
    """Pure-Python agentic loop.  No Docker, no YAML, no Jinja2.

    Faithfully mirrors the behaviour of the CLI's ``_run_turn.py.j2`` template
    so that SDK-run agents and Docker-run agents behave identically.

    Args:
        provider:      Injected LLM adapter (AnthropicAdapter, OpenAIAdapter …).
        dispatcher:    Tool dispatcher with all sources initialised.
        config:        Runner configuration.
        checkpointer:  Optional persistence backend.
        event_bus:     Optional event bus (wired in PR 20).

    Example::

        from ninetrix.providers.anthropic import AnthropicAdapter
        from ninetrix.runtime.runner import AgentRunner, RunnerConfig
        from ninetrix.runtime.dispatcher import ToolDispatcher, LocalToolSource
        from ninetrix.registry import _registry

        provider = AnthropicAdapter(model="claude-sonnet-4-6")
        dispatcher = ToolDispatcher([LocalToolSource(_registry.all())])
        runner = AgentRunner(
            provider=provider,
            dispatcher=dispatcher,
            config=RunnerConfig(name="my-agent", system_prompt="You are helpful."),
        )
        result = await runner.run("What is 2 + 2?")
        print(result.output)   # "4"
    """

    def __init__(
        self,
        provider: LLMProviderAdapter,
        dispatcher: ToolDispatcher,
        config: RunnerConfig,
        checkpointer: Optional[CheckpointerProtocol] = None,
        event_bus: Any = None,                  # EventBus added in PR 20
    ) -> None:
        self.provider = provider
        self.dispatcher = dispatcher
        self.config = config
        self.checkpointer = checkpointer
        self.event_bus = event_bus              # reserved — not used until PR 20
        self._budget = BudgetTracker(max_usd=config.max_budget_usd)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        message: str,
        *,
        thread_id: Optional[str] = None,
        prior_history: Optional[list[dict]] = None,
        attachments: Optional[list] = None,
        tool_context: Optional[dict[str, Any]] = None,
    ) -> AgentResult:
        """Execute one full agent run and return the result.

        Args:
            message:       The user's input message.
            thread_id:     Conversation ID.  A new UUID is generated when
                           ``None`` and no prior thread is being resumed.
            prior_history: Messages to restore (overrides checkpointer load).
            attachments:   Image / document attachments for the first user turn.
            tool_context:  Per-request context dict injected into tool calls.

        Returns:
            :class:`~ninetrix.AgentResult` with the final output and telemetry.

        Raises:
            BudgetExceededError: Spending limit exceeded during the run.
            OutputParseError:    Structured output parse failed after all retries.
            ProviderError:       LLM API call failed.
        """
        thread_id = thread_id or uuid.uuid4().hex[:16]
        self._budget.reset()

        await self._emit("run.start", thread_id, {
            "agent_name": self.config.name,
            "model": self.config.model,
        })

        # ── Restore prior history ──────────────────────────────────────
        restored: list[dict] = []
        if prior_history is not None:
            restored = [m for m in prior_history if m.get("role") != "system"]
        elif self.checkpointer is not None:
            saved = await self.checkpointer.load(thread_id)
            if saved:
                restored = [
                    m for m in (saved.get("history") or [])
                    if m.get("role") != "system"
                ]

        # ── Assemble initial message list ─────────────────────────────
        history = MessageHistory(max_tokens=self.config.history_max_tokens)
        if self.config.system_prompt:
            history.append({"role": "system", "content": self.config.system_prompt})
        for msg in restored:
            history.append(msg)
        history.append({"role": "user", "content": message})

        # ── Tool definitions + output schema ───────────────────────────
        tools = self.dispatcher.all_tool_definitions()
        output_schema = _get_output_schema(self.config.output_type)
        pconfig = self.config.effective_provider_config()

        # ── Agentic loop ───────────────────────────────────────────────
        final_content = ""
        steps = 0

        for turn in range(self.config.max_turns):
            trimmed = history.trim()

            await self._emit("turn.start", thread_id, {"turn": turn})

            response = await self.provider.complete(
                trimmed,
                tools,
                attachments=attachments if turn == 0 else None,
                output_schema=output_schema,
                config=pconfig,
            )
            steps += 1

            # Charge budget (raises BudgetExceededError if over limit)
            self._budget.charge(
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                model=self.config.model,
            )

            # Append assistant message
            history.append(_build_assistant_message(response))

            # No tool calls → stop turn
            if not response.tool_calls:
                final_content = response.content or ""
                await self._emit("turn.end", thread_id, {
                    "turn": turn,
                    "content": final_content,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                })
                break

            # Dispatch tool calls
            for tc in response.tool_calls:
                await self._emit("tool.call", thread_id, {
                    "tool_name": tc.name,
                    "tool_call_id": tc.id,
                    "arguments": tc.arguments,
                    "turn": turn,
                })
                result_str = await _dispatch_tool(
                    dispatcher=self.dispatcher,
                    tool_call=tc,
                    thread_id=thread_id,
                    agent_name=self.config.name,
                    turn_index=turn,
                    extra_context=tool_context,
                    timeout=self.config.tool_timeout,
                )
                await self._emit("tool.result", thread_id, {
                    "tool_name": tc.name,
                    "tool_call_id": tc.id,
                    "result": result_str,
                    "turn": turn,
                })
                history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

        else:
            # Reached max_turns without a stop
            final_content = "[Agent reached maximum turn limit without a final answer.]"

        # ── Structured output parse ────────────────────────────────────
        parsed_output: Any = final_content
        if self.config.output_type is not None and final_content:
            parsed_output = await self._parse_structured_output(
                raw=final_content,
                history=history,
                tools=tools,
                pconfig=pconfig,
                output_schema=output_schema,
            )

        # ── Save checkpoint ────────────────────────────────────────────
        usage = self._budget.usage()
        final_history = [m for m in history.messages() if m.get("role") != "system"]

        if self.checkpointer is not None:
            await self.checkpointer.save(
                thread_id=thread_id,
                agent_id=self.config.name,
                history=final_history,
                tokens_used=usage.total_tokens,
            )

        await self._emit("run.end", thread_id, {
            "agent_name": self.config.name,
            "tokens_used": usage.total_tokens,
            "cost_usd": usage.cost_usd,
            "steps": steps,
        })

        return AgentResult(
            output=parsed_output,
            thread_id=thread_id,
            tokens_used=usage.total_tokens,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=usage.cost_usd,
            steps=steps,
            history=final_history,
        )

    # ------------------------------------------------------------------
    # Event emission helper
    # ------------------------------------------------------------------

    async def _emit(self, event_type: str, thread_id: str, data: dict) -> None:
        """Emit a lifecycle event if an event_bus is configured.

        Never raises — event handler errors are silently swallowed so they
        cannot interrupt the agent run.
        """
        if self.event_bus is None:
            return
        try:
            from ninetrix.observability.events import AgentEvent
            await self.event_bus.emit(AgentEvent(
                type=event_type,
                thread_id=thread_id,
                agent_name=self.config.name,
                data=data,
            ))
        except Exception:
            pass  # hook errors must never crash the agent

    # ------------------------------------------------------------------
    # Structured output parse + retry
    # ------------------------------------------------------------------

    async def _parse_structured_output(
        self,
        raw: str,
        history: MessageHistory,
        tools: list[dict],
        pconfig: ProviderConfig,
        output_schema: Optional[dict],
    ) -> Any:
        """Parse *raw* into ``config.output_type``, retrying on failure.

        Algorithm:
          1. Attempt ``output_type.model_validate_json(raw)``.
          2. On ``ValidationError``: append correction prompt, call provider
             once more, retry parse.
          3. On final failure: raise :exc:`~ninetrix.OutputParseError`.
        """
        output_type = self.config.output_type
        retries = self.config.output_retries

        # First attempt
        first_error: Exception | None = None
        try:
            return output_type.model_validate_json(raw)
        except Exception as exc:
            first_error = exc  # capture before Python deletes the except-var
            if retries <= 0:
                raise OutputParseError(
                    f"Failed to parse LLM output as {output_type.__name__}.\n"
                    f"  Why: {exc}\n"
                    "  Fix: check that the model is returning valid JSON, or "
                    "increase output_retries.",
                    raw_output=raw,
                    validation_errors=_extract_validation_errors(exc),
                    attempts=1,
                ) from exc

        # Retry: inject correction prompt
        correction_messages = list(history.messages()) + [
            {
                "role": "user",
                "content": (
                    "Your previous response could not be parsed as valid JSON. "
                    f"Error: {first_error}\n\n"
                    "Respond ONLY with valid JSON matching the required schema. "
                    "No explanation, no markdown, no code fences."
                ),
            }
        ]

        retry_response = await self.provider.complete(
            correction_messages,
            tools,
            output_schema=output_schema,
            config=pconfig,
        )
        self._budget.charge(
            input_tokens=retry_response.input_tokens,
            output_tokens=retry_response.output_tokens,
            model=self.config.model,
        )

        retry_raw = retry_response.content or ""
        try:
            return output_type.model_validate_json(retry_raw)
        except Exception as second_err:
            raise OutputParseError(
                f"Failed to parse LLM output as {output_type.__name__} after "
                f"{retries + 1} attempt(s).\n"
                f"  Last error: {second_err}\n"
                "  Fix: check that output_type schema is compatible with the "
                "model's output, or switch to a more capable model.",
                raw_output=retry_raw,
                validation_errors=_extract_validation_errors(second_err),
                attempts=retries + 1,
            ) from second_err


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_output_schema(output_type: Any) -> Optional[dict]:
    """Return the JSON schema dict for a Pydantic model type, or ``None``."""
    if output_type is None:
        return None
    if hasattr(output_type, "model_json_schema"):
        try:
            return output_type.model_json_schema()
        except Exception:
            pass
    return None


def _build_assistant_message(response: Any) -> dict:
    """Build an OpenAI-compatible assistant message dict from an LLMResponse."""
    tool_calls_payload = [
        {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
            },
        }
        for tc in (response.tool_calls or [])
    ]
    return {
        "role": "assistant",
        "content": response.content or "",
        "tool_calls": tool_calls_payload or None,
    }


async def _dispatch_tool(
    *,
    dispatcher: ToolDispatcher,
    tool_call: Any,
    thread_id: str,
    agent_name: str,
    turn_index: int,
    extra_context: Optional[dict[str, Any]],
    timeout: float,
) -> str:
    """Dispatch a single tool call with timeout + error capture."""
    from ninetrix.runtime.dispatcher import LocalToolSource

    try:
        result = await asyncio.wait_for(
            dispatcher.call(
                tool_call.name,
                tool_call.arguments,
                thread_id=thread_id,
                agent_name=agent_name,
                turn_index=turn_index,
                extra_context=extra_context,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        result = (
            f"Tool '{tool_call.name}' timed out after {timeout:.0f}s. "
            "The operation was cancelled."
        )
    except Exception as exc:
        result = f"Tool error: {type(exc).__name__}: {exc}"

    # Sanitise to safe ASCII (matches CLI template behaviour)
    result = result.encode("utf-8", "replace").decode("utf-8")
    if not result.isascii():
        result = result.encode("ascii", "backslashreplace").decode("ascii")

    return result


def _extract_validation_errors(exc: Exception) -> list[dict]:
    """Extract Pydantic validation error list, or return a single generic entry."""
    try:
        # pydantic v2 ValidationError has .errors()
        if hasattr(exc, "errors"):
            return exc.errors()  # type: ignore[return-value]
    except Exception:
        pass
    return [{"msg": str(exc)}]
