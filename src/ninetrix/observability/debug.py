"""
observability/debug.py — Human-friendly debug trace for Audience 1.

Call enable_debug() to activate. Subscribes to the agent's EventBus and
pretty-prints every lifecycle event in a structured, colored-friendly format.

Also sets the root ninetrix logger to DEBUG so stdlib log calls in
runner.py / dispatcher.py become visible.

Usage::

    from ninetrix import enable_debug
    enable_debug()
    result = agent.run("analyse this")
    # → structured trace printed to stderr

Or with the pretty-printer attached to a specific agent::

    from ninetrix import enable_debug
    enable_debug(agent=agent)
    result = agent.run("analyse this")
    # → structured trace + per-event pretty output to stderr
"""

from __future__ import annotations

import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Module-level timing state
# ---------------------------------------------------------------------------

# thread_id → run start time (float epoch seconds)
_run_times: dict[str, float] = {}

# "thread_id:tool_name" → tool call start time (float epoch seconds)
_tool_times: dict[str, float] = {}


# ---------------------------------------------------------------------------
# Pretty-printer handler
# ---------------------------------------------------------------------------


def _debug_handler(event: Any) -> None:
    """Pretty-print a single AgentEvent to stderr.

    Handles all standard lifecycle event types.  Unknown types are printed
    with a generic format.  All output goes to sys.stderr directly to avoid
    double-output through the logging subsystem.
    """
    etype = event.type
    agent = event.agent_name or "agent"
    tid = event.thread_id or ""
    data = event.data or {}

    # Short thread_id prefix for readability
    tid_short = tid[:6] if tid else "------"

    if etype == "run.start":
        _run_times[tid] = time.monotonic()
        model = data.get("model", "")
        print(
            f"[agent:{agent}] run started  model={model}  thread={tid_short}",
            file=sys.stderr,
        )

    elif etype == "turn.start":
        turn = data.get("turn", 0)
        print(f"[agent:{agent}]   turn {turn + 1}", file=sys.stderr)

    elif etype == "tool.call":
        tool_name = data.get("tool_name", "")
        args = data.get("arguments", {})
        key = f"{tid}:{tool_name}"
        _tool_times[key] = time.monotonic()
        # Truncate args for display
        args_str = str(args)
        if len(args_str) > 80:
            args_str = args_str[:80] + "..."
        print(
            f"[agent:{agent}]   -> tool:{tool_name}  args={args_str}",
            file=sys.stderr,
        )

    elif etype == "tool.result":
        tool_name = data.get("tool_name", "")
        result = data.get("result", "")
        key = f"{tid}:{tool_name}"
        elapsed_ms = ""
        if key in _tool_times:
            elapsed = (time.monotonic() - _tool_times.pop(key)) * 1000
            elapsed_ms = f"  ({elapsed:.0f}ms)"
        # Truncate result preview
        preview = str(result)[:120]
        print(
            f"[agent:{agent}]   <- tool:{tool_name}{elapsed_ms}  {preview}",
            file=sys.stderr,
        )

    elif etype == "turn.end":
        turn = data.get("turn", 0)
        in_tok = data.get("input_tokens", 0)
        out_tok = data.get("output_tokens", 0)
        # Rough cost estimate — just show token counts; full cost is at run.end
        print(
            f"[agent:{agent}]   turn {turn + 1} done  in={in_tok} out={out_tok} tokens",
            file=sys.stderr,
        )

    elif etype == "run.end":
        tokens = data.get("tokens_used", 0)
        cost = data.get("cost_usd", 0.0)
        steps = data.get("steps", 0)
        elapsed_str = ""
        if tid in _run_times:
            elapsed = time.monotonic() - _run_times.pop(tid)
            elapsed_str = f"  elapsed={elapsed:.1f}s"
        print(
            f"[agent:{agent}] run done  turns={steps}  {tokens} tokens  "
            f"${cost:.4f}{elapsed_str}",
            file=sys.stderr,
        )

    elif etype == "error":
        msg = data.get("message", data.get("error", str(data)))
        # Clean up timing state on error
        _run_times.pop(tid, None)
        print(f"[agent:{agent}] error: {msg}", file=sys.stderr)

    else:
        # Generic fallback for unknown event types
        print(f"[agent:{agent}] {etype}  {data}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def attach_debug_listener(event_bus: Any) -> None:
    """Subscribe the pretty-printer handler to *event_bus*.

    Subscribes to the global wildcard ``"*"`` so every lifecycle event
    triggers the pretty-printer.

    Args:
        event_bus: An :class:`~ninetrix.observability.events.EventBus` instance.

    Example::

        from ninetrix import Agent
        from ninetrix.observability.debug import attach_debug_listener

        agent = Agent(provider="anthropic", model="claude-sonnet-4-6")
        attach_debug_listener(agent._event_bus)
        result = agent.run("hello")
    """
    event_bus.subscribe("*", _debug_handler)
