"""
runtime/history.py — sliding-window message history with token budget.

Layer: L3 (runtime) — may import _internals + stdlib only.

Pinning rules (mirrors the CLI entrypoint template):
  1. System message is always kept (role == "system").
  2. First user message is always kept (prevents context-free replies).
  3. "thinking" role messages are always dropped.
  4. Orphan tool_result messages (no preceding tool_use) are dropped.
  5. When trimming, oldest non-pinned messages are removed first.
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Token counting — best-effort, no hard dependency on litellm
# ---------------------------------------------------------------------------

def _count_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate token count for a list of messages.

    Tries litellm first (accurate), falls back to character-based heuristic
    (1 token ≈ 4 chars) — good enough for budget decisions.
    """
    try:
        import litellm  # type: ignore[import]
        # litellm.token_counter accepts model + messages
        return litellm.token_counter(model="gpt-3.5-turbo", messages=messages)
    except Exception:
        # Rough fallback: count characters in JSON serialization
        total_chars = len(json.dumps(messages, ensure_ascii=False))
        return max(1, total_chars // 4)


# ---------------------------------------------------------------------------
# MessageHistory
# ---------------------------------------------------------------------------

class MessageHistory:
    """Sliding-window message store with pinned-message protection.

    Args:
        max_tokens: Hard token ceiling for the history window.  When
            ``trim()`` is called the history is reduced until it fits.
            Defaults to 8 000 tokens (safe for most 16 k-context models).

    The history is a plain list of OpenAI-compatible message dicts::

        {"role": "system" | "user" | "assistant" | "tool_use" | "tool_result", "content": ...}

    Example::

        history = MessageHistory(max_tokens=8000)
        history.append({"role": "system", "content": "You are helpful."})
        history.append({"role": "user", "content": "Hello"})
        trimmed = history.messages()   # returns list, always fits budget
    """

    def __init__(self, max_tokens: int = 8_000) -> None:
        if max_tokens < 1:
            raise ValueError(
                f"max_tokens must be >= 1, got {max_tokens}. "
                "Why: a zero or negative limit would discard all messages. "
                "Fix: pass a positive integer, e.g. MessageHistory(max_tokens=8000)."
            )
        self._max_tokens = max_tokens
        self._messages: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def append(self, message: dict[str, Any]) -> None:
        """Append a single message dict to the history."""
        self._messages.append(message)

    def extend(self, messages: list[dict[str, Any]]) -> None:
        """Append multiple message dicts."""
        self._messages.extend(messages)

    def clear(self) -> None:
        """Remove all messages."""
        self._messages.clear()

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    def messages(self) -> list[dict[str, Any]]:
        """Return the current message list (not trimmed).

        Call ``trim()`` first if you need a budget-respecting window.
        """
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    # ------------------------------------------------------------------
    # Trimming
    # ------------------------------------------------------------------

    def trim(self) -> list[dict[str, Any]]:
        """Return a trimmed copy of the history that fits within *max_tokens*.

        The original ``_messages`` list is **not** mutated — the trimmed
        window is only used for the current LLM call.

        Algorithm:
          1. Drop all messages with ``role == "thinking"``.
          2. Drop orphan ``tool_result`` messages (no preceding ``tool_use``).
          3. If total tokens still exceed the budget, remove the oldest
             non-pinned message and repeat.  System + first-user are pinned.
        """
        filtered = _filter_messages(self._messages)
        return _trim_to_budget(filtered, self._max_tokens)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def token_count(self) -> int:
        """Estimate current token usage (before trimming)."""
        if not self._messages:
            return 0
        return _count_tokens(self._messages)

    def __repr__(self) -> str:
        return (
            f"MessageHistory(messages={len(self._messages)}, "
            f"max_tokens={self._max_tokens})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _filter_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop thinking messages and orphan tool_results."""
    # Pass 1: remove "thinking" role
    without_thinking = [m for m in messages if m.get("role") != "thinking"]

    # Pass 2: remove orphan tool_result (no preceding tool_use in same turn)
    result: list[dict[str, Any]] = []
    for msg in without_thinking:
        if msg.get("role") == "tool_result":
            # Keep only if the immediately preceding message is tool_use
            if result and result[-1].get("role") == "tool_use":
                result.append(msg)
            # else: orphan — drop silently
        else:
            result.append(msg)
    return result


def _trim_to_budget(
    messages: list[dict[str, Any]],
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Trim *messages* until they fit within *max_tokens*.

    Pinned messages (system + first user) are never removed.
    Oldest non-pinned messages are removed first.
    """
    if not messages:
        return []

    # Identify pinned indices
    pinned: set[int] = set()
    first_user_found = False
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            pinned.add(i)
        elif msg.get("role") == "user" and not first_user_found:
            pinned.add(i)
            first_user_found = True

    window = list(messages)

    while True:
        tokens = _count_tokens(window)
        if tokens <= max_tokens:
            break

        # Find oldest non-pinned message
        # Indices in 'window' shift as we remove — recalculate pinned each iteration
        original_indices = list(range(len(window)))
        drop_idx = None
        for local_i, orig_i in enumerate(original_indices):
            if orig_i not in pinned:
                drop_idx = local_i
                break

        if drop_idx is None:
            # Only pinned messages remain — can't trim further
            break

        # Also drop the following tool_result if this was a tool_use
        if (
            window[drop_idx].get("role") == "tool_use"
            and drop_idx + 1 < len(window)
            and window[drop_idx + 1].get("role") == "tool_result"
        ):
            window.pop(drop_idx + 1)

        window.pop(drop_idx)

        # Rebuild original_indices tracking (pinned set stays based on original positions,
        # but we're working on a shrinking copy — just break if only pinned remain)
        # After removal, re-identify pinned in the new window
        pinned = _find_pinned(window)

    return window


def _find_pinned(messages: list[dict[str, Any]]) -> set[int]:
    """Return the set of indices that must never be removed."""
    pinned: set[int] = set()
    first_user_found = False
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            pinned.add(i)
        elif msg.get("role") == "user" and not first_user_found:
            pinned.add(i)
            first_user_found = True
    return pinned
