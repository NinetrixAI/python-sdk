"""Tests for runtime/history.py and runtime/budget.py — PR 14."""

from __future__ import annotations

import warnings

import pytest

from ninetrix.runtime.history import (
    MessageHistory,
    _filter_messages,
    _find_pinned,
    _trim_to_budget,
)
from ninetrix.runtime.budget import (
    BudgetTracker,
    BudgetUsage,
    _model_cost,
    _PRICE_FALLBACK,
)
from ninetrix._internals.types import BudgetExceededError


# =============================================================================
# MessageHistory — basic CRUD
# =============================================================================


def test_append_and_messages():
    h = MessageHistory()
    h.append({"role": "user", "content": "hello"})
    assert len(h.messages()) == 1
    assert h.messages()[0]["content"] == "hello"


def test_extend():
    h = MessageHistory()
    h.extend([
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ])
    assert len(h) == 2


def test_clear():
    h = MessageHistory()
    h.append({"role": "user", "content": "x"})
    h.clear()
    assert len(h) == 0


def test_messages_returns_copy():
    h = MessageHistory()
    h.append({"role": "user", "content": "x"})
    copy = h.messages()
    copy.append({"role": "user", "content": "y"})
    # Original should be unchanged
    assert len(h) == 1


def test_invalid_max_tokens():
    with pytest.raises(ValueError, match="max_tokens must be >= 1"):
        MessageHistory(max_tokens=0)


def test_repr():
    h = MessageHistory(max_tokens=4000)
    r = repr(h)
    assert "4000" in r
    assert "MessageHistory" in r


def test_token_count_empty():
    h = MessageHistory()
    assert h.token_count() == 0


# =============================================================================
# _filter_messages — thinking + orphan tool_result
# =============================================================================


def test_filter_removes_thinking():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "thinking", "content": "Let me think..."},
        {"role": "assistant", "content": "hello"},
    ]
    result = _filter_messages(msgs)
    assert len(result) == 2
    assert all(m["role"] != "thinking" for m in result)


def test_filter_keeps_paired_tool_use_result():
    msgs = [
        {"role": "user", "content": "search"},
        {"role": "tool_use", "content": "call"},
        {"role": "tool_result", "content": "result"},
    ]
    result = _filter_messages(msgs)
    assert len(result) == 3


def test_filter_drops_orphan_tool_result():
    msgs = [
        {"role": "user", "content": "search"},
        {"role": "tool_result", "content": "orphan"},  # no preceding tool_use
        {"role": "assistant", "content": "done"},
    ]
    result = _filter_messages(msgs)
    assert len(result) == 2
    assert all(m["role"] != "tool_result" for m in result)


def test_filter_empty():
    assert _filter_messages([]) == []


def test_filter_keeps_all_valid():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    assert _filter_messages(msgs) == msgs


# =============================================================================
# _find_pinned
# =============================================================================


def test_find_pinned_system_and_first_user():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "u2"},
    ]
    pinned = _find_pinned(msgs)
    assert 0 in pinned  # system
    assert 1 in pinned  # first user
    assert 2 not in pinned
    assert 3 not in pinned  # second user not pinned


def test_find_pinned_no_system():
    msgs = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    pinned = _find_pinned(msgs)
    assert 0 in pinned
    assert 1 not in pinned


def test_find_pinned_empty():
    assert _find_pinned([]) == set()


# =============================================================================
# _trim_to_budget
# =============================================================================


def test_trim_empty():
    assert _trim_to_budget([], 1000) == []


def test_trim_fits_budget():
    msgs = [{"role": "user", "content": "hi"}]
    # Should return unchanged
    result = _trim_to_budget(msgs, 10_000)
    assert result == msgs


def test_trim_removes_oldest_non_pinned():
    # System + first-user are pinned; assistant messages should be removed first
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a1"},
        {"role": "assistant", "content": "a2"},
        {"role": "assistant", "content": "a3"},
    ]
    # Very tight budget — should strip older assistant messages
    result = _trim_to_budget(msgs, 50)
    # System and first user must survive
    roles = [m["role"] for m in result]
    assert "system" in roles
    assert result[1]["role"] == "user"


def test_trim_never_removes_system():
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hello"},
    ]
    # Extremely tight budget — only pinned messages; they must be preserved
    result = _trim_to_budget(msgs, 1)
    roles = [m["role"] for m in result]
    assert "system" in roles
    assert "user" in roles


def test_trim_drops_tool_result_with_tool_use():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "tool_use", "content": "call"},
        {"role": "tool_result", "content": "result"},
    ]
    # Tight budget: tool_use + tool_result should be dropped together
    result = _trim_to_budget(msgs, 20)
    roles = [m["role"] for m in result]
    assert "tool_use" not in roles
    assert "tool_result" not in roles


# =============================================================================
# MessageHistory.trim()
# =============================================================================


def test_trim_method_does_not_mutate():
    h = MessageHistory(max_tokens=100)
    h.append({"role": "system", "content": "s"})
    h.append({"role": "user", "content": "u"})
    h.append({"role": "assistant", "content": "a" * 500})
    original_len = len(h)
    h.trim()
    assert len(h) == original_len  # internal list unchanged


def test_trim_method_returns_list():
    h = MessageHistory()
    h.append({"role": "user", "content": "hi"})
    result = h.trim()
    assert isinstance(result, list)


# =============================================================================
# BudgetTracker — _model_cost
# =============================================================================


def test_model_cost_exact_match():
    price = _model_cost("claude-sonnet-4-6")
    assert price == (3.00, 15.00)


def test_model_cost_provider_prefixed():
    price = _model_cost("anthropic/claude-sonnet-4-6")
    assert price == (3.00, 15.00)


def test_model_cost_prefix_match():
    # A version-suffixed name should match the base
    price = _model_cost("claude-haiku-4-5-20251001")
    assert price == (0.25, 1.25)


def test_model_cost_fallback():
    price = _model_cost("unknown-model-xyz")
    assert price == _PRICE_FALLBACK


def test_model_cost_gpt4o():
    assert _model_cost("gpt-4o") == (5.00, 15.00)


def test_model_cost_gemini():
    assert _model_cost("gemini-1.5-pro") == (1.25, 5.00)


# =============================================================================
# BudgetTracker — basic charging
# =============================================================================


def test_charge_returns_cost():
    tracker = BudgetTracker()
    cost = tracker.charge(input_tokens=1_000_000, output_tokens=0, model="claude-sonnet-4-6")
    assert cost == pytest.approx(3.00)


def test_charge_accumulates():
    tracker = BudgetTracker()
    tracker.charge(100, 50, model="gpt-4o-mini")
    tracker.charge(200, 100, model="gpt-4o-mini")
    usage = tracker.usage()
    assert usage.input_tokens == 300
    assert usage.output_tokens == 150


def test_charge_output_tokens():
    tracker = BudgetTracker()
    cost = tracker.charge(input_tokens=0, output_tokens=1_000_000, model="claude-sonnet-4-6")
    assert cost == pytest.approx(15.00)


def test_usage_snapshot():
    tracker = BudgetTracker(max_usd=5.0)
    tracker.charge(100, 50, model="gpt-4o")
    usage = tracker.usage()
    assert isinstance(usage, BudgetUsage)
    assert usage.max_usd == 5.0
    assert usage.total_tokens == 150
    assert usage.remaining_usd is not None
    assert usage.cost_usd > 0


def test_usage_unlimited():
    tracker = BudgetTracker()
    usage = tracker.usage()
    assert usage.max_usd is None
    assert usage.remaining_usd is None


def test_reset():
    tracker = BudgetTracker(max_usd=10.0)
    tracker.charge(100, 50, model="gpt-4o")
    tracker.reset()
    usage = tracker.usage()
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.cost_usd == 0.0


# =============================================================================
# BudgetTracker — budget enforcement
# =============================================================================


def test_budget_exceeded_raises():
    tracker = BudgetTracker(max_usd=0.001)  # tiny budget
    with pytest.raises(BudgetExceededError, match="Budget exceeded"):
        # Charge 1M tokens of claude-sonnet-4-6 at $3/M input = $3 > $0.001
        tracker.charge(1_000_000, 0, model="claude-sonnet-4-6")


def test_budget_not_exceeded_under_limit():
    tracker = BudgetTracker(max_usd=100.0)
    # 100 tokens at $3/M = $0.0003 — well under budget
    tracker.charge(100, 0, model="claude-sonnet-4-6")
    assert tracker.usage().cost_usd < 100.0


def test_budget_warning_fires():
    fired = []

    def on_warn(usage: BudgetUsage) -> None:
        fired.append(usage)

    tracker = BudgetTracker(max_usd=1.0, budget_warning=0.5)
    tracker.on_warning(on_warn)

    # Charge enough to exceed 50% of $1.00
    # gpt-4o: $5/M input — 110k tokens = $0.55 > $0.50
    tracker.charge(110_000, 0, model="gpt-4o")

    assert len(fired) == 1
    assert fired[0].cost_usd >= 0.5


def test_budget_warning_fires_only_once():
    fired = []

    def on_warn(usage: BudgetUsage) -> None:
        fired.append(usage)

    tracker = BudgetTracker(max_usd=1.0, budget_warning=0.5)
    tracker.on_warning(on_warn)

    # Two charges that each cross the threshold
    tracker.charge(110_000, 0, model="gpt-4o")
    tracker.charge(10_000, 0, model="gpt-4o")

    assert len(fired) == 1  # only fires once


def test_budget_warning_stdlib_warning():
    tracker = BudgetTracker(max_usd=1.0, budget_warning=0.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tracker.charge(110_000, 0, model="gpt-4o")
    assert any("Budget warning" in str(w.message) for w in caught)


def test_invalid_max_usd():
    with pytest.raises(ValueError, match="max_usd must be > 0"):
        BudgetTracker(max_usd=0.0)

    with pytest.raises(ValueError, match="max_usd must be > 0"):
        BudgetTracker(max_usd=-1.0)


def test_invalid_budget_warning():
    with pytest.raises(ValueError, match="budget_warning must be between"):
        BudgetTracker(max_usd=1.0, budget_warning=1.5)


# =============================================================================
# BudgetUsage helpers
# =============================================================================


def test_budget_usage_remaining():
    usage = BudgetUsage(input_tokens=0, output_tokens=0, cost_usd=0.3, max_usd=1.0)
    assert usage.remaining_usd == pytest.approx(0.7)


def test_budget_usage_remaining_clamped_at_zero():
    # Over-budget snapshot (can happen if charged right at limit)
    usage = BudgetUsage(input_tokens=0, output_tokens=0, cost_usd=1.5, max_usd=1.0)
    assert usage.remaining_usd == 0.0


def test_budget_usage_total_tokens():
    usage = BudgetUsage(input_tokens=300, output_tokens=100, cost_usd=0.0, max_usd=None)
    assert usage.total_tokens == 400


def test_repr():
    tracker = BudgetTracker(max_usd=5.0)
    r = repr(tracker)
    assert "BudgetTracker" in r
    assert "5.0" in r
