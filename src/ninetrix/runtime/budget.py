"""
runtime/budget.py — per-run cost tracking with model pricing table.

Layer: L3 (runtime) — may import _internals + stdlib only.

Pricing is in USD per million tokens (input, output).
The table mirrors the CLI entrypoint template so costs are consistent
across the runner and the SDK.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from ninetrix._internals.types import BudgetExceededError

# ---------------------------------------------------------------------------
# Model pricing table  (USD / 1M tokens)
# ---------------------------------------------------------------------------

# (input_per_M, output_per_M)
_MODEL_PRICES: dict[str, tuple[float, float]] = {
    # Anthropic Claude
    "claude-opus-4-6":       (15.00, 75.00),
    "claude-sonnet-4-6":     (3.00,  15.00),
    "claude-haiku-4-5":      (0.25,   1.25),
    "claude-opus-4-5":       (15.00, 75.00),
    "claude-sonnet-4-5":     (3.00,  15.00),
    "claude-haiku-4-5-20251001": (0.25, 1.25),
    # OpenAI
    "gpt-4o":                (5.00,  15.00),
    "gpt-4o-mini":           (0.15,   0.60),
    "gpt-4-turbo":           (10.00, 30.00),
    "o1":                    (15.00, 60.00),
    "o1-mini":               (3.00,  12.00),
    "o3-mini":               (1.10,   4.40),
    # Google
    "gemini-1.5-pro":        (1.25,   5.00),
    "gemini-1.5-flash":      (0.075,  0.30),
    "gemini-2.0-flash":      (0.10,   0.40),
    "gemini-2.5-pro":        (1.25,   10.00),
    "gemini-2.5-flash":      (0.075,  0.30),
}

# Fallback when model is not found in the table
_PRICE_FALLBACK: tuple[float, float] = (1.00, 3.00)


def _model_cost(model: str, provider: Optional[str] = None) -> tuple[float, float]:
    """Return (input_per_M, output_per_M) for the given model name.

    Handles:
    - Bare model names: ``"claude-sonnet-4-6"``
    - Provider-prefixed names: ``"anthropic/claude-sonnet-4-6"``
    - Prefix matching: ``"claude-sonnet-4-6-20251022"`` → ``"claude-sonnet-4-6"``
    """
    # Strip provider prefix if present (e.g. "anthropic/claude-sonnet-4-6")
    if "/" in model:
        model = model.split("/", 1)[1]

    # Exact match
    if model in _MODEL_PRICES:
        return _MODEL_PRICES[model]

    # Prefix match — find the longest key that is a prefix of the model name
    best_key = ""
    for key in _MODEL_PRICES:
        if model.startswith(key) and len(key) > len(best_key):
            best_key = key

    if best_key:
        return _MODEL_PRICES[best_key]

    return _PRICE_FALLBACK


# ---------------------------------------------------------------------------
# BudgetUsage — snapshot returned to callers
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BudgetUsage:
    """Immutable snapshot of accumulated spend.

    Attributes:
        input_tokens: Total prompt tokens charged so far.
        output_tokens: Total completion tokens charged so far.
        cost_usd: Total cost in US dollars.
        max_usd: Configured budget ceiling (None = unlimited).
        remaining_usd: Remaining budget (None if no ceiling).
    """

    input_tokens: int
    output_tokens: int
    cost_usd: float
    max_usd: Optional[float]

    @property
    def remaining_usd(self) -> Optional[float]:
        if self.max_usd is None:
            return None
        return max(0.0, self.max_usd - self.cost_usd)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# BudgetTracker
# ---------------------------------------------------------------------------

class BudgetTracker:
    """Track token costs and enforce a USD spending limit.

    Args:
        max_usd: Hard spending ceiling in USD.  Raises
            :exc:`~ninetrix.BudgetExceededError` when exceeded.
            ``None`` means unlimited.
        budget_warning: Fraction of *max_usd* at which a warning is emitted
            (default ``0.8`` → 80 %).  Set to ``1.0`` to disable warnings.

    Example::

        tracker = BudgetTracker(max_usd=1.0)
        tracker.charge(input_tokens=500, output_tokens=200, model="claude-sonnet-4-6")
        usage = tracker.usage()
        print(f"Spent ${usage.cost_usd:.4f} of ${usage.max_usd:.2f}")
    """

    def __init__(
        self,
        max_usd: Optional[float] = None,
        budget_warning: float = 0.8,
    ) -> None:
        if max_usd is not None and max_usd <= 0:
            raise ValueError(
                f"max_usd must be > 0, got {max_usd}. "
                "Why: a zero or negative budget would reject every call. "
                "Fix: pass a positive number like BudgetTracker(max_usd=1.0), "
                "or omit it for unlimited spending."
            )
        if not (0.0 <= budget_warning <= 1.0):
            raise ValueError(
                f"budget_warning must be between 0 and 1, got {budget_warning}. "
                "Fix: use a fraction like 0.8 for 80% warning threshold."
            )

        self._max_usd = max_usd
        self._budget_warning = budget_warning
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._cost_usd: float = 0.0
        self._warning_fired: bool = False
        self._warning_callback: Optional[object] = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def charge(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        provider: Optional[str] = None,
    ) -> float:
        """Record token usage and return the cost of this call in USD.

        Args:
            input_tokens: Number of prompt tokens consumed.
            output_tokens: Number of completion tokens consumed.
            model: Model name (bare or provider-prefixed).
            provider: Optional provider hint (e.g. ``"anthropic"``).

        Returns:
            The USD cost charged for this single call.

        Raises:
            BudgetExceededError: If the accumulated spend exceeds *max_usd*.
        """
        input_per_m, output_per_m = _model_cost(model, provider)
        call_cost = (input_tokens * input_per_m + output_tokens * output_per_m) / 1_000_000

        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        self._cost_usd += call_cost

        # Warning at threshold
        if (
            self._max_usd is not None
            and not self._warning_fired
            and self._cost_usd >= self._max_usd * self._budget_warning
        ):
            self._warning_fired = True
            self._fire_warning()

        # Hard limit
        if self._max_usd is not None and self._cost_usd > self._max_usd:
            raise BudgetExceededError(
                f"Budget exceeded: spent ${self._cost_usd:.4f} of "
                f"${self._max_usd:.2f} limit. "
                "Why: the agent consumed more tokens than the configured budget. "
                "Fix: increase max_usd, reduce prompt size, or use a cheaper model."
            )

        return call_cost

    def usage(self) -> BudgetUsage:
        """Return an immutable snapshot of current spend."""
        return BudgetUsage(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            cost_usd=self._cost_usd,
            max_usd=self._max_usd,
        )

    def reset(self) -> None:
        """Reset all counters to zero (e.g. between agent runs)."""
        self._input_tokens = 0
        self._output_tokens = 0
        self._cost_usd = 0.0
        self._warning_fired = False

    # ------------------------------------------------------------------
    # Warning hook
    # ------------------------------------------------------------------

    def on_warning(self, callback: object) -> None:
        """Register a callable that fires at the budget warning threshold.

        The callback receives one argument: the current :class:`BudgetUsage`.
        """
        self._warning_callback = callback

    def _fire_warning(self) -> None:
        import warnings
        usage = self.usage()
        warnings.warn(
            f"Budget warning: ${usage.cost_usd:.4f} spent of "
            f"${self._max_usd:.2f} limit ({self._budget_warning * 100:.0f}% threshold).",
            stacklevel=4,
        )
        if callable(self._warning_callback):
            self._warning_callback(usage)  # type: ignore[call-arg]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BudgetTracker(cost_usd={self._cost_usd:.4f}, "
            f"max_usd={self._max_usd}, "
            f"tokens={self._input_tokens + self._output_tokens})"
        )
