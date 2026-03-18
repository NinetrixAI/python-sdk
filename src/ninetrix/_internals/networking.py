"""
ninetrix._internals.networking
==============================
L1 kernel — stdlib only, no ninetrix imports.

Contains:
- TimeoutConfig        — per-operation timeout settings
- BackoffStrategy      — enum for retry delay calculation
- RetryPolicy          — max_retries + backoff + retryable status codes
- with_retry()         — async helper that applies RetryPolicy to a coroutine
- CircuitBreaker       — CLOSED/OPEN/HALF_OPEN state machine per service
- CircuitBreakerRegistry — process-wide singleton registry keyed by service name
- ProviderRateLimiter  — process-wide token bucket per provider
"""

from __future__ import annotations

import asyncio
import enum
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# TimeoutConfig
# ---------------------------------------------------------------------------

@dataclass
class TimeoutConfig:
    """
    Per-operation timeout settings (seconds).

    connect:  TCP connection + TLS handshake.
    read:     Time to receive the first byte after the request is sent.
    total:    Hard wall-clock cap for the entire operation.

    Set to None to disable a specific timeout.
    """
    connect: float | None = 10.0
    read: float | None = 60.0
    total: float | None = 120.0

    # Per-operation overrides (optional)
    llm_complete: float | None = None       # overrides total for LLM calls
    tool_call: float | None = None          # overrides total for tool calls
    checkpoint: float | None = None         # overrides total for DB writes

    def for_llm(self) -> float | None:
        return self.llm_complete if self.llm_complete is not None else self.total

    def for_tool(self) -> float | None:
        return self.tool_call if self.tool_call is not None else self.total

    def for_checkpoint(self) -> float | None:
        return self.checkpoint if self.checkpoint is not None else self.total


# ---------------------------------------------------------------------------
# BackoffStrategy + RetryPolicy
# ---------------------------------------------------------------------------

class BackoffStrategy(enum.Enum):
    """Delay calculation strategy between retry attempts."""
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"   # recommended for production


@dataclass
class RetryPolicy:
    """
    Controls how with_retry() retries a failing coroutine.

    max_retries:           Maximum number of retry attempts (not counting the first try).
    base_delay:            Base delay in seconds for backoff calculation.
    max_delay:             Cap on computed delay (seconds).
    backoff:               Which BackoffStrategy to use.
    retryable_status_codes: HTTP status codes that should trigger a retry.
                            Empty = retry on any ProviderError with retryable=True.
    jitter_range:          ± fraction of computed delay to add as random jitter
                           (only used by EXPONENTIAL_JITTER; range 0.0–1.0).
    """
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER
    retryable_status_codes: frozenset[int] = field(
        default_factory=lambda: frozenset({429, 500, 502, 503, 504})
    )
    jitter_range: float = 0.25

    def delay_for(self, attempt: int) -> float:
        """
        Compute the delay (seconds) before attempt N (0-indexed).
        attempt=0 → first retry, attempt=1 → second retry, etc.
        """
        match self.backoff:
            case BackoffStrategy.CONSTANT:
                delay = self.base_delay
            case BackoffStrategy.LINEAR:
                delay = self.base_delay * (attempt + 1)
            case BackoffStrategy.EXPONENTIAL:
                delay = self.base_delay * (2 ** attempt)
            case BackoffStrategy.EXPONENTIAL_JITTER:
                delay = self.base_delay * (2 ** attempt)
                jitter = delay * self.jitter_range * (2 * random.random() - 1)
                delay = delay + jitter

        return max(0.0, min(delay, self.max_delay))

    def should_retry(self, exc: BaseException) -> bool:
        """
        Return True if exc warrants a retry according to this policy.
        Imports ProviderError lazily to stay L1-compatible (no circular dep).
        """
        # Inline import — safe: ProviderError is defined in the same L1 module
        from ninetrix._internals.types import ProviderError, NetworkError

        if isinstance(exc, ProviderError):
            if exc.retryable:
                return True
            if exc.status_code is not None and exc.status_code in self.retryable_status_codes:
                return True
        if isinstance(exc, NetworkError):
            return exc.retryable
        return False


# Default policies — reuse instead of constructing new ones per call
RETRY_DEFAULT = RetryPolicy()
RETRY_AGGRESSIVE = RetryPolicy(max_retries=5, base_delay=0.5, max_delay=30.0)
RETRY_NONE = RetryPolicy(max_retries=0)


# ---------------------------------------------------------------------------
# with_retry()
# ---------------------------------------------------------------------------

async def with_retry(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy = RETRY_DEFAULT,
    *,
    operation: str = "",
) -> T:
    """
    Execute an async callable with retry logic defined by policy.

    Args:
        fn:        Zero-argument async callable to retry (use functools.partial or lambda).
        policy:    RetryPolicy controlling delays and max attempts.
        operation: Human-readable label for log messages (e.g. "anthropic.complete").

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception from fn() after all retries are exhausted.

    Example:
        result = await with_retry(
            lambda: adapter.complete(messages, tools),
            policy=RETRY_DEFAULT,
            operation="anthropic.complete",
        )
    """
    last_exc: BaseException | None = None

    for attempt in range(policy.max_retries + 1):
        try:
            return await fn()
        except BaseException as exc:
            last_exc = exc

            is_last = attempt >= policy.max_retries
            if is_last or not policy.should_retry(exc):
                raise

            delay = policy.delay_for(attempt)
            label = f" [{operation}]" if operation else ""
            logger.warning(
                "ninetrix%s: attempt %d/%d failed (%s), retrying in %.1fs",
                label,
                attempt + 1,
                policy.max_retries + 1,
                type(exc).__name__,
                delay,
            )
            await asyncio.sleep(delay)

    # Should never reach here — loop always raises or returns
    assert last_exc is not None
    raise last_exc  # pragma: no cover


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class CircuitState(enum.Enum):
    CLOSED = "closed"           # normal operation — calls pass through
    OPEN = "open"               # tripped — calls fail fast without executing
    HALF_OPEN = "half_open"     # cooldown elapsed — one probe call allowed


class CircuitBreakerOpen(Exception):
    """
    Raised by CircuitBreaker.call() when the circuit is OPEN.

    What:  The circuit breaker for '{name}' is open.
    Why:   {failure_count} consecutive failures exceeded the threshold ({threshold}).
    Fix:   The circuit will automatically reset after {reset_timeout}s.
           Check the health of the upstream service.
    """

    def __init__(self, name: str, failure_count: int, threshold: int, reset_timeout: float) -> None:
        super().__init__(
            f"Circuit breaker '{name}' is open — upstream service appears down.\n"
            f"  Why: {failure_count} consecutive failures exceeded threshold ({threshold}).\n"
            f"  Fix: Circuit resets automatically after {reset_timeout:.0f}s. "
            f"Check the health of the upstream service."
        )
        self.name = name
        self.failure_count = failure_count
        self.threshold = threshold
        self.reset_timeout = reset_timeout


class CircuitBreaker:
    """
    CLOSED → OPEN → HALF_OPEN → CLOSED state machine.

    CLOSED:    Calls pass through. Failure counter increments on exception.
               When failures >= failure_threshold → transition to OPEN.

    OPEN:      Calls fail immediately with CircuitBreakerOpen (no upstream call).
               After reset_timeout seconds → transition to HALF_OPEN.

    HALF_OPEN: One probe call is allowed.
               Success → CLOSED (counter reset).
               Failure → back to OPEN (timer restarted).
    """

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        half_open_max_calls: int = 1,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: float | None = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def _maybe_transition_to_half_open(self) -> None:
        """Called while holding _lock. Transitions OPEN → HALF_OPEN after timeout."""
        if (
            self._state == CircuitState.OPEN
            and self._opened_at is not None
            and time.monotonic() - self._opened_at >= self.reset_timeout
        ):
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0
            logger.info("CircuitBreaker '%s': OPEN → HALF_OPEN (probe allowed)", self.name)

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        """
        Execute fn() through the circuit breaker.

        Raises CircuitBreakerOpen if the circuit is OPEN.
        Propagates fn()'s exception and updates failure count on error.
        """
        async with self._lock:
            self._maybe_transition_to_half_open()

            if self._state == CircuitState.OPEN:
                raise CircuitBreakerOpen(
                    self.name,
                    self._failure_count,
                    self.failure_threshold,
                    self.reset_timeout,
                )

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpen(
                        self.name,
                        self._failure_count,
                        self.failure_threshold,
                        self.reset_timeout,
                    )
                self._half_open_calls += 1

        try:
            result = await fn()
        except BaseException as exc:
            async with self._lock:
                self._on_failure()
            raise
        else:
            async with self._lock:
                self._on_success()
            return result

    def _on_success(self) -> None:
        """Called while holding _lock."""
        if self._state in (CircuitState.HALF_OPEN, CircuitState.CLOSED):
            if self._failure_count > 0:
                logger.info(
                    "CircuitBreaker '%s': recovered → CLOSED (was %d failures)",
                    self.name, self._failure_count,
                )
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._opened_at = None

    def _on_failure(self) -> None:
        """Called while holding _lock."""
        self._failure_count += 1

        if self._state == CircuitState.HALF_OPEN:
            # Probe failed — back to OPEN
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            logger.warning(
                "CircuitBreaker '%s': HALF_OPEN probe failed → OPEN (reset in %.0fs)",
                self.name, self.reset_timeout,
            )
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            logger.warning(
                "CircuitBreaker '%s': CLOSED → OPEN (%d/%d failures, reset in %.0fs)",
                self.name, self._failure_count, self.failure_threshold, self.reset_timeout,
            )

    def reset(self) -> None:
        """Force-reset to CLOSED. Useful in tests."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at = None
        self._half_open_calls = 0


# ---------------------------------------------------------------------------
# CircuitBreakerRegistry — process-wide singleton
# ---------------------------------------------------------------------------

class CircuitBreakerRegistry:
    """
    Process-wide registry of CircuitBreaker instances, keyed by service name.
    All Agent instances share the same breakers — one OPEN state affects all.

    Usage:
        breaker = CircuitBreakerRegistry.get("mcp-gateway")
        result = await breaker.call(lambda: http.post(...))
    """
    _registry: dict[str, CircuitBreaker] = {}

    @classmethod
    def get(
        cls,
        name: str,
        *,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
    ) -> CircuitBreaker:
        """Return (or create) the CircuitBreaker for name."""
        if name not in cls._registry:
            cls._registry[name] = CircuitBreaker(
                name,
                failure_threshold=failure_threshold,
                reset_timeout=reset_timeout,
            )
        return cls._registry[name]

    @classmethod
    def reset_all(cls) -> None:
        """Reset all circuit breakers. Useful in tests."""
        for cb in cls._registry.values():
            cb.reset()
        cls._registry.clear()


# ---------------------------------------------------------------------------
# ProviderRateLimiter — process-wide token bucket per provider
# ---------------------------------------------------------------------------

# Default requests-per-second per provider (conservative tier-1 defaults)
_DEFAULT_RPS: dict[str, float] = {
    "anthropic": 50.0,
    "openai": 60.0,
    "google": 60.0,
    "groq": 30.0,
    "mistral": 20.0,
    "litellm": 50.0,
}


class ProviderRateLimiter:
    """
    Process-wide token bucket rate limiter, keyed by provider name.
    All Agent instances hitting the same provider share one bucket.

    Uses an async token bucket: tokens refill at `rps` rate.
    Callers await acquire() before each LLM request.

    Usage:
        limiter = ProviderRateLimiter.for_provider("anthropic")
        await limiter.acquire()
        response = await adapter.complete(...)
    """
    _instances: dict[str, "ProviderRateLimiter"] = {}

    def __init__(self, provider: str, rps: float) -> None:
        self._provider = provider
        self._rps = rps
        self._tokens = rps           # start full
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    @classmethod
    def for_provider(cls, provider: str, *, rps: float | None = None) -> "ProviderRateLimiter":
        """
        Return (or create) the process-wide limiter for provider.
        Pass rps= to override the default on first creation.
        """
        if provider not in cls._instances:
            effective_rps = rps if rps is not None else _DEFAULT_RPS.get(provider, 10.0)
            cls._instances[provider] = cls(provider, effective_rps)
        return cls._instances[provider]

    async def acquire(self, tokens: float = 1.0) -> None:
        """
        Acquire tokens from the bucket, sleeping if necessary.
        Blocks until the requested tokens are available.
        """
        async with self._lock:
            self._refill()
            wait = 0.0
            if self._tokens < tokens:
                wait = (tokens - self._tokens) / self._rps
                self._tokens = 0.0
            else:
                self._tokens -= tokens

        if wait > 0:
            await asyncio.sleep(wait)

    def _refill(self) -> None:
        """Add tokens based on elapsed time. Called while holding _lock."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._rps, self._tokens + elapsed * self._rps)
        self._last_refill = now

    @classmethod
    def reset_all(cls) -> None:
        """Clear all limiter instances. Useful in tests."""
        cls._instances.clear()
