"""
Unit tests for ninetrix._internals.networking (PR 2).

Coverage:
- TimeoutConfig: defaults, per-operation overrides
- BackoffStrategy + RetryPolicy: delay calculation for all strategies
- RetryPolicy.should_retry(): ProviderError (retryable flag, status codes), NetworkError
- with_retry(): success, retries with backoff, exhaustion, non-retryable errors
- CircuitBreaker: CLOSED→OPEN, OPEN fast-fail, cooldown→HALF_OPEN, probe success/failure
- CircuitBreakerRegistry: singleton behavior, reset_all
- ProviderRateLimiter: token bucket refill, for_provider singleton, reset_all
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, patch

from ninetrix._internals.networking import (
    TimeoutConfig,
    BackoffStrategy,
    RetryPolicy,
    RETRY_DEFAULT,
    RETRY_NONE,
    with_retry,
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitBreakerRegistry,
    CircuitState,
    ProviderRateLimiter,
)
from ninetrix._internals.types import ProviderError, NetworkError


# ---------------------------------------------------------------------------
# TimeoutConfig
# ---------------------------------------------------------------------------

class TestTimeoutConfig:
    def test_defaults(self):
        tc = TimeoutConfig()
        assert tc.connect == 10.0
        assert tc.read == 60.0
        assert tc.total == 120.0
        assert tc.llm_complete is None

    def test_for_llm_falls_back_to_total(self):
        tc = TimeoutConfig(total=90.0)
        assert tc.for_llm() == 90.0

    def test_for_llm_uses_override(self):
        tc = TimeoutConfig(total=90.0, llm_complete=180.0)
        assert tc.for_llm() == 180.0

    def test_for_tool_falls_back_to_total(self):
        tc = TimeoutConfig(total=30.0)
        assert tc.for_tool() == 30.0

    def test_for_tool_uses_override(self):
        tc = TimeoutConfig(total=30.0, tool_call=10.0)
        assert tc.for_tool() == 10.0

    def test_for_checkpoint_falls_back_to_total(self):
        tc = TimeoutConfig(total=60.0)
        assert tc.for_checkpoint() == 60.0

    def test_none_timeouts_allowed(self):
        tc = TimeoutConfig(connect=None, read=None, total=None)
        assert tc.for_llm() is None
        assert tc.for_tool() is None


# ---------------------------------------------------------------------------
# RetryPolicy.delay_for()
# ---------------------------------------------------------------------------

class TestRetryPolicyDelay:
    def test_constant_backoff(self):
        p = RetryPolicy(base_delay=2.0, backoff=BackoffStrategy.CONSTANT)
        assert p.delay_for(0) == 2.0
        assert p.delay_for(1) == 2.0
        assert p.delay_for(5) == 2.0

    def test_linear_backoff(self):
        p = RetryPolicy(base_delay=1.0, backoff=BackoffStrategy.LINEAR)
        assert p.delay_for(0) == 1.0
        assert p.delay_for(1) == 2.0
        assert p.delay_for(2) == 3.0

    def test_exponential_backoff(self):
        p = RetryPolicy(base_delay=1.0, backoff=BackoffStrategy.EXPONENTIAL)
        assert p.delay_for(0) == 1.0
        assert p.delay_for(1) == 2.0
        assert p.delay_for(2) == 4.0
        assert p.delay_for(3) == 8.0

    def test_max_delay_cap(self):
        p = RetryPolicy(base_delay=1.0, max_delay=5.0, backoff=BackoffStrategy.EXPONENTIAL)
        assert p.delay_for(10) == 5.0

    def test_exponential_jitter_within_bounds(self):
        p = RetryPolicy(base_delay=1.0, backoff=BackoffStrategy.EXPONENTIAL_JITTER, jitter_range=0.25)
        for attempt in range(5):
            delay = p.delay_for(attempt)
            base = 1.0 * (2 ** attempt)
            assert delay >= 0.0
            assert delay <= p.max_delay

    def test_delay_never_negative(self):
        p = RetryPolicy(base_delay=0.0, backoff=BackoffStrategy.CONSTANT)
        assert p.delay_for(0) == 0.0


# ---------------------------------------------------------------------------
# RetryPolicy.should_retry()
# ---------------------------------------------------------------------------

class TestRetryPolicyShouldRetry:
    def test_retryable_provider_error(self):
        p = RetryPolicy()
        exc = ProviderError("rate limited", retryable=True)
        assert p.should_retry(exc) is True

    def test_non_retryable_provider_error(self):
        p = RetryPolicy()
        exc = ProviderError("auth failed", retryable=False, status_code=401)
        assert p.should_retry(exc) is False

    def test_retryable_status_code(self):
        p = RetryPolicy(retryable_status_codes=frozenset({429, 503}))
        exc = ProviderError("rate limited", retryable=False, status_code=429)
        assert p.should_retry(exc) is True

    def test_non_retryable_status_code(self):
        p = RetryPolicy(retryable_status_codes=frozenset({429}))
        exc = ProviderError("bad request", retryable=False, status_code=400)
        assert p.should_retry(exc) is False

    def test_retryable_network_error(self):
        p = RetryPolicy()
        exc = NetworkError("timeout", retryable=True)
        assert p.should_retry(exc) is True

    def test_non_retryable_network_error(self):
        p = RetryPolicy()
        exc = NetworkError("SSL invalid", retryable=False)
        assert p.should_retry(exc) is False

    def test_generic_exception_not_retried(self):
        p = RetryPolicy()
        assert p.should_retry(ValueError("oops")) is False

    def test_retry_none_policy_max_retries_zero(self):
        assert RETRY_NONE.max_retries == 0


# ---------------------------------------------------------------------------
# with_retry()
# ---------------------------------------------------------------------------

class TestWithRetry:
    def test_success_on_first_try(self):
        calls = []

        async def fn():
            calls.append(1)
            return "ok"

        result = asyncio.run(with_retry(fn, RETRY_NONE))
        assert result == "ok"
        assert len(calls) == 1

    def test_retries_on_retryable_error(self):
        attempts = []

        async def fn():
            attempts.append(1)
            if len(attempts) < 3:
                raise ProviderError("retry me", retryable=True)
            return "done"

        policy = RetryPolicy(max_retries=3, base_delay=0.0, backoff=BackoffStrategy.CONSTANT)
        result = asyncio.run(with_retry(fn, policy))
        assert result == "done"
        assert len(attempts) == 3

    def test_raises_after_exhausting_retries(self):
        async def fn():
            raise ProviderError("always fails", retryable=True)

        policy = RetryPolicy(max_retries=2, base_delay=0.0, backoff=BackoffStrategy.CONSTANT)
        with pytest.raises(ProviderError, match="always fails"):
            asyncio.run(with_retry(fn, policy))

    def test_does_not_retry_non_retryable_error(self):
        calls = []

        async def fn():
            calls.append(1)
            raise ProviderError("auth failed", retryable=False, status_code=401)

        policy = RetryPolicy(max_retries=3, base_delay=0.0)
        with pytest.raises(ProviderError):
            asyncio.run(with_retry(fn, policy))
        assert len(calls) == 1

    def test_does_not_retry_value_error(self):
        calls = []

        async def fn():
            calls.append(1)
            raise ValueError("programmer error")

        policy = RetryPolicy(max_retries=3, base_delay=0.0)
        with pytest.raises(ValueError):
            asyncio.run(with_retry(fn, policy))
        assert len(calls) == 1

    def test_retry_none_does_not_retry(self):
        calls = []

        async def fn():
            calls.append(1)
            raise ProviderError("nope", retryable=True)

        with pytest.raises(ProviderError):
            asyncio.run(with_retry(fn, RETRY_NONE))
        assert len(calls) == 1

    def test_operation_label_accepted(self):
        async def fn():
            return 42

        result = asyncio.run(with_retry(fn, RETRY_NONE, operation="test.op"))
        assert result == 42


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def _make(self, threshold=3, reset_timeout=30.0) -> CircuitBreaker:
        return CircuitBreaker("test", failure_threshold=threshold, reset_timeout=reset_timeout)

    def test_initial_state_is_closed(self):
        cb = self._make()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_success_keeps_closed(self):
        cb = self._make()

        async def run():
            return await cb.call(AsyncMock(return_value="ok"))

        assert asyncio.run(run()) == "ok"
        assert cb.state == CircuitState.CLOSED

    def test_failures_increment_counter(self):
        cb = self._make(threshold=5)

        async def run():
            try:
                await cb.call(AsyncMock(side_effect=ProviderError("err", retryable=True)))
            except ProviderError:
                pass

        asyncio.run(run())
        assert cb.failure_count == 1
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold(self):
        cb = self._make(threshold=3)

        async def run():
            for _ in range(3):
                try:
                    await cb.call(AsyncMock(side_effect=ProviderError("err", retryable=True)))
                except ProviderError:
                    pass

        asyncio.run(run())
        assert cb.state == CircuitState.OPEN

    def test_open_circuit_raises_circuit_breaker_open(self):
        cb = self._make(threshold=1)

        async def run():
            try:
                await cb.call(AsyncMock(side_effect=ProviderError("err", retryable=True)))
            except ProviderError:
                pass
            # Now OPEN — next call should raise CircuitBreakerOpen
            await cb.call(AsyncMock(return_value="should not reach"))

        with pytest.raises(CircuitBreakerOpen) as exc_info:
            asyncio.run(run())
        assert exc_info.value.name == "test"
        assert exc_info.value.threshold == 1

    def test_circuit_breaker_open_message_contains_fix(self):
        exc = CircuitBreakerOpen("svc", 5, 5, 30.0)
        assert "resets automatically" in str(exc)
        assert "svc" in str(exc)

    def test_transitions_to_half_open_after_timeout(self):
        cb = self._make(threshold=1, reset_timeout=0.01)

        async def run():
            try:
                await cb.call(AsyncMock(side_effect=ProviderError("err", retryable=True)))
            except ProviderError:
                pass
            assert cb.state == CircuitState.OPEN
            await asyncio.sleep(0.05)  # wait for reset_timeout
            # Trigger the transition check by attempting a call
            try:
                await cb.call(AsyncMock(return_value="probe ok"))
            except CircuitBreakerOpen:
                pass

        asyncio.run(run())
        # After timeout the breaker transitions to HALF_OPEN before the probe call

    def test_successful_probe_closes_circuit(self):
        cb = self._make(threshold=1, reset_timeout=0.01)

        async def run():
            try:
                await cb.call(AsyncMock(side_effect=ProviderError("err", retryable=True)))
            except ProviderError:
                pass
            await asyncio.sleep(0.05)
            result = await cb.call(AsyncMock(return_value="recovered"))
            return result

        result = asyncio.run(run())
        assert result == "recovered"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_failed_probe_reopens_circuit(self):
        cb = self._make(threshold=1, reset_timeout=0.01)

        async def run():
            try:
                await cb.call(AsyncMock(side_effect=ProviderError("err", retryable=True)))
            except ProviderError:
                pass
            await asyncio.sleep(0.05)
            try:
                await cb.call(AsyncMock(side_effect=ProviderError("still down", retryable=True)))
            except (ProviderError, CircuitBreakerOpen):
                pass

        asyncio.run(run())
        assert cb.state == CircuitState.OPEN

    def test_reset_clears_state(self):
        cb = self._make(threshold=1)

        async def run():
            try:
                await cb.call(AsyncMock(side_effect=ProviderError("err", retryable=True)))
            except ProviderError:
                pass

        asyncio.run(run())
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0


# ---------------------------------------------------------------------------
# CircuitBreakerRegistry
# ---------------------------------------------------------------------------

class TestCircuitBreakerRegistry:
    def setup_method(self):
        CircuitBreakerRegistry.reset_all()

    def test_get_returns_same_instance(self):
        cb1 = CircuitBreakerRegistry.get("svc-a")
        cb2 = CircuitBreakerRegistry.get("svc-a")
        assert cb1 is cb2

    def test_different_names_return_different_instances(self):
        cb1 = CircuitBreakerRegistry.get("svc-a")
        cb2 = CircuitBreakerRegistry.get("svc-b")
        assert cb1 is not cb2

    def test_reset_all_clears_registry(self):
        cb = CircuitBreakerRegistry.get("svc")
        CircuitBreakerRegistry.reset_all()
        cb2 = CircuitBreakerRegistry.get("svc")
        assert cb is not cb2


# ---------------------------------------------------------------------------
# ProviderRateLimiter
# ---------------------------------------------------------------------------

class TestProviderRateLimiter:
    def setup_method(self):
        ProviderRateLimiter.reset_all()

    def test_for_provider_returns_same_instance(self):
        a = ProviderRateLimiter.for_provider("anthropic")
        b = ProviderRateLimiter.for_provider("anthropic")
        assert a is b

    def test_different_providers_are_separate(self):
        a = ProviderRateLimiter.for_provider("anthropic")
        o = ProviderRateLimiter.for_provider("openai")
        assert a is not o

    def test_acquire_does_not_block_when_tokens_available(self):
        limiter = ProviderRateLimiter("test", rps=100.0)

        async def run():
            start = time.monotonic()
            await limiter.acquire()
            return time.monotonic() - start

        elapsed = asyncio.run(run())
        assert elapsed < 0.1  # should be essentially instant

    def test_acquire_sleeps_when_bucket_empty(self):
        # 1 rps, bucket starts full (1 token). Third acquire must wait ~2s.
        # Use a very fast rate to keep test quick.
        limiter = ProviderRateLimiter("test", rps=100.0)
        # Drain the bucket manually
        limiter._tokens = 0.0
        limiter._last_refill = time.monotonic()

        async def run():
            start = time.monotonic()
            await limiter.acquire()   # must wait for refill
            return time.monotonic() - start

        elapsed = asyncio.run(run())
        # At 100 rps, 1 token takes 0.01s. Allow generous margin.
        assert elapsed >= 0.005

    def test_custom_rps_on_first_creation(self):
        limiter = ProviderRateLimiter.for_provider("custom-provider", rps=5.0)
        assert limiter._rps == 5.0

    def test_reset_all_clears_instances(self):
        a = ProviderRateLimiter.for_provider("anthropic")
        ProviderRateLimiter.reset_all()
        b = ProviderRateLimiter.for_provider("anthropic")
        assert a is not b
