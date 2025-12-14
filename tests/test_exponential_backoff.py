"""Unit tests for ExponentialBackoff.

Tests delay calculation with various attempts, header honoring
(Retry-After, X-RateLimit-Reset), and budget enforcement.

Requirements: 3.11, 3.12, 3.15
"""

import time
from datetime import datetime, timezone, timedelta
from email.utils import format_datetime

from core.security.circuit_breaker import ExponentialBackoff


class TestExponentialBackoffDelayCalculation:
    """Tests for delay calculation with various attempts.

    Requirements: 3.11
    """

    def test_first_attempt_uses_base_delay(self):
        """First attempt (attempt=0) uses base delay."""
        backoff = ExponentialBackoff(
            base_delay=1.0,
            multiplier=2.0,
            jitter_factor=0.0,  # Disable jitter for predictable testing
        )

        delay = backoff.get_delay(0)
        assert delay == 1.0

    def test_delay_increases_exponentially(self):
        """Delay increases exponentially with each attempt."""
        backoff = ExponentialBackoff(
            base_delay=1.0,
            multiplier=2.0,
            jitter_factor=0.0,
        )

        assert backoff.get_delay(0) == 1.0   # 1 * 2^0 = 1
        assert backoff.get_delay(1) == 2.0   # 1 * 2^1 = 2
        assert backoff.get_delay(2) == 4.0   # 1 * 2^2 = 4
        assert backoff.get_delay(3) == 8.0   # 1 * 2^3 = 8
        assert backoff.get_delay(4) == 16.0  # 1 * 2^4 = 16

    def test_delay_capped_at_max_delay(self):
        """Delay is capped at max_delay."""
        backoff = ExponentialBackoff(
            base_delay=1.0,
            multiplier=2.0,
            max_delay=10.0,
            jitter_factor=0.0,
        )

        # Attempt 4 would be 16, but capped at 10
        assert backoff.get_delay(4) == 10.0
        # Attempt 5 would be 32, but capped at 10
        assert backoff.get_delay(5) == 10.0

    def test_custom_base_delay(self):
        """Custom base delay is respected."""
        backoff = ExponentialBackoff(
            base_delay=2.0,
            multiplier=2.0,
            jitter_factor=0.0,
        )

        assert backoff.get_delay(0) == 2.0   # 2 * 2^0 = 2
        assert backoff.get_delay(1) == 4.0   # 2 * 2^1 = 4
        assert backoff.get_delay(2) == 8.0   # 2 * 2^2 = 8

    def test_custom_multiplier(self):
        """Custom multiplier is respected."""
        backoff = ExponentialBackoff(
            base_delay=1.0,
            multiplier=3.0,
            jitter_factor=0.0,
        )

        assert backoff.get_delay(0) == 1.0   # 1 * 3^0 = 1
        assert backoff.get_delay(1) == 3.0   # 1 * 3^1 = 3
        assert backoff.get_delay(2) == 9.0   # 1 * 3^2 = 9

    def test_jitter_adds_variance(self):
        """Jitter adds variance to delay."""
        backoff = ExponentialBackoff(
            base_delay=10.0,
            multiplier=2.0,
            max_delay=100.0,
            jitter_factor=0.5,  # 50% jitter
        )

        # Collect multiple samples
        delays = [backoff.get_delay(0) for _ in range(100)]

        # Base delay is 10, with 50% jitter, range should be [10, 15]
        assert all(10.0 <= d <= 15.0 for d in delays)

        # Should have some variance (not all the same)
        unique_delays = set(delays)
        assert len(unique_delays) > 1

    def test_default_values(self):
        """Default values are applied correctly."""
        backoff = ExponentialBackoff()

        # Default base_delay=1.0, multiplier=2.0, max_delay=60.0
        # With default jitter_factor=0.1, delay should be in range [1.0, 1.1]
        delay = backoff.get_delay(0)
        assert 1.0 <= delay <= 1.1

    def test_zero_jitter_factor(self):
        """Zero jitter factor produces deterministic delays."""
        backoff = ExponentialBackoff(
            base_delay=5.0,
            multiplier=2.0,
            jitter_factor=0.0,
        )

        # Multiple calls should return same value
        delays = [backoff.get_delay(1) for _ in range(10)]
        assert all(d == 10.0 for d in delays)


class TestExponentialBackoffHeaderHonoring:
    """Tests for header honoring (Retry-After, X-RateLimit-Reset).

    Requirements: 3.12, 3.13, 3.14, 3.15, 3.16, 3.17
    """

    def test_retry_after_numeric_seconds(self):
        """Retry-After header with numeric seconds is honored."""
        backoff = ExponentialBackoff(
            base_delay=1.0,
            jitter_factor=0.0,
        )

        delay = backoff.get_delay_honoring_headers(
            attempt=0,
            retry_after="30",
        )

        assert delay == 30.0

    def test_retry_after_http_date(self):
        """Retry-After header with HTTP-date is honored."""
        backoff = ExponentialBackoff(
            base_delay=1.0,
            jitter_factor=0.0,
        )

        # Create a date 45 seconds in the future
        future_time = datetime.now(timezone.utc) + timedelta(seconds=45)
        http_date = format_datetime(future_time, usegmt=True)

        delay = backoff.get_delay_honoring_headers(
            attempt=0,
            retry_after=http_date,
        )

        # Should be approximately 45 seconds (allow some tolerance)
        assert 43.0 <= delay <= 47.0

    def test_rate_limit_reset_unix_timestamp(self):
        """X-RateLimit-Reset header with Unix timestamp is honored."""
        backoff = ExponentialBackoff(
            base_delay=1.0,
            jitter_factor=0.0,
        )

        # Create a timestamp 60 seconds in the future
        future_timestamp = time.time() + 60

        delay = backoff.get_delay_honoring_headers(
            attempt=0,
            rate_limit_reset=str(future_timestamp),
        )

        # Should be approximately 60 seconds (allow some tolerance)
        assert 58.0 <= delay <= 62.0

    def test_uses_larger_delay_when_both_headers_present(self):
        """When both headers present, uses larger delay (stricter limit)."""
        backoff = ExponentialBackoff(
            base_delay=1.0,
            jitter_factor=0.0,
        )

        # Retry-After: 30 seconds
        # X-RateLimit-Reset: 60 seconds in future
        future_timestamp = time.time() + 60

        delay = backoff.get_delay_honoring_headers(
            attempt=0,
            retry_after="30",
            rate_limit_reset=str(future_timestamp),
        )

        # Should use the larger value (60 seconds)
        assert 58.0 <= delay <= 62.0
    def test_uses_larger_of_both_headers(self):
        """When both headers present with different values, uses larger."""
        backoff = ExponentialBackoff(
            base_delay=1.0,
            jitter_factor=0.0,
        )

        # Retry-After: 90 seconds (larger)
        # X-RateLimit-Reset: 30 seconds in future
        future_timestamp = time.time() + 30

        delay = backoff.get_delay_honoring_headers(
            attempt=0,
            retry_after="90",
            rate_limit_reset=str(future_timestamp),
        )

        # Should use the larger value (90 seconds)
        assert delay == 90.0

    def test_fallback_to_exponential_when_no_headers(self):
        """Falls back to exponential backoff when no headers provided."""
        backoff = ExponentialBackoff(
            base_delay=2.0,
            multiplier=2.0,
            jitter_factor=0.0,
        )

        delay = backoff.get_delay_honoring_headers(
            attempt=2,
            retry_after=None,
            rate_limit_reset=None,
        )

        # Should be 2 * 2^2 = 8
        assert delay == 8.0

    def test_fallback_when_retry_after_malformed(self):
        """Falls back to exponential when Retry-After is malformed."""
        backoff = ExponentialBackoff(
            base_delay=1.0,
            multiplier=2.0,
            jitter_factor=0.0,
        )

        delay = backoff.get_delay_honoring_headers(
            attempt=1,
            retry_after="not-a-number-or-date",
        )

        # Should fall back to exponential: 1 * 2^1 = 2
        assert delay == 2.0

    def test_fallback_when_rate_limit_reset_malformed(self):
        """Falls back to exponential when X-RateLimit-Reset is malformed."""
        backoff = ExponentialBackoff(
            base_delay=1.0,
            multiplier=2.0,
            jitter_factor=0.0,
        )

        delay = backoff.get_delay_honoring_headers(
            attempt=1,
            rate_limit_reset="invalid-timestamp",
        )

        # Should fall back to exponential: 1 * 2^1 = 2
        assert delay == 2.0

    def test_negative_delay_from_header_is_invalid(self):
        """Negative delay from header is treated as invalid."""
        backoff = ExponentialBackoff(
            base_delay=1.0,
            multiplier=2.0,
            jitter_factor=0.0,
        )

        # Timestamp in the past produces negative delay
        past_timestamp = time.time() - 60

        delay = backoff.get_delay_honoring_headers(
            attempt=1,
            rate_limit_reset=str(past_timestamp),
        )

        # Should fall back to exponential: 1 * 2^1 = 2
        assert delay == 2.0

    def test_delay_exceeding_max_header_delay_is_invalid(self):
        """Delay exceeding 3600 seconds (1 hour) is treated as invalid."""
        backoff = ExponentialBackoff(
            base_delay=1.0,
            multiplier=2.0,
            jitter_factor=0.0,
        )

        # 7200 seconds (2 hours) exceeds max
        delay = backoff.get_delay_honoring_headers(
            attempt=1,
            retry_after="7200",
        )

        # Should fall back to exponential: 1 * 2^1 = 2
        assert delay == 2.0

    def test_delay_at_max_header_delay_is_valid(self):
        """Delay at exactly 3600 seconds (1 hour) is valid."""
        backoff = ExponentialBackoff(
            base_delay=1.0,
            jitter_factor=0.0,
        )

        delay = backoff.get_delay_honoring_headers(
            attempt=0,
            retry_after="3600",
        )

        assert delay == 3600.0

    def test_empty_retry_after_falls_back(self):
        """Empty Retry-After header falls back to exponential."""
        backoff = ExponentialBackoff(
            base_delay=1.0,
            multiplier=2.0,
            jitter_factor=0.0,
        )

        delay = backoff.get_delay_honoring_headers(
            attempt=1,
            retry_after="",
        )

        # Should fall back to exponential: 1 * 2^1 = 2
        assert delay == 2.0

    def test_whitespace_in_header_is_trimmed(self):
        """Whitespace in header values is trimmed."""
        backoff = ExponentialBackoff(
            base_delay=1.0,
            jitter_factor=0.0,
        )

        delay = backoff.get_delay_honoring_headers(
            attempt=0,
            retry_after="  30  ",
        )

        assert delay == 30.0

    def test_float_retry_after_is_accepted(self):
        """Float value in Retry-After is accepted."""
        backoff = ExponentialBackoff(
            base_delay=1.0,
            jitter_factor=0.0,
        )

        delay = backoff.get_delay_honoring_headers(
            attempt=0,
            retry_after="30.5",
        )

        assert delay == 30.5


class TestExponentialBackoffBudgetEnforcement:
    """Tests for budget enforcement.

    Requirements: 3.20, 3.21
    """

    def test_within_budget_returns_true_when_under_limit(self):
        """is_within_budget returns True when under budget."""
        backoff = ExponentialBackoff()

        # 100 seconds remaining, 50% budget = 50 seconds
        # Next delay of 10 seconds is within budget
        result = backoff.is_within_budget(
            next_delay=10.0,
            remaining_seconds=100.0,
            budget_fraction=0.5,
        )

        assert result is True

    def test_within_budget_returns_false_when_over_limit(self):
        """is_within_budget returns False when over budget."""
        backoff = ExponentialBackoff()

        # 100 seconds remaining, 50% budget = 50 seconds
        # Next delay of 60 seconds exceeds budget
        result = backoff.is_within_budget(
            next_delay=60.0,
            remaining_seconds=100.0,
            budget_fraction=0.5,
        )

        assert result is False

    def test_cumulative_delay_tracking(self):
        """Cumulative delay is tracked correctly."""
        backoff = ExponentialBackoff()

        assert backoff.cumulative_delay == 0.0

        backoff.record_delay(5.0)
        assert backoff.cumulative_delay == 5.0

        backoff.record_delay(10.0)
        assert backoff.cumulative_delay == 15.0

    def test_cumulative_delay_affects_budget_check(self):
        """Cumulative delay is considered in budget check."""
        backoff = ExponentialBackoff()

        # Record 40 seconds of cumulative delay
        backoff.record_delay(40.0)

        # 100 seconds remaining, 50% budget = 50 seconds
        # Cumulative (40) + next (5) = 45, within budget
        result = backoff.is_within_budget(
            next_delay=5.0,
            remaining_seconds=100.0,
            budget_fraction=0.5,
        )
        assert result is True

        # Cumulative (40) + next (15) = 55, exceeds budget
        result = backoff.is_within_budget(
            next_delay=15.0,
            remaining_seconds=100.0,
            budget_fraction=0.5,
        )
        assert result is False

    def test_reset_cumulative_delay(self):
        """Cumulative delay can be reset."""
        backoff = ExponentialBackoff()

        backoff.record_delay(30.0)
        assert backoff.cumulative_delay == 30.0

        backoff.reset_cumulative_delay()
        assert backoff.cumulative_delay == 0.0

    def test_budget_at_exact_limit(self):
        """Budget check at exact limit returns True."""
        backoff = ExponentialBackoff()

        # 100 seconds remaining, 50% budget = 50 seconds
        # Next delay of exactly 50 seconds is at limit
        result = backoff.is_within_budget(
            next_delay=50.0,
            remaining_seconds=100.0,
            budget_fraction=0.5,
        )

        assert result is True

    def test_custom_budget_fraction(self):
        """Custom budget fraction is respected."""
        backoff = ExponentialBackoff()

        # 100 seconds remaining, 25% budget = 25 seconds
        result = backoff.is_within_budget(
            next_delay=20.0,
            remaining_seconds=100.0,
            budget_fraction=0.25,
        )
        assert result is True

        result = backoff.is_within_budget(
            next_delay=30.0,
            remaining_seconds=100.0,
            budget_fraction=0.25,
        )
        assert result is False

    def test_zero_remaining_time_budget(self):
        """Zero remaining time means no budget available."""
        backoff = ExponentialBackoff()

        result = backoff.is_within_budget(
            next_delay=1.0,
            remaining_seconds=0.0,
            budget_fraction=0.5,
        )

        assert result is False

    def test_zero_next_delay_always_within_budget(self):
        """Zero next delay is always within budget."""
        backoff = ExponentialBackoff()

        # Even with cumulative delay at budget limit
        backoff.record_delay(50.0)

        result = backoff.is_within_budget(
            next_delay=0.0,
            remaining_seconds=100.0,
            budget_fraction=0.5,
        )

        assert result is True

    def test_default_budget_fraction_is_fifty_percent(self):
        """Default budget fraction is 50%."""
        backoff = ExponentialBackoff()

        # 100 seconds remaining, default 50% budget = 50 seconds
        result = backoff.is_within_budget(
            next_delay=49.0,
            remaining_seconds=100.0,
        )
        assert result is True

        result = backoff.is_within_budget(
            next_delay=51.0,
            remaining_seconds=100.0,
        )
        assert result is False
