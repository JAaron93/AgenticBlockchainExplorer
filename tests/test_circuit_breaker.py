"""Unit tests for CircuitBreaker.

Tests state transitions, failure threshold triggering, cool-down window
behavior, and logging of state transitions.

Requirements: 3.13, 3.14
"""

import time
from unittest.mock import MagicMock

from core.security.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitBreakerState,
)


class TestCircuitBreakerStateTransitions:
    """Tests for circuit breaker state transitions.

    Requirements: 3.13, 3.14
    """

    def test_initial_state_is_closed(self):
        """Circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker(explorer_name="etherscan")
        assert cb.state == CircuitBreakerState.CLOSED

    def test_closed_to_open_on_failure_threshold(self):
        """Circuit transitions CLOSED to OPEN when threshold is reached."""
        cb = CircuitBreaker(explorer_name="etherscan", failure_threshold=3)

        # Record failures up to threshold
        cb.record_failure()
        assert cb.state == CircuitBreakerState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitBreakerState.CLOSED
        cb.record_failure()  # Third failure triggers transition
        assert cb.state == CircuitBreakerState.OPEN

    def test_open_to_half_open_after_cooldown(self):
        """Circuit transitions OPEN to HALF_OPEN after cool-down window."""
        cb = CircuitBreaker(
            explorer_name="etherscan",
            failure_threshold=1,
            cool_down_seconds=0.1,  # Short cooldown for testing
        )

        # Trigger OPEN state
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

        # Wait for cooldown
        time.sleep(0.15)

        # State should transition to HALF_OPEN
        assert cb.state == CircuitBreakerState.HALF_OPEN

    def test_half_open_to_closed_on_success(self):
        """Circuit transitions from HALF_OPEN to CLOSED on success."""
        cb = CircuitBreaker(
            explorer_name="etherscan",
            failure_threshold=1,
            cool_down_seconds=0.1,
            half_open_success_threshold=1,
        )

        # Get to HALF_OPEN state
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Record success
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_half_open_to_open_on_failure(self):
        """Circuit transitions from HALF_OPEN to OPEN on failure."""
        cb = CircuitBreaker(
            explorer_name="etherscan",
            failure_threshold=1,
            cool_down_seconds=0.1,
            half_open_failure_threshold=1,
        )

        # Get to HALF_OPEN state
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Record failure
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

    def test_full_cycle_closed_open_half_open_closed(self):
        """Test complete state transition cycle."""
        cb = CircuitBreaker(
            explorer_name="etherscan",
            failure_threshold=2,
            cool_down_seconds=0.1,
            half_open_success_threshold=1,
        )

        # Start CLOSED
        assert cb.state == CircuitBreakerState.CLOSED

        # Transition to OPEN
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

        # Wait for HALF_OPEN
        time.sleep(0.15)
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Transition back to CLOSED
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED


class TestCircuitBreakerFailureThreshold:
    """Tests for failure threshold triggering.

    Requirements: 3.13, 3.18
    """

    def test_failure_count_increments(self):
        """Failure count increments on each failure."""
        cb = CircuitBreaker(explorer_name="etherscan", failure_threshold=5)

        assert cb.failure_count == 0
        cb.record_failure()
        assert cb.failure_count == 1
        cb.record_failure()
        assert cb.failure_count == 2

    def test_failure_count_resets_on_success_in_closed(self):
        """Failure count resets on success in CLOSED state."""
        cb = CircuitBreaker(explorer_name="etherscan", failure_threshold=5)

        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2

        cb.record_success()
        assert cb.failure_count == 0

    def test_exact_threshold_triggers_open(self):
        """Circuit opens exactly at failure threshold."""
        cb = CircuitBreaker(explorer_name="etherscan", failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.CLOSED

        cb.record_failure()  # Exactly at threshold
        assert cb.state == CircuitBreakerState.OPEN

    def test_default_failure_threshold_is_five(self):
        """Default failure threshold is 5."""
        cb = CircuitBreaker(explorer_name="etherscan")

        for _ in range(4):
            cb.record_failure()
        assert cb.state == CircuitBreakerState.CLOSED

        cb.record_failure()  # 5th failure
        assert cb.state == CircuitBreakerState.OPEN

    def test_half_open_success_threshold(self):
        """Multiple successes required when half_open_success_threshold > 1."""
        cb = CircuitBreaker(
            explorer_name="etherscan",
            failure_threshold=1,
            cool_down_seconds=0.1,
            half_open_success_threshold=3,
        )

        # Get to HALF_OPEN
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Need 3 successes
        cb.record_success()
        assert cb.state == CircuitBreakerState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitBreakerState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED


class TestCircuitBreakerCooldownWindow:
    """Tests for cool-down window behavior.

    Requirements: 3.14, 3.22
    """

    def test_requests_blocked_during_cooldown(self):
        """Requests are blocked while in OPEN state during cooldown."""
        cb = CircuitBreaker(
            explorer_name="etherscan",
            failure_threshold=1,
            cool_down_seconds=1.0,  # Long enough to test
        )

        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.is_allowed() is False

    def test_remaining_cooldown_decreases(self):
        """Remaining cooldown time decreases over time."""
        cb = CircuitBreaker(
            explorer_name="etherscan",
            failure_threshold=1,
            cool_down_seconds=0.5,
        )

        cb.record_failure()
        initial_remaining = cb.remaining_cooldown

        time.sleep(0.1)

        later_remaining = cb.remaining_cooldown
        assert later_remaining < initial_remaining

    def test_remaining_cooldown_zero_when_closed(self):
        """Remaining cooldown is 0 when circuit is CLOSED."""
        cb = CircuitBreaker(explorer_name="etherscan")
        assert cb.remaining_cooldown == 0.0

    def test_remaining_cooldown_zero_after_expiry(self):
        """Remaining cooldown is 0 after cooldown expires."""
        cb = CircuitBreaker(
            explorer_name="etherscan",
            failure_threshold=1,
            cool_down_seconds=0.1,
        )

        cb.record_failure()
        time.sleep(0.15)

        # State check triggers transition to HALF_OPEN
        _ = cb.state
        assert cb.remaining_cooldown == 0.0

    def test_default_cooldown_is_five_minutes(self):
        """Default cool-down is 300 seconds (5 minutes)."""
        cb = CircuitBreaker(explorer_name="etherscan", failure_threshold=1)
        cb.record_failure()

        # Should be close to 300 seconds
        assert 299 < cb.remaining_cooldown <= 300


class TestCircuitBreakerIsAllowed:
    """Tests for is_allowed() method.

    Requirements: 3.19, 3.20
    """

    def test_allowed_when_closed(self):
        """Requests are allowed when circuit is CLOSED."""
        cb = CircuitBreaker(explorer_name="etherscan")
        assert cb.is_allowed() is True

    def test_not_allowed_when_open(self):
        """Requests are not allowed when circuit is OPEN."""
        cb = CircuitBreaker(explorer_name="etherscan", failure_threshold=1)
        cb.record_failure()
        assert cb.is_allowed() is False

    def test_allowed_when_half_open(self):
        """Requests are allowed when circuit is HALF_OPEN."""
        cb = CircuitBreaker(
            explorer_name="etherscan",
            failure_threshold=1,
            cool_down_seconds=0.1,
        )

        cb.record_failure()
        time.sleep(0.15)

        assert cb.state == CircuitBreakerState.HALF_OPEN
        assert cb.is_allowed() is True


class TestCircuitBreakerOpenError:
    """Tests for CircuitBreakerOpenError.

    Requirements: 3.19
    """

    def test_error_contains_explorer_name(self):
        """Error contains the explorer name."""
        error = CircuitBreakerOpenError(
            explorer_name="etherscan",
            remaining_cooldown_seconds=60.0,
        )

        assert error.explorer_name == "etherscan"
        assert "etherscan" in str(error)

    def test_error_contains_remaining_cooldown(self):
        """Error contains remaining cooldown time."""
        error = CircuitBreakerOpenError(
            explorer_name="etherscan",
            remaining_cooldown_seconds=45.5,
        )

        assert error.remaining_cooldown_seconds == 45.5
        assert "45.5" in str(error)

    def test_error_code_is_circuit_open(self):
        """Error code is CIRCUIT_OPEN."""
        error = CircuitBreakerOpenError(
            explorer_name="etherscan",
            remaining_cooldown_seconds=60.0,
        )

        assert error.error_code == "CIRCUIT_OPEN"

    def test_get_open_error_method(self):
        """get_open_error() returns correct error."""
        cb = CircuitBreaker(
            explorer_name="bscscan",
            failure_threshold=1,
            cool_down_seconds=120.0,
        )

        cb.record_failure()
        error = cb.get_open_error()

        assert isinstance(error, CircuitBreakerOpenError)
        assert error.explorer_name == "bscscan"
        assert error.remaining_cooldown_seconds > 0


class TestCircuitBreakerLogging:
    """Tests for logging of state transitions.

    Requirements: 3.25
    """

    def test_logs_closed_to_open_transition(self):
        """Logs transition from CLOSED to OPEN."""
        mock_logger = MagicMock()
        cb = CircuitBreaker(
            explorer_name="etherscan",
            failure_threshold=1,
            logger=mock_logger,
        )

        cb.record_failure()

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "Circuit breaker state transition" in call_args[0][0]

        extra = call_args[1]["extra"]
        assert extra["explorer_name"] == "etherscan"
        assert extra["previous_state"] == "closed"
        assert extra["new_state"] == "open"

    def test_logs_open_to_half_open_transition(self):
        """Logs transition from OPEN to HALF_OPEN."""
        mock_logger = MagicMock()
        cb = CircuitBreaker(
            explorer_name="etherscan",
            failure_threshold=1,
            cool_down_seconds=0.1,
            logger=mock_logger,
        )

        cb.record_failure()
        mock_logger.reset_mock()

        time.sleep(0.15)
        _ = cb.state  # Trigger transition

        mock_logger.info.assert_called_once()
        extra = mock_logger.info.call_args[1]["extra"]
        assert extra["previous_state"] == "open"
        assert extra["new_state"] == "half_open"

    def test_logs_half_open_to_closed_transition(self):
        """Logs transition from HALF_OPEN to CLOSED."""
        mock_logger = MagicMock()
        cb = CircuitBreaker(
            explorer_name="etherscan",
            failure_threshold=1,
            cool_down_seconds=0.1,
            half_open_success_threshold=1,
            logger=mock_logger,
        )

        cb.record_failure()
        time.sleep(0.15)
        _ = cb.state  # Trigger HALF_OPEN
        mock_logger.reset_mock()

        cb.record_success()

        mock_logger.info.assert_called_once()
        extra = mock_logger.info.call_args[1]["extra"]
        assert extra["previous_state"] == "half_open"
        assert extra["new_state"] == "closed"

    def test_logs_half_open_to_open_transition(self):
        """Logs transition from HALF_OPEN to OPEN."""
        mock_logger = MagicMock()
        cb = CircuitBreaker(
            explorer_name="etherscan",
            failure_threshold=1,
            cool_down_seconds=0.1,
            half_open_failure_threshold=1,
            logger=mock_logger,
        )

        cb.record_failure()
        time.sleep(0.15)
        _ = cb.state  # Trigger HALF_OPEN
        mock_logger.reset_mock()

        cb.record_failure()

        mock_logger.info.assert_called_once()
        extra = mock_logger.info.call_args[1]["extra"]
        assert extra["previous_state"] == "half_open"
        assert extra["new_state"] == "open"

    def test_log_includes_failure_count(self):
        """Log includes failure count."""
        mock_logger = MagicMock()
        cb = CircuitBreaker(
            explorer_name="etherscan",
            failure_threshold=3,
            logger=mock_logger,
        )

        cb.record_failure()
        cb.record_failure()
        cb.record_failure()

        extra = mock_logger.info.call_args[1]["extra"]
        assert "failure_count" in extra

    def test_log_includes_failure_threshold(self):
        """Log includes failure threshold."""
        mock_logger = MagicMock()
        cb = CircuitBreaker(
            explorer_name="etherscan",
            failure_threshold=7,
            logger=mock_logger,
        )

        for _ in range(7):
            cb.record_failure()

        extra = mock_logger.info.call_args[1]["extra"]
        assert extra["failure_threshold"] == 7

    def test_log_includes_cooldown_seconds(self):
        """Log includes cool-down seconds."""
        mock_logger = MagicMock()
        cb = CircuitBreaker(
            explorer_name="etherscan",
            failure_threshold=1,
            cool_down_seconds=180.0,
            logger=mock_logger,
        )

        cb.record_failure()

        extra = mock_logger.info.call_args[1]["extra"]
        assert extra["cool_down_seconds"] == 180.0

    def test_log_includes_timestamp(self):
        """Log includes ISO-formatted timestamp."""
        mock_logger = MagicMock()
        cb = CircuitBreaker(
            explorer_name="etherscan",
            failure_threshold=1,
            logger=mock_logger,
        )

        cb.record_failure()

        extra = mock_logger.info.call_args[1]["extra"]
        assert "timestamp" in extra
        # Should be ISO format
        assert "T" in extra["timestamp"]

    def test_no_logging_without_logger(self):
        """No errors when logger is not provided."""
        cb = CircuitBreaker(
            explorer_name="etherscan",
            failure_threshold=1,
        )

        # Should not raise any errors
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN


class TestCircuitBreakerReset:
    """Tests for reset() method."""

    def test_reset_returns_to_closed(self):
        """Reset returns circuit to CLOSED state."""
        cb = CircuitBreaker(explorer_name="etherscan", failure_threshold=1)

        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

        cb.reset()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_reset_clears_failure_count(self):
        """Reset clears failure count."""
        cb = CircuitBreaker(explorer_name="etherscan", failure_threshold=5)

        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2

        cb.reset()
        assert cb.failure_count == 0

    def test_reset_clears_remaining_cooldown(self):
        """Reset clears remaining cooldown."""
        cb = CircuitBreaker(
            explorer_name="etherscan",
            failure_threshold=1,
            cool_down_seconds=300.0,
        )

        cb.record_failure()
        assert cb.remaining_cooldown > 0

        cb.reset()
        assert cb.remaining_cooldown == 0.0
