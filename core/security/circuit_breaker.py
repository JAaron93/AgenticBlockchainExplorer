"""Circuit breaker and exponential backoff for API resilience.

This module provides the CircuitBreaker and ExponentialBackoff classes
for handling API failures gracefully with retry logic and circuit breaking.

Requirements: 3.11, 3.12, 3.13, 3.14, 3.20, 3.21
"""

import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.security.secure_logger import SecureLogger


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and request is rejected.

    Requirements: 3.19, 3.20
    """

    def __init__(
        self,
        explorer_name: str,
        remaining_cooldown_seconds: float,
    ):
        self.explorer_name = explorer_name
        self.remaining_cooldown_seconds = remaining_cooldown_seconds
        self.error_code = "CIRCUIT_OPEN"
        super().__init__(
            f"Circuit breaker is OPEN for {explorer_name}. "
            f"Remaining cool-down: {remaining_cooldown_seconds:.1f}s"
        )


class ExponentialBackoff:
    """Calculates exponential backoff delays with jitter.

    This class implements exponential backoff with configurable parameters
    and support for honoring rate-limit headers from API responses.

    Requirements: 3.11, 3.12, 3.20, 3.21
    """

    # Maximum valid delay from rate-limit headers (1 hour)
    MAX_HEADER_DELAY_SECONDS = 3600

    def __init__(
        self,
        base_delay: float = 1.0,
        multiplier: float = 2.0,
        max_delay: float = 60.0,
        jitter_factor: float = 0.1,
    ):
        """Initialize backoff calculator.

        Args:
            base_delay: Base delay in seconds for first retry.
            multiplier: Multiplier applied for each subsequent retry.
            max_delay: Maximum delay cap in seconds.
            jitter_factor: Random jitter factor (0.0 to 1.0) to add variance.
        """
        self._base_delay = base_delay
        self._multiplier = multiplier
        self._max_delay = max_delay
        self._jitter_factor = jitter_factor
        self._cumulative_delay = 0.0

    @property
    def cumulative_delay(self) -> float:
        """Get total cumulative delay so far."""
        return self._cumulative_delay

    def reset_cumulative_delay(self) -> None:
        """Reset cumulative delay counter."""
        self._cumulative_delay = 0.0

    def get_delay(self, attempt: int) -> float:
        """Get delay for given attempt number with jitter.

        Args:
            attempt: Attempt number (0-indexed, so first retry is attempt 0).

        Returns:
            Delay in seconds with jitter applied.

        Requirements: 3.11
        """
        # Calculate base exponential delay
        delay = self._base_delay * (self._multiplier ** attempt)

        # Apply max delay cap
        delay = min(delay, self._max_delay)

        # Apply jitter (random variance)
        if self._jitter_factor > 0:
            jitter = delay * self._jitter_factor * random.random()
            delay = delay + jitter

        return delay

    def get_delay_honoring_headers(
        self,
        attempt: int,
        retry_after: Optional[str] = None,
        rate_limit_reset: Optional[str] = None,
    ) -> float:
        """Get delay honoring rate-limit headers if present.

        Header precedence (per Requirements 3.12):
        1. Retry-After header takes highest precedence if present and valid
        2. X-RateLimit-Reset is used if Retry-After is absent
        3. Exponential backoff is used as fallback

        If both headers are present with different valid values, uses the
        larger of the two computed delays to ensure compliance with the
        stricter limit.

        Args:
            attempt: Attempt number (0-indexed).
            retry_after: Value of Retry-After header.
            rate_limit_reset: Value of X-RateLimit-Reset header.

        Returns:
            Delay in seconds.

        Requirements: 3.12, 3.13, 3.14, 3.15, 3.16, 3.17
        """
        delays_from_headers: list[float] = []

        # Parse Retry-After header
        if retry_after:
            parsed_delay = self._parse_retry_after(retry_after)
            if parsed_delay is not None:
                delays_from_headers.append(parsed_delay)

        # Parse X-RateLimit-Reset header
        if rate_limit_reset:
            parsed_delay = self._parse_rate_limit_reset(rate_limit_reset)
            if parsed_delay is not None:
                delays_from_headers.append(parsed_delay)

        # If we have valid header delays, use the maximum (stricter limit)
        if delays_from_headers:
            return max(delays_from_headers)

        # Fall back to exponential backoff
        return self.get_delay(attempt)

    def _parse_retry_after(self, value: str) -> Optional[float]:
        """Parse Retry-After header value.

        Supports both delta-seconds (numeric) and HTTP-date formats
        per RFC 7231.

        Args:
            value: Retry-After header value.

        Returns:
            Delay in seconds, or None if invalid.

        Requirements: 3.13, 3.16, 3.17
        """
        if not value:
            return None

        value = value.strip()

        # Try parsing as delta-seconds (numeric)
        try:
            delay = float(value)
            return self._validate_header_delay(delay)
        except ValueError:
            pass

        # Try parsing as HTTP-date
        try:
            parsed_date = parsedate_to_datetime(value)
            now = datetime.now(timezone.utc)
            delay = (parsed_date - now).total_seconds()
            return self._validate_header_delay(delay)
        except (ValueError, TypeError):
            pass

        # Malformed header
        return None

    def _parse_rate_limit_reset(self, value: str) -> Optional[float]:
        """Parse X-RateLimit-Reset header value.

        Treats value as Unix epoch timestamp.

        Args:
            value: X-RateLimit-Reset header value.

        Returns:
            Delay in seconds, or None if invalid.

        Requirements: 3.14, 3.16, 3.17
        """
        if not value:
            return None

        try:
            reset_timestamp = float(value.strip())
            now = time.time()
            delay = reset_timestamp - now
            return self._validate_header_delay(delay)
        except (ValueError, TypeError):
            return None

    def _validate_header_delay(self, delay: float) -> Optional[float]:
        """Validate delay from header is within acceptable bounds.

        Args:
            delay: Computed delay in seconds.

        Returns:
            Delay if valid, None if invalid (negative or exceeds max).

        Requirements: 3.16
        """
        if delay < 0 or delay > self.MAX_HEADER_DELAY_SECONDS:
            return None
        return delay

    def is_within_budget(
        self,
        next_delay: float,
        remaining_seconds: float,
        budget_fraction: float = 0.5,
    ) -> bool:
        """Check if delay fits within remaining time budget.

        The retry budget is measured as total cumulative delay time that
        does not exceed a fraction of the remaining overall run timeout.

        Args:
            next_delay: The next delay to be applied.
            remaining_seconds: Remaining time in the overall run timeout.
            budget_fraction: Fraction of remaining time for retries.

        Returns:
            True if (cumulative_delay + next_delay) is within budget.

        Requirements: 3.20, 3.21
        """
        budget = remaining_seconds * budget_fraction
        return (self._cumulative_delay + next_delay) <= budget

    def record_delay(self, delay: float) -> None:
        """Record a delay that was actually used.

        Call this after waiting to track cumulative delay for budget.

        Args:
            delay: The delay that was applied.
        """
        self._cumulative_delay += delay


class CircuitBreakerState(Enum):
    """Circuit breaker states.

    Requirements: 3.13, 3.14
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for explorer API resilience.

    Implements the circuit breaker pattern to prevent cascading failures
    when an external API is degraded or unavailable.

    State transitions:
    - CLOSED -> OPEN: When failure_count >= failure_threshold
    - OPEN -> HALF_OPEN: Automatically after cool_down_seconds
    - HALF_OPEN -> CLOSED: After half_open_success_threshold successes
    - HALF_OPEN -> OPEN: Immediately on any failure

    Requirements: 3.13, 3.14, 3.18, 3.19, 3.22, 3.23, 3.24, 3.25
    """

    def __init__(
        self,
        explorer_name: str,
        failure_threshold: int = 5,
        cool_down_seconds: float = 300.0,
        half_open_success_threshold: int = 1,
        half_open_failure_threshold: int = 1,
        logger: Optional["SecureLogger"] = None,
    ):
        """Initialize circuit breaker.

        Args:
            explorer_name: Name of the explorer this breaker protects.
            failure_threshold: Number of failures before CLOSED->OPEN.
            cool_down_seconds: Time in OPEN before HALF-OPEN transition.
            half_open_success_threshold: Successes for HALF-OPEN->CLOSED.
            half_open_failure_threshold: Failures for HALF-OPEN->OPEN.
            logger: Logger for state transitions.
        """
        self._explorer_name = explorer_name
        self._failure_threshold = failure_threshold
        self._cool_down_seconds = cool_down_seconds
        self._half_open_success_threshold = half_open_success_threshold
        self._half_open_failure_threshold = half_open_failure_threshold
        self._logger = logger

        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._last_state_change: Optional[float] = None
        self._opened_at: Optional[float] = None

    @property
    def explorer_name(self) -> str:
        """Get the explorer name this circuit breaker protects."""
        return self._explorer_name

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit state.

        This property also handles automatic OPEN->HALF_OPEN transition
        when the cool-down window has elapsed.

        Requirements: 3.22
        """
        if self._state == CircuitBreakerState.OPEN:
            if self._opened_at is not None:
                elapsed = time.time() - self._opened_at
                if elapsed >= self._cool_down_seconds:
                    self._transition_to(CircuitBreakerState.HALF_OPEN)

        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count

    @property
    def remaining_cooldown(self) -> float:
        """Get remaining cool-down time in seconds.

        Returns 0 if not in OPEN state or cool-down has elapsed.
        """
        if self._state != CircuitBreakerState.OPEN or self._opened_at is None:
            return 0.0

        elapsed = time.time() - self._opened_at
        remaining = self._cool_down_seconds - elapsed
        return max(0.0, remaining)

    def is_allowed(self) -> bool:
        """Check if request is allowed through circuit.

        Returns:
            True if request should proceed, False if circuit is open.

        Requirements: 3.19, 3.20
        """
        current_state = self.state

        if current_state == CircuitBreakerState.OPEN:
            return False

        return True

    def record_success(self) -> None:
        """Record successful request.

        In CLOSED state: resets failure count.
        In HALF_OPEN state: increments success count, may close circuit.

        Requirements: 3.23
        """
        if self._state == CircuitBreakerState.CLOSED:
            self._failure_count = 0

        elif self._state == CircuitBreakerState.HALF_OPEN:
            self._success_count += 1

            if self._success_count >= self._half_open_success_threshold:
                self._transition_to(CircuitBreakerState.CLOSED)

    def record_failure(self) -> None:
        """Record failed request.

        In CLOSED state: increments failure count, may open circuit.
        In HALF_OPEN state: immediately transitions back to OPEN.

        Requirements: 3.18, 3.24
        """
        self._last_failure_time = time.time()

        if self._state == CircuitBreakerState.CLOSED:
            self._failure_count += 1

            if self._failure_count >= self._failure_threshold:
                self._transition_to(CircuitBreakerState.OPEN)

        elif self._state == CircuitBreakerState.HALF_OPEN:
            self._transition_to(CircuitBreakerState.OPEN)

    def _transition_to(self, new_state: CircuitBreakerState) -> None:
        """Transition to new state with logging.

        Args:
            new_state: The state to transition to.

        Requirements: 3.25
        """
        old_state = self._state
        now = time.time()

        self._state = new_state
        self._last_state_change = now

        if new_state == CircuitBreakerState.OPEN:
            self._opened_at = now
            self._success_count = 0

        elif new_state == CircuitBreakerState.HALF_OPEN:
            self._success_count = 0

        elif new_state == CircuitBreakerState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
            self._opened_at = None

        if self._logger:
            timestamp = datetime.fromtimestamp(now, tz=timezone.utc)
            self._logger.info(
                "Circuit breaker state transition",
                extra={
                    "explorer_name": self._explorer_name,
                    "previous_state": old_state.value,
                    "new_state": new_state.value,
                    "failure_count": self._failure_count,
                    "failure_threshold": self._failure_threshold,
                    "cool_down_seconds": self._cool_down_seconds,
                    "timestamp": timestamp.isoformat(),
                },
            )

    def get_open_error(self) -> CircuitBreakerOpenError:
        """Get error to raise when circuit is open.

        Returns:
            CircuitBreakerOpenError with explorer name and cooldown.

        Requirements: 3.19
        """
        return CircuitBreakerOpenError(
            explorer_name=self._explorer_name,
            remaining_cooldown_seconds=self.remaining_cooldown,
        )

    def reset(self) -> None:
        """Reset circuit breaker to initial CLOSED state.

        Useful for testing or manual intervention.
        """
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._last_state_change = None
        self._opened_at = None
