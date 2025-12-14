"""Timeout management for agent collection runs.

This module provides the TimeoutManager class that manages hierarchical
timeouts for agent runs, including overall run timeout and per-collection
timeouts.

Requirements: 6.1, 6.2, 6.3, 6.6
"""

import asyncio
import logging
import time
from typing import Any, Coroutine, Optional, TypeVar

from config.models import TimeoutConfig


logger = logging.getLogger(__name__)

T = TypeVar("T")


class AgentTimeoutError(Exception):
    """Raised when a timeout is exceeded."""

    pass


class CollectionTimeoutError(AgentTimeoutError):
    """Raised when per-collection timeout is exceeded."""

    def __init__(self, collection_name: str, timeout_seconds: float):
        self.collection_name = collection_name
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Collection '{collection_name}' exceeded timeout of "
            f"{timeout_seconds:.1f} seconds"
        )


class OverallTimeoutError(AgentTimeoutError):
    """Raised when overall run timeout is exceeded."""

    def __init__(self, timeout_seconds: float, elapsed_seconds: float):
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds
        super().__init__(
            f"Overall run timeout of {timeout_seconds:.1f} seconds exceeded "
            f"(elapsed: {elapsed_seconds:.1f} seconds)"
        )


class TimeoutManager:
    """Manages hierarchical timeouts for agent runs.

    This class handles timeout enforcement at multiple levels:
    - Overall run timeout: Maximum time for the entire agent run
    - Per-collection timeout: Maximum time for each stablecoin/explorer pair

    The per-collection timeout may be dynamically adjusted based on the
    number of collections to ensure all can complete within the overall
    timeout.

    Requirements: 6.1, 6.2, 6.3, 6.6
    """

    # Minimum per-collection timeout in seconds (Requirement 6.5)
    MIN_PER_COLLECTION_TIMEOUT = 60.0

    # Warning threshold - warn when this many seconds remain
    WARNING_THRESHOLD_SECONDS = 60.0

    def __init__(self, config: TimeoutConfig, num_collections: int):
        """Initialize with timeout config and expected collection count.

        Args:
            config: TimeoutConfig with timeout settings.
            num_collections: Expected number of collections (stablecoins × explorers).

        Raises:
            ValueError: If num_collections is invalid or timeouts cannot be satisfied.
        """
        if num_collections < 1:
            raise ValueError("num_collections must be at least 1")

        self._config = config
        self._num_collections = num_collections
        self._start_time: Optional[float] = None
        self._is_started = False

        # Calculate effective per-collection timeout (Requirement 6.5)
        self._effective_per_collection_timeout = self._calculate_per_collection_timeout()

        logger.info(
            f"TimeoutManager initialized: overall={config.overall_run_timeout_seconds}s, "
            f"per_collection={self._effective_per_collection_timeout:.1f}s, "
            f"shutdown={config.shutdown_timeout_seconds}s, "
            f"num_collections={num_collections}"
        )

    def _calculate_per_collection_timeout(self) -> float:
        """Calculate effective per-collection timeout.

        If the configured per-collection timeout would cause the total
        collection time to exceed the overall timeout, dynamically adjust
        it down (with a minimum of MIN_PER_COLLECTION_TIMEOUT).

        Returns:
            Effective per-collection timeout in seconds.

        Raises:
            ValueError: If timeout constraints cannot be satisfied.

        Requirements: 6.5
        """
        overall = self._config.overall_run_timeout_seconds
        shutdown = self._config.shutdown_timeout_seconds
        configured_per_collection = self._config.per_collection_timeout_seconds

        # Available time for all collections (excluding shutdown buffer)
        available_time = overall - shutdown

        # Maximum time if all collections run sequentially
        max_sequential_time = self._num_collections * configured_per_collection

        if max_sequential_time <= available_time:
            # Configured timeout fits within overall timeout
            return configured_per_collection

        # Need to adjust per-collection timeout
        adjusted_timeout = available_time / self._num_collections

        if adjusted_timeout < self.MIN_PER_COLLECTION_TIMEOUT:
            raise ValueError(
                f"Cannot satisfy timeout constraints: "
                f"{self._num_collections} collections × "
                f"{self.MIN_PER_COLLECTION_TIMEOUT}s minimum = "
                f"{self._num_collections * self.MIN_PER_COLLECTION_TIMEOUT}s, "
                f"but only {available_time}s available "
                f"(overall={overall}s - shutdown={shutdown}s). "
                f"Reduce number of collections or increase overall timeout."
            )

        logger.warning(
            f"Adjusted per-collection timeout from {configured_per_collection}s "
            f"to {adjusted_timeout:.1f}s to fit {self._num_collections} collections "
            f"within overall timeout of {overall}s"
        )

        return adjusted_timeout

    @property
    def overall_timeout(self) -> float:
        """Get overall run timeout in seconds."""
        return float(self._config.overall_run_timeout_seconds)

    @property
    def per_collection_timeout(self) -> float:
        """Get per-stablecoin collection timeout in seconds.

        This may be less than the configured value if dynamic adjustment
        was applied to fit all collections within the overall timeout.
        """
        return self._effective_per_collection_timeout

    @property
    def shutdown_timeout(self) -> float:
        """Get shutdown timeout in seconds."""
        return float(self._config.shutdown_timeout_seconds)

    def time_remaining(self) -> float:
        """Get remaining time for overall run.

        Returns:
            Remaining time in seconds, or overall_timeout if not started.
        """
        if not self._is_started or self._start_time is None:
            return self.overall_timeout

        elapsed = time.monotonic() - self._start_time
        remaining = self.overall_timeout - elapsed
        return max(0.0, remaining)

    def time_elapsed(self) -> float:
        """Get elapsed time since start.

        Returns:
            Elapsed time in seconds, or 0.0 if not started.
        """
        if not self._is_started or self._start_time is None:
            return 0.0

        return time.monotonic() - self._start_time

    def should_terminate(self) -> bool:
        """Check if overall timeout is approaching (within 60s).

        Returns:
            True if remaining time is less than WARNING_THRESHOLD_SECONDS.

        Requirements: 6.3
        """
        remaining = self.time_remaining()
        return remaining <= self.WARNING_THRESHOLD_SECONDS

    def is_expired(self) -> bool:
        """Check if overall timeout has been exceeded.

        Returns:
            True if no time remaining.
        """
        return self.time_remaining() <= 0.0

    def start(self) -> None:
        """Start the timeout clock.

        Should be called at the beginning of the agent run.
        """
        if self._is_started:
            logger.warning("TimeoutManager already started, ignoring start()")
            return

        self._start_time = time.monotonic()
        self._is_started = True
        logger.info(
            f"TimeoutManager started: overall_timeout={self.overall_timeout}s"
        )

    def reset(self) -> None:
        """Reset the timeout manager to initial state.

        Useful for testing or restarting a run.
        """
        self._start_time = None
        self._is_started = False
        logger.debug("TimeoutManager reset")

    async def run_with_timeout(
        self,
        coro: Coroutine[Any, Any, T],
        timeout: Optional[float] = None,
        collection_name: Optional[str] = None,
    ) -> T:
        """Run coroutine with timeout, raising AgentTimeoutError if exceeded.

        Args:
            coro: Coroutine to run.
            timeout: Timeout in seconds. If None, uses per_collection_timeout.
            collection_name: Name of collection for error messages.

        Returns:
            Result of the coroutine.

        Raises:
            CollectionTimeoutError: If the specified timeout is exceeded.
            OverallTimeoutError: If overall timeout is exceeded.
            asyncio.CancelledError: If the task is cancelled.

        Requirements: 6.1, 6.2, 6.6
        """
        if not self._is_started:
            self.start()

        # Check if overall timeout already exceeded
        if self.is_expired():
            raise OverallTimeoutError(
                self.overall_timeout,
                self.time_elapsed(),
            )

        # Determine effective timeout
        effective_timeout = timeout if timeout is not None else self.per_collection_timeout

        # Cap timeout to remaining time
        remaining = self.time_remaining()
        if effective_timeout > remaining:
            logger.warning(
                f"Requested timeout {effective_timeout:.1f}s exceeds remaining "
                f"time {remaining:.1f}s, using remaining time"
            )
            effective_timeout = remaining

        # Log if approaching overall timeout (Requirement 6.3)
        if self.should_terminate():
            logger.warning(
                f"Overall timeout approaching: {remaining:.1f}s remaining. "
                f"Preparing for graceful termination."
            )

        try:
            result = await asyncio.wait_for(coro, timeout=effective_timeout)
            return result
        except asyncio.TimeoutError:
            # Determine which timeout was hit
            if self.is_expired():
                raise OverallTimeoutError(
                    self.overall_timeout,
                    self.time_elapsed(),
                )
            else:
                name = collection_name or "unknown"
                raise CollectionTimeoutError(name, effective_timeout)

    def get_status(self) -> dict:
        """Get current timeout status for logging/monitoring.

        Returns:
            Dictionary with timeout status information.
        """
        return {
            "is_started": self._is_started,
            "overall_timeout_seconds": self.overall_timeout,
            "per_collection_timeout_seconds": self.per_collection_timeout,
            "shutdown_timeout_seconds": self.shutdown_timeout,
            "time_elapsed_seconds": self.time_elapsed(),
            "time_remaining_seconds": self.time_remaining(),
            "should_terminate": self.should_terminate(),
            "is_expired": self.is_expired(),
            "num_collections": self._num_collections,
        }
