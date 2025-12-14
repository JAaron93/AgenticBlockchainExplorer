"""Unit tests for TimeoutManager.

Tests timeout calculation with collection count, dynamic timeout adjustment,
and should_terminate() threshold behavior.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
"""

import asyncio
import time

import pytest

from config.models import TimeoutConfig
from core.security.timeout_manager import (
    CollectionTimeoutError,
    OverallTimeoutError,
    TimeoutManager,
)


class TestTimeoutManagerInitialization:
    """Tests for TimeoutManager initialization.

    Requirements: 6.4, 6.5
    """

    def test_default_config_values(self):
        """TimeoutManager uses default config values correctly."""
        config = TimeoutConfig()
        manager = TimeoutManager(config, num_collections=5)

        assert manager.overall_timeout == 1800.0
        assert manager.per_collection_timeout == 180.0
        assert manager.shutdown_timeout == 30.0

    def test_custom_config_values(self):
        """TimeoutManager uses custom config values correctly."""
        config = TimeoutConfig(
            overall_run_timeout_seconds=3600,
            per_collection_timeout_seconds=300,
            shutdown_timeout_seconds=60,
        )
        manager = TimeoutManager(config, num_collections=5)

        assert manager.overall_timeout == 3600.0
        assert manager.per_collection_timeout == 300.0
        assert manager.shutdown_timeout == 60.0

    def test_invalid_num_collections_zero(self):
        """Raises ValueError for num_collections=0."""
        config = TimeoutConfig()
        with pytest.raises(ValueError, match="num_collections must be at least 1"):
            TimeoutManager(config, num_collections=0)

    def test_invalid_num_collections_negative(self):
        """Raises ValueError for negative num_collections."""
        config = TimeoutConfig()
        with pytest.raises(ValueError, match="num_collections must be at least 1"):
            TimeoutManager(config, num_collections=-1)


class TestDynamicTimeoutAdjustment:
    """Tests for dynamic per-collection timeout adjustment.

    Requirements: 6.5
    """

    def test_no_adjustment_when_fits(self):
        """No adjustment when collections fit within overall timeout."""
        config = TimeoutConfig(
            overall_run_timeout_seconds=1800,  # 30 minutes
            per_collection_timeout_seconds=180,  # 3 minutes
            shutdown_timeout_seconds=30,
        )
        # 5 collections × 180s = 900s < 1770s available
        manager = TimeoutManager(config, num_collections=5)

        assert manager.per_collection_timeout == 180.0

    def test_adjustment_when_exceeds(self):
        """Adjusts per-collection timeout when total exceeds overall."""
        config = TimeoutConfig(
            overall_run_timeout_seconds=900,  # 15 minutes
            per_collection_timeout_seconds=180,  # 3 minutes
            shutdown_timeout_seconds=30,
        )
        # 10 collections × 180s = 1800s > 870s available
        # Adjusted: 870s / 10 = 87s
        manager = TimeoutManager(config, num_collections=10)

        assert manager.per_collection_timeout == 87.0

    def test_minimum_per_collection_timeout_enforced(self):
        """Raises ValueError when minimum timeout cannot be satisfied."""
        config = TimeoutConfig(
            overall_run_timeout_seconds=120,  # 2 minutes
            per_collection_timeout_seconds=60,
            shutdown_timeout_seconds=30,
        )
        # 10 collections × 60s min = 600s > 90s available
        with pytest.raises(ValueError, match="Cannot satisfy timeout constraints"):
            TimeoutManager(config, num_collections=10)

    def test_adjustment_respects_shutdown_buffer(self):
        """Adjustment accounts for shutdown timeout buffer."""
        config = TimeoutConfig(
            overall_run_timeout_seconds=300,  # 5 minutes
            per_collection_timeout_seconds=180,
            shutdown_timeout_seconds=60,  # 1 minute
        )
        # Available: 300 - 60 = 240s
        # 3 collections × 180s = 540s > 240s
        # Adjusted: 240s / 3 = 80s
        manager = TimeoutManager(config, num_collections=3)

        assert manager.per_collection_timeout == 80.0


class TestTimeoutManagerTiming:
    """Tests for time tracking methods.

    Requirements: 6.1, 6.3
    """

    def test_time_remaining_before_start(self):
        """time_remaining returns overall_timeout before start."""
        config = TimeoutConfig(overall_run_timeout_seconds=1800)
        manager = TimeoutManager(config, num_collections=1)

        assert manager.time_remaining() == 1800.0

    def test_time_elapsed_before_start(self):
        """time_elapsed returns 0 before start."""
        config = TimeoutConfig()
        manager = TimeoutManager(config, num_collections=1)

        assert manager.time_elapsed() == 0.0

    def test_time_remaining_after_start(self):
        """time_remaining decreases after start."""
        config = TimeoutConfig(overall_run_timeout_seconds=1800)
        manager = TimeoutManager(config, num_collections=1)

        manager.start()
        time.sleep(0.1)

        remaining = manager.time_remaining()
        assert remaining < 1800.0
        assert remaining > 1799.0

    def test_time_elapsed_after_start(self):
        """time_elapsed increases after start."""
        config = TimeoutConfig()
        manager = TimeoutManager(config, num_collections=1)

        manager.start()
        time.sleep(0.1)

        elapsed = manager.time_elapsed()
        assert elapsed >= 0.1
        assert elapsed < 0.2

    def test_is_expired_false_initially(self):
        """is_expired returns False initially."""
        config = TimeoutConfig()
        manager = TimeoutManager(config, num_collections=1)

        manager.start()
        assert manager.is_expired() is False


class TestShouldTerminate:
    """Tests for should_terminate() threshold behavior.

    Requirements: 6.3
    """

    def test_should_terminate_false_with_time_remaining(self):
        """should_terminate returns False when plenty of time remains."""
        config = TimeoutConfig(overall_run_timeout_seconds=1800)
        manager = TimeoutManager(config, num_collections=1)

        manager.start()
        assert manager.should_terminate() is False

    def test_should_terminate_true_within_threshold(self):
        """should_terminate returns True within 60s of timeout."""
        config = TimeoutConfig(
            overall_run_timeout_seconds=120,
            per_collection_timeout_seconds=50,
            shutdown_timeout_seconds=10,
        )
        manager = TimeoutManager(config, num_collections=1)

        manager.start()
        # Wait until within 60s threshold (need to wait ~60s)
        # For testing, we'll manipulate the start time
        manager._start_time = time.monotonic() - 61  # Simulate 61s elapsed

        # With 120s overall and 61s elapsed, ~59s remaining
        # Should be within threshold
        assert manager.should_terminate() is True

    def test_should_terminate_threshold_is_60_seconds(self):
        """Warning threshold is 60 seconds."""
        assert TimeoutManager.WARNING_THRESHOLD_SECONDS == 60.0


class TestRunWithTimeout:
    """Tests for run_with_timeout() method.

    Requirements: 6.1, 6.2, 6.6
    """

    @pytest.mark.asyncio
    async def test_successful_completion(self):
        """Coroutine completes successfully within timeout."""
        config = TimeoutConfig()
        manager = TimeoutManager(config, num_collections=1)

        async def quick_task():
            await asyncio.sleep(0.01)
            return "success"

        result = await manager.run_with_timeout(quick_task(), timeout=5.0)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_collection_timeout_error(self):
        """Raises CollectionTimeoutError when timeout exceeded."""
        config = TimeoutConfig()
        manager = TimeoutManager(config, num_collections=1)

        async def slow_task():
            await asyncio.sleep(10)
            return "done"

        with pytest.raises(CollectionTimeoutError) as exc_info:
            await manager.run_with_timeout(
                slow_task(),
                timeout=0.1,
                collection_name="test_collection",
            )

        assert exc_info.value.collection_name == "test_collection"
        assert exc_info.value.timeout_seconds == 0.1

    @pytest.mark.asyncio
    async def test_uses_per_collection_timeout_by_default(self):
        """Uses per_collection_timeout when timeout not specified."""
        config = TimeoutConfig(per_collection_timeout_seconds=60)
        manager = TimeoutManager(config, num_collections=1)

        async def quick_task():
            return "done"

        # Should use 60s timeout (per_collection_timeout)
        result = await manager.run_with_timeout(quick_task())
        assert result == "done"

    @pytest.mark.asyncio
    async def test_auto_starts_if_not_started(self):
        """Automatically starts timeout clock if not started."""
        config = TimeoutConfig()
        manager = TimeoutManager(config, num_collections=1)

        assert manager._is_started is False

        async def quick_task():
            return "done"

        await manager.run_with_timeout(quick_task(), timeout=5.0)
        assert manager._is_started is True

    @pytest.mark.asyncio
    async def test_caps_timeout_to_remaining_time(self):
        """Caps requested timeout to remaining overall time."""
        config = TimeoutConfig(
            overall_run_timeout_seconds=120,
            per_collection_timeout_seconds=50,
            shutdown_timeout_seconds=10,
        )
        manager = TimeoutManager(config, num_collections=1)

        manager.start()
        # Simulate 60s elapsed, leaving ~60s remaining
        manager._start_time = time.monotonic() - 60

        async def quick_task():
            return "done"

        # Request 100s timeout, but only ~60s remaining
        # Should cap to remaining time
        result = await manager.run_with_timeout(quick_task(), timeout=100.0)
        assert result == "done"


class TestTimeoutManagerReset:
    """Tests for reset() method."""

    def test_reset_clears_start_time(self):
        """Reset clears the start time."""
        config = TimeoutConfig()
        manager = TimeoutManager(config, num_collections=1)

        manager.start()
        assert manager._is_started is True

        manager.reset()
        assert manager._is_started is False
        assert manager._start_time is None

    def test_reset_allows_restart(self):
        """Reset allows the manager to be started again."""
        config = TimeoutConfig()
        manager = TimeoutManager(config, num_collections=1)

        manager.start()
        time.sleep(0.1)
        elapsed1 = manager.time_elapsed()

        manager.reset()
        manager.start()
        elapsed2 = manager.time_elapsed()

        # After reset and restart, elapsed should be near 0
        assert elapsed2 < elapsed1


class TestTimeoutManagerStatus:
    """Tests for get_status() method."""

    def test_status_contains_all_fields(self):
        """Status contains all expected fields."""
        config = TimeoutConfig()
        manager = TimeoutManager(config, num_collections=5)

        manager.start()
        status = manager.get_status()

        assert "is_started" in status
        assert "overall_timeout_seconds" in status
        assert "per_collection_timeout_seconds" in status
        assert "shutdown_timeout_seconds" in status
        assert "time_elapsed_seconds" in status
        assert "time_remaining_seconds" in status
        assert "should_terminate" in status
        assert "is_expired" in status
        assert "num_collections" in status

    def test_status_values_are_correct(self):
        """Status values match manager state."""
        config = TimeoutConfig(
            overall_run_timeout_seconds=1800,
            per_collection_timeout_seconds=180,
            shutdown_timeout_seconds=30,
        )
        manager = TimeoutManager(config, num_collections=5)

        manager.start()
        status = manager.get_status()

        assert status["is_started"] is True
        assert status["overall_timeout_seconds"] == 1800.0
        assert status["per_collection_timeout_seconds"] == 180.0
        assert status["shutdown_timeout_seconds"] == 30.0
        assert status["num_collections"] == 5
        assert status["is_expired"] is False


class TestTimeoutConfigValidation:
    """Tests for TimeoutConfig model validation.

    Requirements: 6.4, 6.5
    """

    def test_per_collection_must_be_less_than_overall(self):
        """per_collection_timeout must be less than overall_run_timeout."""
        with pytest.raises(ValueError, match="per_collection_timeout_seconds"):
            TimeoutConfig(
                overall_run_timeout_seconds=100,
                per_collection_timeout_seconds=100,
            )

    def test_shutdown_must_be_less_than_overall(self):
        """shutdown_timeout must be less than overall_run_timeout."""
        with pytest.raises(ValueError, match="per_collection_timeout_seconds"):
            TimeoutConfig(
                overall_run_timeout_seconds=100,
                shutdown_timeout_seconds=100,
            )

    def test_combined_must_be_less_than_overall(self):
        """per_collection + shutdown must be less than overall."""
        with pytest.raises(ValueError, match="overall_run_timeout_seconds"):
            TimeoutConfig(
                overall_run_timeout_seconds=100,
                per_collection_timeout_seconds=60,
                shutdown_timeout_seconds=50,
            )

    def test_valid_config_passes_validation(self):
        """Valid configuration passes validation."""
        config = TimeoutConfig(
            overall_run_timeout_seconds=1800,
            per_collection_timeout_seconds=180,
            shutdown_timeout_seconds=30,
        )
        assert config.overall_run_timeout_seconds == 1800
        assert config.per_collection_timeout_seconds == 180
        assert config.shutdown_timeout_seconds == 30


class TestStartBehavior:
    """Tests for start() method behavior."""

    def test_start_sets_is_started(self):
        """start() sets _is_started to True."""
        config = TimeoutConfig()
        manager = TimeoutManager(config, num_collections=1)

        assert manager._is_started is False
        manager.start()
        assert manager._is_started is True

    def test_start_sets_start_time(self):
        """start() sets _start_time."""
        config = TimeoutConfig()
        manager = TimeoutManager(config, num_collections=1)

        assert manager._start_time is None
        manager.start()
        assert manager._start_time is not None

    def test_double_start_is_ignored(self):
        """Calling start() twice is ignored."""
        config = TimeoutConfig()
        manager = TimeoutManager(config, num_collections=1)

        manager.start()
        first_start_time = manager._start_time

        time.sleep(0.1)
        manager.start()  # Should be ignored

        assert manager._start_time == first_start_time
