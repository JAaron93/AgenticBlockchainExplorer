"""Unit tests for GracefulTerminator.

Tests for graceful termination handling including:
- Task cancellation (Requirement 3.7)
- Partial result persistence (Requirements 3.8, 3.9)
- Termination logging (Requirement 3.10)

Requirements: 3.7, 3.8, 3.9, 3.10
"""

import asyncio
import json
import logging
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, List

import pytest

from collectors.models import ActivityType, ExplorerData, Holder, Transaction
from core.security.graceful_terminator import (
    ExplorerDataOutput,
    GracefulTerminator,
    TerminationReport,
)


# --- Test Fixtures ---


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory for tests."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def terminator(temp_output_dir: Path) -> GracefulTerminator:
    """Create a GracefulTerminator instance for testing."""
    return GracefulTerminator(
        shutdown_timeout=5.0,
        output_directory=temp_output_dir,
    )


@pytest.fixture
def sample_transaction() -> Transaction:
    """Create a sample transaction for testing."""
    return Transaction(
        transaction_hash="0x" + "a" * 64,
        block_number=12345678,
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        from_address="0x" + "1" * 40,
        to_address="0x" + "2" * 40,
        amount=Decimal("1000.50"),
        stablecoin="USDC",
        chain="ethereum",
        activity_type=ActivityType.TRANSACTION,
        source_explorer="etherscan",
        gas_used=21000,
        gas_price=Decimal("50000000000"),
    )


@pytest.fixture
def sample_holder() -> Holder:
    """Create a sample holder for testing."""
    return Holder(
        address="0x" + "3" * 40,
        balance=Decimal("50000.00"),
        stablecoin="USDT",
        chain="ethereum",
        first_seen=datetime(2023, 1, 1, 0, 0, 0),
        last_activity=datetime(2024, 1, 15, 10, 30, 0),
        is_store_of_value=True,
        source_explorer="etherscan",
    )


@pytest.fixture
def sample_explorer_data(
    sample_transaction: Transaction,
    sample_holder: Holder,
) -> ExplorerData:
    """Create sample ExplorerData for testing."""
    return ExplorerData(
        explorer_name="etherscan",
        chain="ethereum",
        transactions=[sample_transaction],
        holders=[sample_holder],
        errors=[],
        collection_time_seconds=45.2,
    )


# --- Task Cancellation Tests (Requirement 3.7) ---


class TestTaskCancellation:
    """Tests for task cancellation during graceful termination."""

    @pytest.mark.asyncio
    async def test_cancel_tasks_cancels_all_pending_tasks(
        self,
        terminator: GracefulTerminator,
    ) -> None:
        """Test that _cancel_tasks cancels all pending tasks.
        
        Requirement 3.7: WHEN initiating graceful termination THEN the system
        SHALL cancel any pending API requests immediately.
        """
        # Create mock tasks that are not done
        tasks: List[asyncio.Task[Any]] = []
        
        async def long_running_task() -> None:
            await asyncio.sleep(100)
        
        for _ in range(3):
            task = asyncio.create_task(long_running_task())
            tasks.append(task)
        
        # Give tasks time to start
        await asyncio.sleep(0.01)
        
        # Verify tasks are running
        for task in tasks:
            assert not task.done()
        
        # Cancel tasks
        terminator._cancel_tasks(tasks)
        
        # Wait for cancellation to propagate
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all tasks are cancelled or done
        for task in tasks:
            assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_cancel_tasks_skips_already_done_tasks(
        self,
        terminator: GracefulTerminator,
    ) -> None:
        """Test that _cancel_tasks skips tasks that are already done."""
        async def quick_task() -> str:
            return "done"
        
        # Create a task that completes immediately
        task = asyncio.create_task(quick_task())
        await task  # Wait for completion
        
        # Should not raise when cancelling already-done task
        terminator._cancel_tasks([task])
        
        # Task should still be done (not cancelled)
        assert task.done()
        assert not task.cancelled()

    @pytest.mark.asyncio
    async def test_terminate_cancels_pending_tasks(
        self,
        terminator: GracefulTerminator,
        temp_output_dir: Path,
    ) -> None:
        """Test that terminate() cancels all pending tasks.
        
        Requirement 3.7: WHEN initiating graceful termination THEN the system
        SHALL cancel any pending API requests immediately.
        """
        cancelled_count = 0
        
        async def cancellable_task() -> None:
            nonlocal cancelled_count
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                cancelled_count += 1
                raise
        
        tasks = [asyncio.create_task(cancellable_task()) for _ in range(3)]
        
        # Give tasks time to start
        await asyncio.sleep(0.01)
        
        # Terminate
        report = await terminator.terminate(
            reason="test_timeout",
            pending_tasks=tasks,
            partial_results=[],
        )
        
        # All tasks should have been cancelled
        assert cancelled_count == 3
        assert report.reason == "test_timeout"


# --- Partial Result Persistence Tests (Requirements 3.8, 3.9) ---


class TestPartialResultPersistence:
    """Tests for partial result persistence during graceful termination."""

    @pytest.mark.asyncio
    async def test_flush_results_writes_partial_status(
        self,
        terminator: GracefulTerminator,
        sample_explorer_data: ExplorerData,
        temp_output_dir: Path,
    ) -> None:
        """Test that _flush_results writes results with 'partial' status.
        
        Requirement 3.8: WHEN initiating graceful termination THEN the system
        SHALL flush and write any in-progress results atomically to output
        with a "partial" status flag in metadata.
        """
        output_file, records = await terminator._flush_results(
            results=[sample_explorer_data],
            termination_reason="timeout",
        )
        
        assert output_file is not None
        assert records == 2  # 1 transaction + 1 holder
        assert output_file is not None
        
        # Read and verify the output file
        with open(output_file, "r") as f:
            data = json.load(f)
        
        assert len(data) == 1
        assert data[0]["status"] == "partial"
        assert data[0]["termination_reason"] == "timeout"

    @pytest.mark.asyncio
    async def test_flush_results_includes_all_data(
        self,
        terminator: GracefulTerminator,
        sample_explorer_data: ExplorerData,
        temp_output_dir: Path,
    ) -> None:
        """Test that _flush_results includes all collected data."""
        output_file, _ = await terminator._flush_results(
            results=[sample_explorer_data],
            termination_reason="resource_exhaustion",
        )
        assert output_file is not None
        
        with open(output_file, "r") as f:
            data = json.load(f)
        
        output = data[0]
        assert output["explorer_name"] == "etherscan"
        assert output["chain"] == "ethereum"
        assert len(output["transactions"]) == 1
        assert len(output["holders"]) == 1
        assert output["total_records"] == 2

    @pytest.mark.asyncio
    async def test_flush_results_handles_empty_results(
        self,
        terminator: GracefulTerminator,
    ) -> None:
        """Test that _flush_results handles empty results gracefully."""
        output_file, records = await terminator._flush_results(
            results=[],
            termination_reason="timeout",
        )
        
        assert output_file is None
        assert records == 0

    @pytest.mark.asyncio
    async def test_flush_results_handles_multiple_explorers(
        self,
        terminator: GracefulTerminator,
        sample_transaction: Transaction,
        sample_holder: Holder,
        temp_output_dir: Path,
    ) -> None:
        """Test that _flush_results handles data from multiple explorers."""
        explorer_data_1 = ExplorerData(
            explorer_name="etherscan",
            chain="ethereum",
            transactions=[sample_transaction],
            holders=[],
            errors=[],
            collection_time_seconds=30.0,
        )
        
        # Create a second explorer data with different chain
        tx2 = Transaction(
            transaction_hash="0x" + "b" * 64,
            block_number=12345679,
            timestamp=datetime(2024, 1, 15, 10, 31, 0),
            from_address="0x" + "4" * 40,
            to_address="0x" + "5" * 40,
            amount=Decimal("2000.00"),
            stablecoin="USDT",
            chain="bsc",
            activity_type=ActivityType.TRANSACTION,
            source_explorer="bscscan",
        )
        
        explorer_data_2 = ExplorerData(
            explorer_name="bscscan",
            chain="bsc",
            transactions=[tx2],
            holders=[sample_holder],
            errors=[],
            collection_time_seconds=25.0,
        )
        
        output_file, records = await terminator._flush_results(
            results=[explorer_data_1, explorer_data_2],
            termination_reason="timeout",
        )
        
        assert records == 3  # 2 transactions + 1 holder
        
        with open(output_file, "r") as f:
            data = json.load(f)
        
        assert len(data) == 2
        assert data[0]["explorer_name"] == "etherscan"
        assert data[1]["explorer_name"] == "bscscan"

    @pytest.mark.asyncio
    async def test_atomic_write_creates_file(
        self,
        terminator: GracefulTerminator,
        temp_output_dir: Path,
    ) -> None:
        """Test that _atomic_write creates the output file atomically."""
        test_path = temp_output_dir / "test_output.json"
        test_data = [{"key": "value"}]
        
        await terminator._atomic_write(test_path, test_data)
        
        assert test_path.exists()
        with open(test_path, "r") as f:
            loaded = json.load(f)
        assert loaded == test_data

    @pytest.mark.asyncio
    async def test_terminate_persists_partial_results(
        self,
        terminator: GracefulTerminator,
        sample_explorer_data: ExplorerData,
        temp_output_dir: Path,
    ) -> None:
        """Test that terminate() persists partial results.
        
        Requirement 3.8: WHEN initiating graceful termination THEN the system
        SHALL flush and write any in-progress results atomically to output
        with a "partial" status flag in metadata.
        """
        terminator.set_start_time(datetime.utcnow())
        
        report = await terminator.terminate(
            reason="overall_timeout",
            pending_tasks=[],
            partial_results=[sample_explorer_data],
        )
        
        assert report.records_collected == 2
        assert report.records_persisted == 2
        assert report.output_file is not None
        assert report.partial is True
        
        # Verify file was written
        assert Path(report.output_file).exists()

    @pytest.mark.asyncio
    async def test_terminate_respects_shutdown_timeout(
        self,
        temp_output_dir: Path,
    ) -> None:
        """Test that terminate() respects the shutdown timeout.
        
        Requirement 3.9: WHEN initiating graceful termination THEN the system
        SHALL wait for file writes to complete subject to a configurable
        shutdown timeout (default 30 seconds).
        """
        # Create terminator with short timeout
        terminator = GracefulTerminator(
            shutdown_timeout=0.1,  # Very short timeout
            output_directory=temp_output_dir,
        )
        
        # Even with short timeout, should complete normally for small data
        explorer_data = ExplorerData(
            explorer_name="test",
            chain="ethereum",
            transactions=[],
            holders=[],
            errors=[],
            collection_time_seconds=0.0,
        )
        
        report = await terminator.terminate(
            reason="test",
            pending_tasks=[],
            partial_results=[explorer_data],
        )
        
        # Should complete without error
        assert report.reason == "test"


# --- Termination Logging Tests (Requirement 3.10) ---


class TestTerminationLogging:
    """Tests for termination logging during graceful termination."""

    @pytest.mark.asyncio
    async def test_terminate_logs_structured_entry(
        self,
        terminator: GracefulTerminator,
        sample_explorer_data: ExplorerData,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that terminate() logs a structured entry.
        
        Requirement 3.10: WHEN graceful termination completes THEN the system
        SHALL record a structured log entry with termination reason, timestamp,
        records collected, and summary of persisted data.
        """
        terminator.set_start_time(datetime.utcnow())
        
        with caplog.at_level(logging.INFO):
            await terminator.terminate(
                reason="resource_limit_exceeded",
                pending_tasks=[],
                partial_results=[sample_explorer_data],
            )
        
        # Check that termination was logged
        assert any(
            "Graceful termination complete" in record.message
            for record in caplog.records
        )
        
        # Check that initiation was logged
        assert any(
            "Initiating graceful termination" in record.message
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_terminate_logs_task_cancellation_count(
        self,
        terminator: GracefulTerminator,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that terminate() logs the number of tasks being cancelled."""
        async def dummy_task() -> None:
            await asyncio.sleep(100)
        
        tasks = [asyncio.create_task(dummy_task()) for _ in range(5)]
        await asyncio.sleep(0.01)
        
        with caplog.at_level(logging.INFO):
            await terminator.terminate(
                reason="timeout",
                pending_tasks=tasks,
                partial_results=[],
            )
        
        # Check that task count was logged
        assert any(
            "Cancelling 5 pending tasks" in record.message
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_termination_report_contains_all_fields(
        self,
        terminator: GracefulTerminator,
        sample_explorer_data: ExplorerData,
    ) -> None:
        """Test that TerminationReport contains all required fields.
        
        Requirement 3.10: WHEN graceful termination completes THEN the system
        SHALL record a structured log entry with termination reason, timestamp,
        records collected, and summary of persisted data.
        """
        start_time = datetime.utcnow()
        terminator.set_start_time(start_time)
        
        report = await terminator.terminate(
            reason="test_reason",
            pending_tasks=[],
            partial_results=[sample_explorer_data],
        )
        
        # Verify all fields are present
        assert report.reason == "test_reason"
        assert report.timestamp is not None
        assert report.records_collected == 2
        assert report.records_persisted == 2
        assert report.output_file is not None
        assert report.partial is True
        assert report.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_terminate_logs_warning_on_slow_cancellation(
        self,
        temp_output_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that terminate() logs warning when tasks don't cancel."""
        # Create terminator with very short timeout
        terminator = GracefulTerminator(
            shutdown_timeout=0.01,
            output_directory=temp_output_dir,
        )
        
        async def stubborn_task() -> None:
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                # Simulate slow cleanup
                await asyncio.sleep(1)
                raise
        
        task = asyncio.create_task(stubborn_task())
        await asyncio.sleep(0.01)
        
        with caplog.at_level(logging.WARNING):
            await terminator.terminate(
                reason="timeout",
                pending_tasks=[task],
                partial_results=[],
            )
        
        # Should log warning about slow cancellation
        assert any(
            "did not cancel within timeout" in record.message
            for record in caplog.records
        )


# --- ExplorerDataOutput Model Tests ---


class TestExplorerDataOutput:
    """Tests for the ExplorerDataOutput model."""

    def test_explorer_data_output_defaults(self) -> None:
        """Test ExplorerDataOutput default values."""
        output = ExplorerDataOutput(
            explorer_name="test",
            chain="ethereum",
            transactions=[],
            holders=[],
            errors=[],
            collection_time_seconds=0.0,
            total_records=0,
            timestamp="2024-01-15T10:30:00Z",
            run_id="abc123",
        )
        
        assert output.status == "complete"
        assert output.termination_reason is None

    def test_explorer_data_output_partial_status(self) -> None:
        """Test ExplorerDataOutput with partial status."""
        output = ExplorerDataOutput(
            explorer_name="test",
            chain="ethereum",
            transactions=[],
            holders=[],
            errors=[],
            collection_time_seconds=0.0,
            total_records=0,
            status="partial",
            termination_reason="timeout",
            timestamp="2024-01-15T10:30:00Z",
            run_id="abc123",
        )
        
        assert output.status == "partial"
        assert output.termination_reason == "timeout"


# --- TerminationReport Model Tests ---


class TestTerminationReport:
    """Tests for the TerminationReport model."""

    def test_termination_report_creation(self) -> None:
        """Test TerminationReport creation."""
        report = TerminationReport(
            reason="timeout",
            timestamp=datetime.utcnow(),
            records_collected=100,
            records_persisted=95,
            output_file="/path/to/output.json",
            partial=True,
            duration_seconds=45.5,
        )
        
        assert report.reason == "timeout"
        assert report.records_collected == 100
        assert report.records_persisted == 95
        assert report.partial is True

    def test_termination_report_optional_output_file(self) -> None:
        """Test TerminationReport with no output file."""
        report = TerminationReport(
            reason="no_data",
            timestamp=datetime.utcnow(),
            records_collected=0,
            records_persisted=0,
            output_file=None,
            partial=True,
            duration_seconds=0.0,
        )
        
        assert report.output_file is None


# --- Integration Tests ---


class TestGracefulTerminatorIntegration:
    """Integration tests for GracefulTerminator."""

    @pytest.mark.asyncio
    async def test_full_termination_flow(
        self,
        temp_output_dir: Path,
        sample_explorer_data: ExplorerData,
    ) -> None:
        """Test the complete termination flow end-to-end."""
        terminator = GracefulTerminator(
            shutdown_timeout=5.0,
            output_directory=temp_output_dir,
        )
        terminator.set_run_id("test-run-123")
        terminator.set_start_time(datetime.utcnow())
        
        # Create some pending tasks
        async def api_request() -> None:
            await asyncio.sleep(100)
        
        tasks = [asyncio.create_task(api_request()) for _ in range(2)]
        await asyncio.sleep(0.01)
        
        # Execute termination
        report = await terminator.terminate(
            reason="overall_timeout_exceeded",
            pending_tasks=tasks,
            partial_results=[sample_explorer_data],
        )
        
        # Verify report
        assert report.reason == "overall_timeout_exceeded"
        assert report.records_collected == 2
        assert report.records_persisted == 2
        assert report.partial is True
        assert report.output_file is not None
        
        # Verify output file
        with open(report.output_file, "r") as f:
            data = json.load(f)
        
        assert len(data) == 1
        assert data[0]["status"] == "partial"
        assert data[0]["run_id"] == "test-run-123"
        
        # Verify tasks were cancelled
        for task in tasks:
            assert task.done()

    @pytest.mark.asyncio
    async def test_termination_with_errors_in_explorer_data(
        self,
        terminator: GracefulTerminator,
        sample_transaction: Transaction,
    ) -> None:
        """Test termination with explorer data containing errors."""
        explorer_data = ExplorerData(
            explorer_name="etherscan",
            chain="ethereum",
            transactions=[sample_transaction],
            holders=[],
            errors=["Rate limit exceeded", "Partial data collected"],
            collection_time_seconds=30.0,
        )
        
        report = await terminator.terminate(
            reason="rate_limit",
            pending_tasks=[],
            partial_results=[explorer_data],
        )
        
        # Verify errors are preserved in output
        assert report.output_file is not None
        with open(report.output_file, "r") as f:
            data = json.load(f)
        
        assert len(data[0]["errors"]) == 2
        assert "Rate limit exceeded" in data[0]["errors"]
