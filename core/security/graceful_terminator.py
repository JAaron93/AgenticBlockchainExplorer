"""Graceful termination handling for agent runs.

This module provides the GracefulTerminator class that handles graceful
shutdown on timeout or resource exhaustion, ensuring partial results
are persisted before termination.

Requirements: 3.7, 3.8, 3.9, 3.10
"""

import asyncio
import json
import logging
import tempfile
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from collectors.models import ExplorerData


logger = logging.getLogger(__name__)


class ExplorerDataOutput(BaseModel):
    """JSON-serializable output format for ExplorerData with status.

    This model extends the basic ExplorerData with status information
    for tracking partial results during graceful termination.
    """

    explorer_name: str
    chain: str
    transactions: List[Dict[str, Any]]
    holders: List[Dict[str, Any]]
    errors: List[str]
    collection_time_seconds: float
    total_records: int
    # Security hardening additions
    status: Literal["complete", "partial"] = "complete"
    termination_reason: Optional[str] = None
    timestamp: str  # ISO-8601 format
    run_id: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "explorer_name": "etherscan",
                "chain": "ethereum",
                "transactions": [],
                "holders": [],
                "errors": [],
                "collection_time_seconds": 45.2,
                "total_records": 100,
                "status": "partial",
                "termination_reason": "timeout",
                "timestamp": "2024-01-15T10:30:00Z",
                "run_id": "abc123",
            }
        }
    }


class TerminationReport(BaseModel):
    """Report generated on graceful termination.

    Contains details about the termination including reason, timing,
    and summary of data collected and persisted.
    """

    reason: str
    timestamp: datetime
    records_collected: int
    records_persisted: int
    output_file: Optional[str] = None
    partial: bool = True
    duration_seconds: float


class GracefulTerminator:
    """Handles graceful termination of agent runs.

    This class manages the graceful shutdown sequence when an agent run
    needs to terminate due to timeout, resource exhaustion, or other
    conditions. It ensures:
    - Pending tasks are cancelled cleanly
    - Partial results are persisted with appropriate status flags
    - Termination is logged with structured information

    Requirements: 3.7, 3.8, 3.9, 3.10
    """

    def __init__(
        self,
        shutdown_timeout: float = 30.0,
        output_directory: Optional[Path] = None,
    ):
        """Initialize with shutdown timeout.

        Args:
            shutdown_timeout: Maximum time to wait for shutdown operations
                            (file writes, task cancellation) in seconds.
            output_directory: Directory for output files. If None, uses
                            current working directory.
        """
        self._shutdown_timeout = shutdown_timeout
        self._output_directory = output_directory or Path("./output")
        self._run_id = str(uuid.uuid4())[:8]
        self._start_time: Optional[datetime] = None

    def set_run_id(self, run_id: str) -> None:
        """Set the run ID for output file naming.

        Args:
            run_id: Unique identifier for this run.
        """
        self._run_id = run_id

    def set_start_time(self, start_time: datetime) -> None:
        """Set the start time for duration calculation.

        Args:
            start_time: When the run started.
        """
        self._start_time = start_time

    async def terminate(
        self,
        reason: str,
        pending_tasks: List[asyncio.Task[Any]],
        partial_results: List[ExplorerData],
    ) -> TerminationReport:
        """Execute graceful termination sequence.

        This method:
        1. Cancels all pending tasks immediately
        2. Waits for cancellation to complete (with timeout)
        3. Flushes partial results to output with 'partial' status
        4. Logs termination details

        Args:
            reason: Human-readable reason for termination.
            pending_tasks: List of asyncio tasks to cancel.
            partial_results: List of ExplorerData with collected data.

        Returns:
            TerminationReport with details about the termination.

        Requirements: 3.7, 3.8, 3.9, 3.10
        """
        termination_start = datetime.utcnow()

        # Calculate duration
        if self._start_time:
            duration = (termination_start - self._start_time).total_seconds()
        else:
            duration = 0.0

        # Step 1: Cancel pending tasks (Requirement 3.7)
        logger.info(
            f"Initiating graceful termination: {reason}. "
            f"Cancelling {len(pending_tasks)} pending tasks."
        )
        self._cancel_tasks(pending_tasks)

        # Step 2: Wait for cancellation with timeout
        if pending_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending_tasks, return_exceptions=True),
                    timeout=min(self._shutdown_timeout / 2, 10.0),
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Some tasks did not cancel within timeout, proceeding."
                )

        # Step 3: Flush results (Requirements 3.8, 3.9)
        total_collected = sum(r.total_records for r in partial_results)
        output_file = None
        records_persisted = 0

        if partial_results:
            try:
                output_file, records_persisted = await asyncio.wait_for(
                    self._flush_results(partial_results, reason),
                    timeout=self._shutdown_timeout,
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"Failed to flush results within {self._shutdown_timeout}s"
                )
            except Exception as e:
                logger.error(f"Error flushing results: {e}")

        # Step 4: Log termination (Requirement 3.10)
        report = TerminationReport(
            reason=reason,
            timestamp=termination_start,
            records_collected=total_collected,
            records_persisted=records_persisted,
            output_file=output_file,
            partial=True,
            duration_seconds=duration,
        )

        logger.info(
            "Graceful termination complete",
            extra={
                "termination_reason": reason,
                "timestamp": termination_start.isoformat(),
                "records_collected": total_collected,
                "records_persisted": records_persisted,
                "output_file": output_file,
                "duration_seconds": duration,
            },
        )

        return report

    def _cancel_tasks(self, tasks: List[asyncio.Task[Any]]) -> None:
        """Cancel all pending tasks.

        Args:
            tasks: List of asyncio tasks to cancel.

        Requirements: 3.7
        """
        for task in tasks:
            if not task.done():
                task.cancel()

    async def _flush_results(
        self,
        results: List[ExplorerData],
        termination_reason: str,
    ) -> tuple[Optional[str], int]:
        """Write partial results with 'partial' status flag.

        Uses atomic write operations to prevent partial file corruption.

        Args:
            results: List of ExplorerData to persist.
            termination_reason: Reason for termination to include in output.

        Returns:
            Tuple of (output file path, number of records persisted).

        Requirements: 3.8, 3.9
        """
        if not results:
            return None, 0

        # Ensure output directory exists
        self._output_directory.mkdir(parents=True, exist_ok=True)

        # Convert results to output format
        timestamp = datetime.utcnow().isoformat() + "Z"
        output_data: List[Dict[str, Any]] = []
        total_records = 0

        for explorer_data in results:
            # Convert transactions and holders to dicts
            transactions = [
                t.to_dict() for t in explorer_data.transactions
            ]
            holders = [h.to_dict() for h in explorer_data.holders]

            output = ExplorerDataOutput(
                explorer_name=explorer_data.explorer_name,
                chain=explorer_data.chain,
                transactions=transactions,
                holders=holders,
                errors=explorer_data.errors,
                collection_time_seconds=explorer_data.collection_time_seconds,
                total_records=explorer_data.total_records,
                status="partial",
                termination_reason=termination_reason,
                timestamp=timestamp,
                run_id=self._run_id,
            )
            output_data.append(output.model_dump())
            total_records += explorer_data.total_records

        # Generate output filename
        output_filename = f"partial_{self._run_id}_{timestamp[:10]}.json"
        output_path = self._output_directory / output_filename

        # Atomic write using temp file and rename
        await self._atomic_write(output_path, output_data)

        return str(output_path), total_records

    async def _atomic_write(
        self,
        path: Path,
        data: List[Dict[str, Any]],
    ) -> None:
        """Write data atomically using temp file and rename.

        Args:
            path: Target file path.
            data: Data to write as JSON.
        """
        # Create temp file in same directory for atomic rename
        temp_fd, temp_path = tempfile.mkstemp(
            dir=path.parent,
            prefix=".tmp_",
            suffix=".json",
        )

        try:
            # Write to temp file
            content = json.dumps(data, indent=2, default=str)
            with open(temp_fd, "w", encoding="utf-8") as f:
                f.write(content)

            # Atomic rename
            Path(temp_path).rename(path)

        except Exception:
            # Clean up temp file on error
            try:
                Path(temp_path).unlink()
            except OSError:
                pass
            raise
