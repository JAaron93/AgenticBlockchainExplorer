"""Agent orchestrator for coordinating data collection.

Orchestrates data collection from multiple blockchain explorers,
coordinates classification, aggregation, and export of results.
Integrates security components for timeout management and graceful
termination.

Requirements: 3.7, 3.8, 3.9, 3.10, 6.1, 6.2, 6.3, 6.6
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Optional

from config.models import Config, ExplorerConfig, TimeoutConfig
from collectors.base import ExplorerCollector
from collectors.models import ExplorerData
from collectors.etherscan import EtherscanCollector
from collectors.bscscan import BscscanCollector
from collectors.polygonscan import PolygonscanCollector
from collectors.classifier import ActivityClassifier
from collectors.aggregator import DataAggregator, AggregatedData
from collectors.exporter import JSONExporter
from core.db_manager import DatabaseManager


# Use standard logging - will be configured by core.logging when app starts
logger = logging.getLogger(__name__)

# Lazy imports for security components to avoid circular imports
_timeout_manager_class = None
_graceful_terminator_class = None


def _get_timeout_manager_class():
    """Get TimeoutManager class (lazy import)."""
    global _timeout_manager_class
    if _timeout_manager_class is None:
        try:
            from core.security.timeout_manager import TimeoutManager
            _timeout_manager_class = TimeoutManager
        except ImportError as e:
            logger.warning(f"TimeoutManager not available: {e}")
    return _timeout_manager_class


def _get_graceful_terminator_class():
    """Get GracefulTerminator class (lazy import)."""
    global _graceful_terminator_class
    if _graceful_terminator_class is None:
        try:
            from core.security.graceful_terminator import GracefulTerminator
            _graceful_terminator_class = GracefulTerminator
        except ImportError as e:
            logger.warning(f"GracefulTerminator not available: {e}")
    return _graceful_terminator_class


def _set_run_id(run_id: Optional[str]) -> None:
    """Set run_id in logging context if core.logging is available."""
    try:
        from core.logging import set_run_id
        set_run_id(run_id)
    except ImportError:
        pass  # core.logging not available, skip


# Type alias for collector classes
CollectorType = (
    type[EtherscanCollector]
    | type[BscscanCollector]
    | type[PolygonscanCollector]
)

# Mapping of chain names to collector classes
COLLECTOR_CLASSES: dict[str, CollectorType] = {
    "ethereum": EtherscanCollector,
    "bsc": BscscanCollector,
    "polygon": PolygonscanCollector,
}


@dataclass
class CollectionReport:
    """Report of data collection results."""

    run_id: str
    total_records: int
    records_by_source: dict[str, int] = field(default_factory=dict)
    explorers_queried: list[str] = field(default_factory=list)
    explorers_failed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    output_file_path: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if collection was at least partially successful."""
        return self.total_records > 0

    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "run_id": self.run_id,
            "total_records": self.total_records,
            "records_by_source": self.records_by_source,
            "explorers_queried": self.explorers_queried,
            "explorers_failed": self.explorers_failed,
            "errors": self.errors,
            "duration_seconds": self.duration_seconds,
            "output_file_path": self.output_file_path,
            "success": self.success,
        }


@dataclass
class RunConfig:
    """Run-specific configuration overrides."""

    max_records_per_explorer: Optional[int] = None
    explorers: Optional[list[str]] = None
    stablecoins: Optional[list[str]] = None


class AgentOrchestrator:
    """Orchestrates data collection from blockchain explorers.

    Coordinates the collection of stablecoin data from multiple blockchain
    explorers, classifies activities, aggregates results, and exports
    to JSON and database.
    
    Integrates security components:
    - TimeoutManager for enforcing collection timeouts
    - GracefulTerminator for handling timeout/resource exhaustion
    
    Requirements: 3.7, 3.8, 3.9, 3.10, 6.1, 6.2, 6.3, 6.6
    """

    def __init__(
        self,
        config: Config,
        run_id: str,
        db_manager: Optional[DatabaseManager] = None,
        user_id: Optional[str] = None,
        run_config: Optional[RunConfig] = None,
    ):
        """Initialize the orchestrator.

        Args:
            config: Application configuration.
            run_id: Unique identifier for this run.
            db_manager: Database manager for persistence.
            user_id: Optional user ID who initiated the run.
            run_config: Optional run-specific configuration overrides.
        """
        self._config = config
        self._run_id = run_id
        self._db_manager = db_manager
        self._user_id = user_id
        self._run_config = run_config or RunConfig()
        self._collectors: list[ExplorerCollector] = []
        self._classifier = ActivityClassifier()
        self._aggregator = DataAggregator()
        self._exporter = JSONExporter(
            db_manager=db_manager,
            output_directory=config.output.directory
        )
        
        # Security components (initialized lazily)
        self._timeout_manager: Optional[Any] = None
        self._graceful_terminator: Optional[Any] = None
        self._start_time: Optional[datetime] = None
        self._pending_tasks: List[asyncio.Task[Any]] = []
        self._partial_results: List[ExplorerData] = []

    @property
    def run_id(self) -> str:
        """Get the run ID."""
        return self._run_id
    
    def _initialize_security_components(self) -> None:
        """Initialize security components for timeout and graceful termination.
        
        Requirements: 3.7, 3.8, 3.9, 3.10, 6.1, 6.2, 6.3, 6.6
        """
        # Calculate number of collections (stablecoins × explorers)
        num_stablecoins = len(self._config.stablecoins)
        num_explorers = len(self._config.explorers)
        num_collections = num_stablecoins * num_explorers
        
        # Initialize TimeoutManager
        TimeoutManagerClass = _get_timeout_manager_class()
        if TimeoutManagerClass:
            try:
                # Get timeout config from security config if available
                timeout_config = getattr(
                    getattr(self._config, 'security', None),
                    'timeouts',
                    None
                )
                if timeout_config is None:
                    # Use default timeout config
                    timeout_config = TimeoutConfig()
                
                self._timeout_manager = TimeoutManagerClass(
                    config=timeout_config,
                    num_collections=max(num_collections, 1),
                )
                logger.info(
                    f"TimeoutManager initialized: "
                    f"overall={self._timeout_manager.overall_timeout}s, "
                    f"per_collection={self._timeout_manager.per_collection_timeout}s",
                    extra={"run_id": self._run_id}
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize TimeoutManager: {e}",
                    extra={"run_id": self._run_id}
                )
        
        # Initialize GracefulTerminator
        GracefulTerminatorClass = _get_graceful_terminator_class()
        if GracefulTerminatorClass:
            try:
                shutdown_timeout = 30.0
                if self._timeout_manager:
                    shutdown_timeout = self._timeout_manager.shutdown_timeout
                
                self._graceful_terminator = GracefulTerminatorClass(
                    shutdown_timeout=shutdown_timeout,
                    output_directory=self._config.output.directory,
                )
                self._graceful_terminator.set_run_id(self._run_id)
                logger.info(
                    f"GracefulTerminator initialized: shutdown_timeout={shutdown_timeout}s",
                    extra={"run_id": self._run_id}
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize GracefulTerminator: {e}",
                    extra={"run_id": self._run_id}
                )

    def _create_collector(
        self, explorer_config: ExplorerConfig
    ) -> Optional[ExplorerCollector]:
        """Create a collector instance for the given explorer config.

        Args:
            explorer_config: Configuration for the explorer.

        Returns:
            Collector instance or None if chain is not supported.
        """
        collector_class = COLLECTOR_CLASSES.get(explorer_config.chain)
        if collector_class is None:
            logger.warning(
                f"No collector available for chain '{explorer_config.chain}'",
                extra={"run_id": self._run_id, "chain": explorer_config.chain}
            )
            return None

        collector = collector_class(
            config=explorer_config,
            retry_config=self._config.retry
        )
        return collector

    def _initialize_collectors(self) -> list[ExplorerCollector]:
        """Initialize collector instances from configuration.

        Respects run_config.explorers filter if specified.

        Returns:
            List of initialized collectors.
        """
        collectors = []
        allowed_explorers = self._run_config.explorers

        for explorer_config in self._config.explorers:
            # Filter by explorer names if specified in run config
            if allowed_explorers is not None:
                if explorer_config.name not in allowed_explorers:
                    logger.debug(
                        f"Skipping {explorer_config.name} (not in run config)",
                        extra={"run_id": self._run_id}
                    )
                    continue

            collector = self._create_collector(explorer_config)
            if collector:
                collectors.append(collector)
                logger.info(
                    f"Initialized collector for {explorer_config.name}",
                    extra={
                        "run_id": self._run_id,
                        "explorer": explorer_config.name,
                        "chain": explorer_config.chain,
                    }
                )
        return collectors

    def _get_stablecoin_addresses(self, chain: str) -> dict[str, str]:
        """Get stablecoin contract addresses for a chain.

        Respects run_config.stablecoins filter if specified.

        Args:
            chain: The blockchain chain name.

        Returns:
            Dict mapping stablecoin symbol to contract address.
        """
        addresses = {}
        allowed_stablecoins = self._run_config.stablecoins

        for symbol, config in self._config.stablecoins.items():
            # Filter by stablecoin symbols if specified in run config
            if allowed_stablecoins is not None:
                if symbol not in allowed_stablecoins:
                    continue

            address = getattr(config, chain, None)
            if address:
                addresses[symbol] = address
        return addresses

    async def update_progress(
        self,
        progress: float,
        message: str
    ) -> None:
        """Update run progress in database.

        Args:
            progress: Progress percentage (0.0 to 1.0).
            message: Progress message.
        """
        if self._db_manager is None:
            return

        try:
            await self._db_manager.update_run_progress(
                run_id=self._run_id,
                progress=progress,
                message=message
            )
            logger.debug(
                f"Updated progress: {progress:.1%} - {message}",
                extra={"run_id": self._run_id, "progress": progress}
            )
        except Exception as e:
            logger.warning(
                f"Failed to update progress: {e}",
                extra={"run_id": self._run_id, "error": str(e)}
            )

    async def _collect_from_explorer(
        self,
        collector: ExplorerCollector
    ) -> ExplorerData:
        """Collect data from a single explorer with timeout enforcement.

        Implements graceful error handling to return partial results
        when collection partially fails. The explorer is considered
        successful if any data was collected, even with errors.
        
        Uses TimeoutManager for per-collection timeout enforcement.

        Args:
            collector: The collector to use.

        Returns:
            ExplorerData with collected transactions and holders,
            including any errors encountered.
            
        Requirements: 6.1, 6.2, 6.6
        """
        stablecoins = self._get_stablecoin_addresses(collector.chain)
        # Use run config override if specified, otherwise use app config
        max_records = (
            self._run_config.max_records_per_explorer
            or self._config.output.max_records_per_explorer
        )

        logger.info(
            f"Starting collection from {collector.name}",
            extra={
                "run_id": self._run_id,
                "explorer": collector.name,
                "chain": collector.chain,
                "stablecoins": list(stablecoins.keys()),
                "max_records": max_records,
            }
        )

        async def do_collection() -> ExplorerData:
            """Inner collection function to wrap with timeout."""
            async with collector:
                return await collector.collect_all(
                    stablecoins=stablecoins,
                    max_records=max_records,
                    run_id=self._run_id
                )

        try:
            # Use TimeoutManager if available (Requirement 6.1, 6.2)
            if self._timeout_manager:
                result = await self._timeout_manager.run_with_timeout(
                    do_collection(),
                    collection_name=collector.name,
                )
            else:
                result = await do_collection()
            
            # Store partial result for graceful termination
            self._partial_results.append(result)
            
            # Log result summary
            if result.errors:
                logger.warning(
                    f"Collection from {collector.name} completed with {len(result.errors)} errors",
                    extra={
                        "run_id": self._run_id,
                        "explorer": collector.name,
                        "total_records": result.total_records,
                        "error_count": len(result.errors),
                        "partial_success": result.total_records > 0,
                    }
                )
            else:
                logger.info(
                    f"Collection from {collector.name} completed successfully",
                    extra={
                        "run_id": self._run_id,
                        "explorer": collector.name,
                        "total_records": result.total_records,
                    }
                )
            
            return result
            
        except asyncio.TimeoutError as e:
            error_msg = f"Collection timed out for {collector.name}"
            logger.error(
                error_msg,
                extra={
                    "run_id": self._run_id,
                    "explorer": collector.name,
                    "error_type": "timeout",
                }
            )
            result = ExplorerData(
                explorer_name=collector.name,
                chain=collector.chain,
                errors=[error_msg]
            )
            self._partial_results.append(result)
            return result
        except Exception as e:
            error_msg = f"Collection failed for {collector.name}: {type(e).__name__}: {e}"
            logger.error(
                error_msg,
                extra={
                    "run_id": self._run_id,
                    "explorer": collector.name,
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
                exc_info=True
            )
            result = ExplorerData(
                explorer_name=collector.name,
                chain=collector.chain,
                errors=[error_msg]
            )
            self._partial_results.append(result)
            return result

    async def collect_from_all_explorers(self) -> list[ExplorerData]:
        """Collect data from all configured explorers in parallel.

        Uses asyncio.gather to execute collectors concurrently.
        Handles partial failures by continuing with available data.

        Returns:
            List of ExplorerData from each collector.
        """
        self._collectors = self._initialize_collectors()

        if not self._collectors:
            logger.error(
                "No collectors initialized",
                extra={"run_id": self._run_id}
            )
            return []

        await self.update_progress(0.1, "Starting data collection")

        # Execute all collectors in parallel
        tasks = []
        for collector in self._collectors:
            # Create task and track it (Requirement 6.3)
            task = asyncio.create_task(self._collect_from_explorer(collector))
            self._pending_tasks.append(task)
            
            # Remove from pending tasks when done
            # We use a lambda to capturing the task object safely 
            # or simply use the method since the callback receives the task
            task.add_done_callback(lambda t: self._pending_tasks.remove(t) if t in self._pending_tasks else None)
            
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, handling any exceptions
        explorer_data: list[ExplorerData] = []
        for i, result in enumerate(results):
            collector = self._collectors[i]
            if isinstance(result, BaseException):
                logger.error(
                    f"Collector {collector.name} raised exception: {result}",
                    extra={
                        "run_id": self._run_id,
                        "explorer": collector.name,
                        "error": str(result),
                    }
                )
                explorer_data.append(ExplorerData(
                    explorer_name=collector.name,
                    chain=collector.chain,
                    errors=[f"Exception: {result}"]
                ))
            elif isinstance(result, ExplorerData):
                explorer_data.append(result)

        await self.update_progress(0.5, "Data collection complete")
        return explorer_data

    def _classify_data(self, data: AggregatedData) -> AggregatedData:
        """Classify transactions and holders in the aggregated data.

        Args:
            data: Aggregated data to classify.

        Returns:
            Data with updated activity classifications.
        """
        logger.info(
            "Classifying transaction activities",
            extra={
                "run_id": self._run_id,
                "transactions": len(data.transactions),
                "holders": len(data.holders),
            }
        )

        # Classify each transaction
        for tx in data.transactions:
            tx.activity_type = self._classifier.classify_transaction(tx)

        # Classify holders as store of value
        for holder in data.holders:
            self._classifier.classify_holder(holder, data.transactions)

        return data

    def generate_report(
        self,
        results: list[ExplorerData],
        aggregated: AggregatedData,
        duration: float,
        output_path: Optional[str] = None
    ) -> CollectionReport:
        """Generate a collection report.

        Args:
            results: Raw results from each explorer.
            aggregated: Aggregated and processed data.
            duration: Total duration in seconds.
            output_path: Path to output file if exported.

        Returns:
            CollectionReport with summary statistics.
        """
        report = CollectionReport(
            run_id=self._run_id,
            total_records=aggregated.total_records,
            duration_seconds=duration,
            output_file_path=output_path,
        )

        for explorer_data in results:
            report.explorers_queried.append(explorer_data.explorer_name)
            report.records_by_source[explorer_data.explorer_name] = (
                explorer_data.total_records
            )
            if explorer_data.errors:
                report.errors.extend(explorer_data.errors)
                if not explorer_data.success:
                    report.explorers_failed.append(explorer_data.explorer_name)

        logger.info(
            f"Generated report: {report.total_records} total records",
            extra={
                "run_id": self._run_id,
                "total_records": report.total_records,
                "explorers_queried": report.explorers_queried,
                "explorers_failed": report.explorers_failed,
            }
        )

        return report

    async def run(self) -> CollectionReport:
        """Execute the full data collection pipeline with timeout management.

        Orchestrates:
        1. Initialize security components (TimeoutManager, GracefulTerminator)
        2. Collection from all explorers in parallel with timeout enforcement
        3. Classification of activities
        4. Aggregation and deduplication
        5. Export to JSON and database
        6. Graceful termination on timeout or resource exhaustion

        Returns:
            CollectionReport with results summary.
            
        Requirements: 3.7, 3.8, 3.9, 3.10, 6.1, 6.2, 6.3, 6.6
        """
        start_time = time.time()
        self._start_time = datetime.utcnow()
        self._partial_results = []
        
        # Set run_id in logging context for correlation
        _set_run_id(self._run_id)
        
        # Initialize security components
        self._initialize_security_components()
        
        # Set start time for graceful terminator
        if self._graceful_terminator:
            self._graceful_terminator.set_start_time(self._start_time)
        
        # Start timeout manager
        if self._timeout_manager:
            self._timeout_manager.start()

        logger.info(
            f"Starting agent run {self._run_id}",
            extra={
                "run_id": self._run_id,
                "user_id": self._user_id,
                "explorers": [e.name for e in self._config.explorers],
                "timeout_enabled": self._timeout_manager is not None,
            }
        )

        # Update status to running
        if self._db_manager:
            await self._db_manager.update_run_status(
                run_id=self._run_id,
                status="running"
            )

        try:
            # Step 1: Collect from all explorers
            explorer_results = await self.collect_from_all_explorers()
            
            # Check if we should terminate (Requirement 6.3)
            if self._timeout_manager and self._timeout_manager.should_terminate():
                logger.warning(
                    f"Overall timeout approaching, initiating graceful termination",
                    extra={
                        "run_id": self._run_id,
                        "time_remaining": self._timeout_manager.time_remaining(),
                    }
                )
                return await self._handle_graceful_termination(
                    reason="overall_timeout_approaching",
                    explorer_results=explorer_results,
                    start_time=start_time,
                )

            if not explorer_results:
                raise RuntimeError("No data collected from any explorer")

            # Step 2: Aggregate results
            await self.update_progress(0.6, "Aggregating data")
            aggregated = self._aggregator.aggregate(explorer_results)

            # Step 3: Classify activities
            await self.update_progress(0.7, "Classifying activities")
            aggregated = self._classify_data(aggregated)

            # Step 4: Export to JSON and database
            await self.update_progress(0.8, "Exporting results")
            output_path, _ = await self._exporter.export_and_save(
                data=aggregated,
                run_id=self._run_id,
                user_id=self._user_id
            )

            # Generate report
            duration = time.time() - start_time
            report = self.generate_report(
                results=explorer_results,
                aggregated=aggregated,
                duration=duration,
                output_path=output_path
            )

            # Update status to completed
            await self.update_progress(1.0, "Complete")
            if self._db_manager:
                await self._db_manager.update_run_status(
                    run_id=self._run_id,
                    status="completed"
                )

            logger.info(
                f"Agent run {self._run_id} completed successfully",
                extra={
                    "run_id": self._run_id,
                    "total_records": report.total_records,
                    "duration_seconds": duration,
                }
            )

            return report

        except Exception as e:
            duration = time.time() - start_time
            
            # Check if this is a timeout-related error
            error_type = type(e).__name__
            is_timeout = "Timeout" in error_type or "timeout" in str(e).lower()
            
            if is_timeout and self._graceful_terminator and self._partial_results:
                # Handle graceful termination (Requirements 3.7, 3.8, 3.9, 3.10)
                logger.warning(
                    f"Timeout during agent run {self._run_id}, initiating graceful termination",
                    extra={
                        "run_id": self._run_id,
                        "error": str(e),
                        "partial_results_count": len(self._partial_results),
                    }
                )
                return await self._handle_graceful_termination(
                    reason=f"timeout: {e}",
                    explorer_results=self._partial_results,
                    start_time=start_time,
                )
            
            logger.error(
                f"Agent run {self._run_id} failed: {e}",
                extra={
                    "run_id": self._run_id,
                    "error": str(e),
                    "error_type": error_type,
                    "duration_seconds": duration,
                },
                exc_info=True
            )

            # Update status to failed
            if self._db_manager:
                await self._db_manager.update_run_status(
                    run_id=self._run_id,
                    status="failed",
                    error_message=str(e)
                )

            # Return a failure report
            return CollectionReport(
                run_id=self._run_id,
                total_records=0,
                errors=[str(e)],
                duration_seconds=duration,
            )
        
        finally:
            # Clear run_id from logging context
            _set_run_id(None)
    
    async def _handle_graceful_termination(
        self,
        reason: str,
        explorer_results: List[ExplorerData],
        start_time: float,
    ) -> CollectionReport:
        """Handle graceful termination with partial result persistence.
        
        Args:
            reason: Reason for termination.
            explorer_results: Collected results so far.
            start_time: When the run started (time.time()).
            
        Returns:
            CollectionReport with partial results.
            
        Requirements: 3.7, 3.8, 3.9, 3.10
        """
        duration = time.time() - start_time
        
        if self._graceful_terminator:
            # Use GracefulTerminator to persist partial results
            termination_report = await self._graceful_terminator.terminate(
                reason=reason,
                pending_tasks=self._pending_tasks,
                partial_results=explorer_results,
            )
            
            logger.info(
                f"Graceful termination completed for run {self._run_id}",
                extra={
                    "run_id": self._run_id,
                    "reason": reason,
                    "records_collected": termination_report.records_collected,
                    "records_persisted": termination_report.records_persisted,
                    "output_file": termination_report.output_file,
                }
            )
            
            # Update status to partial
            if self._db_manager:
                await self._db_manager.update_run_status(
                    run_id=self._run_id,
                    status="partial",
                    error_message=f"Graceful termination: {reason}"
                )
            
            return CollectionReport(
                run_id=self._run_id,
                total_records=termination_report.records_persisted,
                errors=[f"Graceful termination: {reason}"],
                duration_seconds=duration,
                output_file_path=termination_report.output_file,
            )
        else:
            # Fallback without GracefulTerminator
            total_records = sum(r.total_records for r in explorer_results)
            
            if self._db_manager:
                await self._db_manager.update_run_status(
                    run_id=self._run_id,
                    status="partial",
                    error_message=f"Termination: {reason}"
                )
            
            return CollectionReport(
                run_id=self._run_id,
                total_records=total_records,
                errors=[f"Termination: {reason}"],
                duration_seconds=duration,
            )
