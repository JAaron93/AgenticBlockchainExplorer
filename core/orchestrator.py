"""Agent orchestrator for coordinating data collection.

Orchestrates data collection from multiple blockchain explorers,
coordinates classification, aggregation, and export of results.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from config.models import Config, ExplorerConfig
from collectors.base import ExplorerCollector
from collectors.models import ExplorerData
from collectors.etherscan import EtherscanCollector
from collectors.bscscan import BscscanCollector
from collectors.polygonscan import PolygonscanCollector
from collectors.classifier import ActivityClassifier
from collectors.aggregator import DataAggregator, AggregatedData
from collectors.exporter import JSONExporter
from core.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


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


class AgentOrchestrator:
    """Orchestrates data collection from blockchain explorers.

    Coordinates the collection of stablecoin data from multiple blockchain
    explorers, classifies activities, aggregates results, and exports
    to JSON and database.
    """

    def __init__(
        self,
        config: Config,
        run_id: str,
        db_manager: Optional[DatabaseManager] = None,
        user_id: Optional[str] = None,
    ):
        """Initialize the orchestrator.

        Args:
            config: Application configuration.
            run_id: Unique identifier for this run.
            db_manager: Database manager for persistence.
            user_id: Optional user ID who initiated the run.
        """
        self._config = config
        self._run_id = run_id
        self._db_manager = db_manager
        self._user_id = user_id
        self._collectors: list[ExplorerCollector] = []
        self._classifier = ActivityClassifier()
        self._aggregator = DataAggregator()
        self._exporter = JSONExporter(
            db_manager=db_manager,
            output_directory=config.output.directory
        )

    @property
    def run_id(self) -> str:
        """Get the run ID."""
        return self._run_id

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

        Returns:
            List of initialized collectors.
        """
        collectors = []
        for explorer_config in self._config.explorers:
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

        Args:
            chain: The blockchain chain name.

        Returns:
            Dict mapping stablecoin symbol to contract address.
        """
        addresses = {}
        for symbol, config in self._config.stablecoins.items():
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
        """Collect data from a single explorer.

        Args:
            collector: The collector to use.

        Returns:
            ExplorerData with collected transactions and holders.
        """
        stablecoins = self._get_stablecoin_addresses(collector.chain)
        max_records = self._config.output.max_records_per_explorer

        logger.info(
            f"Starting collection from {collector.name}",
            extra={
                "run_id": self._run_id,
                "explorer": collector.name,
                "chain": collector.chain,
                "stablecoins": list(stablecoins.keys()),
            }
        )

        try:
            async with collector:
                result = await collector.collect_all(
                    stablecoins=stablecoins,
                    max_records=max_records,
                    run_id=self._run_id
                )
            return result
        except Exception as e:
            logger.error(
                f"Collection failed for {collector.name}: {e}",
                extra={
                    "run_id": self._run_id,
                    "explorer": collector.name,
                    "error": str(e),
                },
                exc_info=True
            )
            return ExplorerData(
                explorer_name=collector.name,
                chain=collector.chain,
                errors=[f"Collection failed: {e}"]
            )

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
        tasks = [
            self._collect_from_explorer(collector)
            for collector in self._collectors
        ]

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
        """Execute the full data collection pipeline.

        Orchestrates:
        1. Collection from all explorers in parallel
        2. Classification of activities
        3. Aggregation and deduplication
        4. Export to JSON and database

        Returns:
            CollectionReport with results summary.
        """
        start_time = time.time()

        logger.info(
            f"Starting agent run {self._run_id}",
            extra={
                "run_id": self._run_id,
                "user_id": self._user_id,
                "explorers": [e.name for e in self._config.explorers],
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
            logger.error(
                f"Agent run {self._run_id} failed: {e}",
                extra={
                    "run_id": self._run_id,
                    "error": str(e),
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
