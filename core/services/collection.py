"""Service for unified blockchain data collection.

Provides high-level methods for parallel data gathering from multiple
explorers, integrating security components and rate limiting consistently.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional

from collectors.models import ExplorerData
from collectors.registry import CollectorRegistry
from config.models import Config, ExplorerConfig

logger = logging.getLogger(__name__)


class CollectionService:
    """Service for orchestrating parallel data collection."""

    def __init__(self, config: Config):
        """Initialize the collection service.

        Args:
            config: Application configuration.
        """
        self._config = config

    async def collect_parallel(
        self,
        stablecoins: List[str],
        explorers: Optional[List[str]] = None,
        max_records: Optional[int] = None,
        run_id: Optional[str] = None,
        timeout_manager: Optional[Any] = None
    ) -> List[ExplorerData]:
        """Collect data from multiple explorers in parallel.

        Args:
            stablecoins: List of stablecoin symbols to collect.
            explorers: Optional filter for specific explorers.
            max_records: Maximum records per explorer.
            run_id: Optional run ID for logging.
            timeout_manager: Optional TimeoutManager for enforcement.

        Returns:
            List of ExplorerData results.
        """
        # Filter enabled explorers
        enabled_explorers = []
        for exp_config in self._config.explorers:
            if not exp_config.enabled:
                continue
            if explorers and exp_config.name not in explorers:
                continue
            enabled_explorers.append(exp_config)

        if not enabled_explorers:
            logger.warning("No explorers selected for collection")
            return []

        tasks = []
        for exp_config in enabled_explorers:
            tasks.append(self._collect_from_single(
                exp_config=exp_config,
                stablecoins=stablecoins,
                max_records=max_records,
                run_id=run_id,
                timeout_manager=timeout_manager
            ))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                exp_name = enabled_explorers[i].name
                logger.error(f"Collector {exp_name} failed with exception: {result}")
                final_results.append(ExplorerData(
                    explorer_name=exp_name,
                    chain=enabled_explorers[i].chain,
                    errors=[str(result)]
                ))
            else:
                final_results.append(result)
        
        return final_results

    async def _collect_from_single(
        self,
        exp_config: ExplorerConfig,
        stablecoins: List[str],
        max_records: Optional[int],
        run_id: Optional[str],
        timeout_manager: Optional[Any]
    ) -> ExplorerData:
        """Internal method to collect from a single explorer."""
        collector_class = CollectorRegistry.get_collector_class(exp_config.chain)
        if not collector_class:
            return ExplorerData(
                explorer_name=exp_config.name,
                chain=exp_config.chain,
                errors=[f"No collector class registered for chain {exp_config.chain}"]
            )

        # Get relevant contract addresses
        base_stablecoins = self._config.stablecoins
        contracts = {
            symbol: getattr(base_stablecoins[symbol], exp_config.chain)
            for symbol in stablecoins
            if symbol in base_stablecoins and hasattr(base_stablecoins[symbol], exp_config.chain)
        }

        if not contracts:
            return ExplorerData(
                explorer_name=exp_config.name,
                chain=exp_config.chain,
                errors=["No relevant stablecoin contracts found for this chain"]
            )

        collector = collector_class(
            config=exp_config,
            retry_config=self._config.retry
        )

        async def run_collect():
            async with collector:
                return await collector.collect_all(
                    stablecoins=contracts,
                    max_records=max_records or 1000,
                    run_id=run_id
                )

        try:
            if timeout_manager:
                return await timeout_manager.run_with_timeout(
                    run_collect(),
                    collection_name=exp_config.name
                )
            else:
                return await run_collect()
        except Exception as e:
            logger.error(f"Error collecting from {exp_config.name}: {e}")
            return ExplorerData(
                explorer_name=exp_config.name,
                chain=exp_config.chain,
                errors=[str(e)]
            )
