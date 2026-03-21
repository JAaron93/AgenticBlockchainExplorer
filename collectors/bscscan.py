"""BscScan blockchain explorer collector."""

import logging
from typing import Optional

from config.models import ExplorerConfig, RetryConfig
from collectors.base import ExplorerCollector


# Use standard logging to avoid circular imports
logger = logging.getLogger(__name__)


class BscscanCollector(ExplorerCollector):
    """Collector for BscScan API (Binance Smart Chain).

    Implements data collection from BscScan's API for BEP-20 token
    transactions and holder information. The API is compatible with
    Etherscan's API format.
    """

    # Token decimals for stablecoins on BSC
    # USDC on BSC uses 18 decimals, USDT uses 18 decimals
    TOKEN_DECIMALS = {
        "USDC": 18,
        "USDT": 18,
    }

    def __init__(
        self, config: ExplorerConfig, retry_config: Optional[RetryConfig] = None
    ):
        """Initialize the BscScan collector.

        Args:
            config: Explorer configuration with API details
            retry_config: Retry configuration (uses defaults if not provided)
        """
        super().__init__(config, retry_config)

        # Verify this is configured for BSC
        if config.chain != "bsc":
            logger.warning(
                f"BscscanCollector initialized with chain '{config.chain}', expected 'bsc'"
            )
