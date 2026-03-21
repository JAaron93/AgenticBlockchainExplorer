"""Etherscan blockchain explorer collector."""

import logging
from typing import Optional

from config.models import ExplorerConfig, RetryConfig
from collectors.base import ExplorerCollector


# Use standard logging to avoid circular imports
logger = logging.getLogger(__name__)


class EtherscanCollector(ExplorerCollector):
    """Collector for Etherscan API (Ethereum blockchain).

    Implements data collection from Etherscan's API for ERC-20 token
    transactions and holder information.
    """

    # Token decimals for stablecoins (both USDC and USDT use 6 decimals on Ethereum)
    TOKEN_DECIMALS = {
        "USDC": 6,
        "USDT": 6,
    }

    def __init__(
        self, config: ExplorerConfig, retry_config: Optional[RetryConfig] = None
    ):
        """Initialize the Etherscan collector.

        Args:
            config: Explorer configuration with API details
            retry_config: Retry configuration (uses defaults if not provided)
        """
        super().__init__(config, retry_config)

        # Verify this is configured for Ethereum
        if config.chain != "ethereum":
            logger.warning(
                f"EtherscanCollector initialized with chain '{config.chain}', expected 'ethereum'"
            )
