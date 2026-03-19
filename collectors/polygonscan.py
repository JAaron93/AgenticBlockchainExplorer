"""Polygonscan blockchain explorer collector."""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from config.models import ExplorerConfig, RetryConfig
from collectors.base import ExplorerCollector
from collectors.models import Transaction, Holder, ActivityType


# Use standard logging to avoid circular imports
logger = logging.getLogger(__name__)


class PolygonscanCollector(ExplorerCollector):
    """Collector for Polygonscan API (Polygon/Matic blockchain).

    Implements data collection from Polygonscan's API for ERC-20 token
    transactions and holder information. The API is compatible with
    Etherscan's API format.
    """

    # Token decimals for stablecoins on Polygon
    # USDC on Polygon uses 6 decimals (bridged USDC.e)
    # USDT on Polygon uses 6 decimals
    TOKEN_DECIMALS = {
        "USDC": 6,
        "USDT": 6,
    }

    def __init__(
        self, config: ExplorerConfig, retry_config: Optional[RetryConfig] = None
    ):
        """Initialize the Polygonscan collector.

        Args:
            config: Explorer configuration with API details
            retry_config: Retry configuration (uses defaults if not provided)
        """
        super().__init__(config, retry_config)

        # Verify this is configured for Polygon
        if config.chain != "polygon":
            logger.warning(
                f"PolygonscanCollector initialized with chain '{config.chain}', expected 'polygon'"
            )
