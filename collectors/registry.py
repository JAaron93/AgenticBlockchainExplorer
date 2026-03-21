"""Central registry for blockchain collectors.

Allows for dynamic discovery and registration of blockchain explorer
collectors, supporting a modular plugin-like architecture.
"""

import logging
from typing import Dict, Type, Optional

from collectors.base import ExplorerCollector
from collectors.etherscan import EtherscanCollector
from collectors.bscscan import BscscanCollector
from collectors.polygonscan import PolygonscanCollector

logger = logging.getLogger(__name__)


class CollectorRegistry:
    """Registry for blockchain explorer collectors."""

    _collectors: Dict[str, Type[ExplorerCollector]] = {
        "ethereum": EtherscanCollector,
        "bsc": BscscanCollector,
        "polygon": PolygonscanCollector,
    }

    @classmethod
    def register(
        cls, chain: str, collector_class: Type[ExplorerCollector]
    ) -> None:
        """Register a new collector class for a chain.

        Args:
            chain: The blockchain chain name (e.g., 'ethereum', 'solana').
            collector_class: The collector class to register.
        """
        normalized_chain = chain.lower()
        if normalized_chain in cls._collectors:
            logger.warning(
                f"Overwriting collector for chain '{chain}'"
            )

        cls._collectors[normalized_chain] = collector_class
        logger.info(
            f"Registered collector '{collector_class.__name__}' "
            f"for chain '{chain}'"
        )

    @classmethod
    def get_collector_class(
        cls, chain: str
    ) -> Optional[Type[ExplorerCollector]]:
        """Get the collector class for a chain.

        Args:
            chain: The blockchain chain name.

        Returns:
            The collector class or None if not found.
        """
        return cls._collectors.get(chain.lower())

    @classmethod
    def list_supported_chains(cls) -> list[str]:
        """List all supported blockchain chains.

        Returns:
            List of supported chain names.
        """
        return list(cls._collectors.keys())
