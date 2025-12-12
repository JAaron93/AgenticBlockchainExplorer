"""Data aggregator for combining results from multiple blockchain explorers."""

import logging
from dataclasses import dataclass, field
from decimal import Decimal

from collectors.models import Transaction, Holder, ExplorerData, ActivityType


# Use standard logging to avoid circular imports
logger = logging.getLogger(__name__)


@dataclass
class StablecoinSummary:
    """Summary statistics for a single stablecoin."""

    total_transactions: int = 0
    total_volume: Decimal = Decimal("0")
    unique_addresses: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_transactions": self.total_transactions,
            "total_volume": str(self.total_volume),
            "unique_addresses": self.unique_addresses,
        }


@dataclass
class AggregatedData:
    """Aggregated data from all explorers with summary statistics."""

    transactions: list[Transaction] = field(default_factory=list)
    holders: list[Holder] = field(default_factory=list)
    explorers_queried: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Summary statistics
    by_stablecoin: dict[str, StablecoinSummary] = field(default_factory=dict)
    by_activity_type: dict[str, int] = field(default_factory=dict)
    # by_chain counts total records (transactions + holders) per chain.
    # This is intentionally a combined metric representing all data collected
    # from each blockchain network.
    by_chain: dict[str, int] = field(default_factory=dict)

    @property
    def total_records(self) -> int:
        """Get total number of records."""
        return len(self.transactions) + len(self.holders)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": {
                "by_stablecoin": {
                    coin: summary.to_dict()
                    for coin, summary in self.by_stablecoin.items()
                },
                "by_activity_type": self.by_activity_type,
                "by_chain": self.by_chain,
            },
            "transactions": [tx.to_dict() for tx in self.transactions],
            "holders": [h.to_dict() for h in self.holders],
        }


class DataAggregator:
    """Aggregates and deduplicates data from multiple blockchain explorers.

    Combines transaction and holder data from multiple sources, removes
    duplicates, and generates summary statistics.
    """

    def aggregate(
        self, explorer_results: list[ExplorerData]
    ) -> AggregatedData:
        """Aggregate data from multiple explorer results.

        Args:
            explorer_results: List of ExplorerData from different collectors

        Returns:
            AggregatedData with deduplicated transactions, merged holders,
            and summary statistics
        """
        result = AggregatedData()

        all_transactions: list[Transaction] = []
        all_holders: list[Holder] = []

        for explorer_data in explorer_results:
            result.explorers_queried.append(explorer_data.explorer_name)
            all_transactions.extend(explorer_data.transactions)
            all_holders.extend(explorer_data.holders)
            result.errors.extend(explorer_data.errors)

            logger.info(
                f"Aggregating {len(explorer_data.transactions)} transactions "
                f"and {len(explorer_data.holders)} holders "
                f"from {explorer_data.explorer_name}",
                extra={
                    "explorer": explorer_data.explorer_name,
                    "transactions": len(explorer_data.transactions),
                    "holders": len(explorer_data.holders),
                }
            )

        # Deduplicate transactions
        result.transactions = self.deduplicate_transactions(all_transactions)

        # Merge holder data
        result.holders = self.merge_holder_data(all_holders)

        # Generate summary statistics
        self._generate_summary_statistics(result)

        logger.info(
            f"Aggregation complete: {len(result.transactions)} unique "
            f"transactions, {len(result.holders)} unique holders "
            f"from {len(result.explorers_queried)} explorers",
            extra={
                "total_transactions": len(result.transactions),
                "total_holders": len(result.holders),
                "explorers": result.explorers_queried,
            }
        )

        return result

    def deduplicate_transactions(
        self,
        transactions: list[Transaction]
    ) -> list[Transaction]:
        """Remove duplicate transactions based on transaction hash.

        When duplicates are found (same transaction hash), keeps the first
        occurrence. Transactions from different chains with the same hash
        are considered different transactions.

        Args:
            transactions: List of transactions with potential duplicates

        Returns:
            List of unique transactions
        """
        seen: dict[tuple[str, str], Transaction] = {}
        duplicates_count = 0

        for tx in transactions:
            # Use (hash, chain) as key since same hash on different chains
            # represents different transactions
            key = (tx.transaction_hash, tx.chain)

            if key not in seen:
                seen[key] = tx
            else:
                duplicates_count += 1
                logger.debug(
                    f"Duplicate transaction: {tx.transaction_hash} "
                    f"on {tx.chain}",
                    extra={
                        "tx_hash": tx.transaction_hash,
                        "chain": tx.chain,
                        "source": tx.source_explorer,
                    }
                )

        if duplicates_count > 0:
            logger.info(
                f"Removed {duplicates_count} duplicate transactions",
                extra={"duplicates_removed": duplicates_count}
            )

        return list(seen.values())

    def merge_holder_data(self, holders: list[Holder]) -> list[Holder]:
        """Merge holder data from multiple sources.

        When the same address holds the same stablecoin on the same chain
        but is reported by multiple explorers, merge the data by:
        - Taking the maximum balance (most recent/accurate)
        - Taking the earliest first_seen date
        - Taking the latest last_activity date
        - OR-ing the is_store_of_value flag

        Args:
            holders: List of holders potentially containing duplicates

        Returns:
            List of merged unique holders
        """
        merged: dict[tuple[str, str, str], Holder] = {}
        merges_count = 0

        for holder in holders:
            # Key is (address, stablecoin, chain)
            key = (holder.address.lower(), holder.stablecoin, holder.chain)

            if key not in merged:
                merged[key] = holder
            else:
                existing = merged[key]
                merges_count += 1

                # Merge: max balance, earliest first_seen, latest last_activity
                source = f"{existing.source_explorer},{holder.source_explorer}"
                merged[key] = Holder(
                    address=existing.address,
                    balance=max(existing.balance, holder.balance),
                    stablecoin=existing.stablecoin,
                    chain=existing.chain,
                    first_seen=min(existing.first_seen, holder.first_seen),
                    last_activity=max(
                        existing.last_activity, holder.last_activity
                    ),
                    is_store_of_value=(
                        existing.is_store_of_value or holder.is_store_of_value
                    ),
                    source_explorer=source,
                )

                logger.debug(
                    f"Merged holder: {holder.address} "
                    f"({holder.stablecoin} on {holder.chain})",
                    extra={
                        "address": holder.address,
                        "stablecoin": holder.stablecoin,
                        "chain": holder.chain,
                    }
                )

        if merges_count > 0:
            logger.info(
                f"Merged {merges_count} duplicate holder records",
                extra={"merges_count": merges_count}
            )

        return list(merged.values())

    def _generate_summary_statistics(self, data: AggregatedData) -> None:
        """Generate summary statistics for the aggregated data.

        Populates the by_stablecoin, by_activity_type, and by_chain
        fields of the AggregatedData object.

        Note on by_chain: This metric intentionally combines transaction
        counts and holder counts to represent total records collected per
        chain. This provides a single view of data volume per blockchain.

        Args:
            data: AggregatedData object to populate with statistics
        """
        # Initialize counters
        stablecoin_stats: dict[str, StablecoinSummary] = {}
        activity_counts: dict[str, int] = {}
        # chain_counts tracks total records (transactions + holders) per chain
        chain_counts: dict[str, int] = {}

        # Track unique addresses per stablecoin
        stablecoin_addresses: dict[str, set[str]] = {}

        # Process transactions - count each transaction toward its chain
        for tx in data.transactions:
            # By stablecoin
            if tx.stablecoin not in stablecoin_stats:
                stablecoin_stats[tx.stablecoin] = StablecoinSummary()
                stablecoin_addresses[tx.stablecoin] = set()

            stats = stablecoin_stats[tx.stablecoin]
            stats.total_transactions += 1
            stats.total_volume += tx.amount
            stablecoin_addresses[tx.stablecoin].add(tx.from_address.lower())
            stablecoin_addresses[tx.stablecoin].add(tx.to_address.lower())

            # By activity type
            activity_key = tx.activity_type.value
            activity_counts[activity_key] = (
                activity_counts.get(activity_key, 0) + 1
            )

            # By chain - count transaction as one record
            chain_counts[tx.chain] = chain_counts.get(tx.chain, 0) + 1

        # Set unique address counts
        for coin, addresses in stablecoin_addresses.items():
            stablecoin_stats[coin].unique_addresses = len(addresses)

        # Process holders - count each holder toward its chain
        for holder in data.holders:
            # By chain - count holder as one record
            chain_counts[holder.chain] = chain_counts.get(holder.chain, 0) + 1

            # Add holder addresses to stablecoin unique addresses
            if holder.stablecoin in stablecoin_addresses:
                stablecoin_addresses[holder.stablecoin].add(
                    holder.address.lower()
                )
                stablecoin_stats[holder.stablecoin].unique_addresses = len(
                    stablecoin_addresses[holder.stablecoin]
                )

            # Count store of value holders in activity type
            if holder.is_store_of_value:
                sov_key = ActivityType.STORE_OF_VALUE.value
                activity_counts[sov_key] = activity_counts.get(sov_key, 0) + 1

        data.by_stablecoin = stablecoin_stats
        data.by_activity_type = activity_counts
        data.by_chain = chain_counts

        logger.debug(
            "Generated summary statistics",
            extra={
                "stablecoins": list(stablecoin_stats.keys()),
                "activity_types": list(activity_counts.keys()),
                "chains": list(chain_counts.keys()),
            }
        )
