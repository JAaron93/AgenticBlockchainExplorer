"""Unit tests for data aggregation and deduplication."""

from datetime import datetime, timezone, timedelta
from decimal import Decimal

import pytest

from collectors.aggregator import DataAggregator, AggregatedData, StablecoinSummary
from collectors.models import ActivityType, Transaction, Holder, ExplorerData


@pytest.fixture
def aggregator():
    """Create a DataAggregator instance."""
    return DataAggregator()


@pytest.fixture
def sample_transaction():
    """Create a sample transaction."""
    return Transaction(
        transaction_hash="0x123abc",
        block_number=18500000,
        timestamp=datetime.now(timezone.utc),
        from_address="0x1111111111111111111111111111111111111111",
        to_address="0x2222222222222222222222222222222222222222",
        amount=Decimal("1000.50"),
        stablecoin="USDC",
        chain="ethereum",
        activity_type=ActivityType.TRANSACTION,
        source_explorer="etherscan",
    )


@pytest.fixture
def sample_holder():
    """Create a sample holder."""
    now = datetime.now(timezone.utc)
    return Holder(
        address="0x1111111111111111111111111111111111111111",
        balance=Decimal("50000"),
        stablecoin="USDC",
        chain="ethereum",
        first_seen=now - timedelta(days=60),
        last_activity=now - timedelta(days=5),
        is_store_of_value=True,
        source_explorer="etherscan",
    )


class TestDeduplicateTransactions:
    """Tests for deduplicate_transactions method."""

    def test_no_duplicates_returns_all(self, aggregator):
        """When no duplicates, all transactions are returned."""
        now = datetime.now(timezone.utc)
        transactions = [
            Transaction(
                transaction_hash=f"0x{i}",
                block_number=18500000 + i,
                timestamp=now,
                from_address="0x1111111111111111111111111111111111111111",
                to_address="0x2222222222222222222222222222222222222222",
                amount=Decimal("100"),
                stablecoin="USDC",
                chain="ethereum",
                activity_type=ActivityType.TRANSACTION,
                source_explorer="etherscan",
            )
            for i in range(5)
        ]

        result = aggregator.deduplicate_transactions(transactions)
        assert len(result) == 5

    def test_duplicates_removed(self, aggregator):
        """Duplicate transactions (same hash, same chain) are removed."""
        now = datetime.now(timezone.utc)
        tx1 = Transaction(
            transaction_hash="0xdupe",
            block_number=18500000,
            timestamp=now,
            from_address="0x1111111111111111111111111111111111111111",
            to_address="0x2222222222222222222222222222222222222222",
            amount=Decimal("100"),
            stablecoin="USDC",
            chain="ethereum",
            activity_type=ActivityType.TRANSACTION,
            source_explorer="etherscan",
        )
        tx2 = Transaction(
            transaction_hash="0xdupe",  # Same hash
            block_number=18500000,
            timestamp=now,
            from_address="0x1111111111111111111111111111111111111111",
            to_address="0x2222222222222222222222222222222222222222",
            amount=Decimal("100"),
            stablecoin="USDC",
            chain="ethereum",  # Same chain
            activity_type=ActivityType.TRANSACTION,
            source_explorer="bscscan",  # Different source
        )

        result = aggregator.deduplicate_transactions([tx1, tx2])
        assert len(result) == 1
        assert result[0].source_explorer == "etherscan"  # First one kept

    def test_same_hash_different_chain_not_duplicate(self, aggregator):
        """Same hash on different chains are different transactions."""
        now = datetime.now(timezone.utc)
        tx1 = Transaction(
            transaction_hash="0xsame",
            block_number=18500000,
            timestamp=now,
            from_address="0x1111111111111111111111111111111111111111",
            to_address="0x2222222222222222222222222222222222222222",
            amount=Decimal("100"),
            stablecoin="USDC",
            chain="ethereum",
            activity_type=ActivityType.TRANSACTION,
            source_explorer="etherscan",
        )
        tx2 = Transaction(
            transaction_hash="0xsame",  # Same hash
            block_number=30000000,
            timestamp=now,
            from_address="0x1111111111111111111111111111111111111111",
            to_address="0x2222222222222222222222222222222222222222",
            amount=Decimal("100"),
            stablecoin="USDC",
            chain="bsc",  # Different chain
            activity_type=ActivityType.TRANSACTION,
            source_explorer="bscscan",
        )

        result = aggregator.deduplicate_transactions([tx1, tx2])
        assert len(result) == 2

    def test_empty_list_returns_empty(self, aggregator):
        """Empty input returns empty output."""
        result = aggregator.deduplicate_transactions([])
        assert len(result) == 0


class TestMergeHolderData:
    """Tests for merge_holder_data method."""

    def test_no_duplicates_returns_all(self, aggregator):
        """When no duplicates, all holders are returned."""
        now = datetime.now(timezone.utc)
        holders = [
            Holder(
                address=f"0x{i}111111111111111111111111111111111111111",
                balance=Decimal("1000"),
                stablecoin="USDC",
                chain="ethereum",
                first_seen=now - timedelta(days=30),
                last_activity=now,
                is_store_of_value=False,
                source_explorer="etherscan",
            )
            for i in range(3)
        ]

        result = aggregator.merge_holder_data(holders)
        assert len(result) == 3

    def test_duplicate_holders_merged(self, aggregator):
        """Duplicate holders (same address, stablecoin, chain) are merged."""
        now = datetime.now(timezone.utc)
        h1 = Holder(
            address="0x1111111111111111111111111111111111111111",
            balance=Decimal("1000"),
            stablecoin="USDC",
            chain="ethereum",
            first_seen=now - timedelta(days=30),
            last_activity=now - timedelta(days=10),
            is_store_of_value=False,
            source_explorer="etherscan",
        )
        h2 = Holder(
            address="0x1111111111111111111111111111111111111111",  # Same
            balance=Decimal("1500"),  # Higher balance
            stablecoin="USDC",  # Same
            chain="ethereum",  # Same
            first_seen=now - timedelta(days=20),  # Later first_seen
            last_activity=now - timedelta(days=5),  # More recent
            is_store_of_value=True,  # Different
            source_explorer="bscscan",
        )

        result = aggregator.merge_holder_data([h1, h2])
        assert len(result) == 1

        merged = result[0]
        # Max balance
        assert merged.balance == Decimal("1500")
        # Earliest first_seen
        assert merged.first_seen == now - timedelta(days=30)
        # Latest last_activity
        assert merged.last_activity == now - timedelta(days=5)
        # OR of is_store_of_value
        assert merged.is_store_of_value is True
        # Combined sources
        assert "etherscan" in merged.source_explorer
        assert "bscscan" in merged.source_explorer

    def test_case_insensitive_address_merge(self, aggregator):
        """Address matching for merge is case-insensitive."""
        now = datetime.now(timezone.utc)
        h1 = Holder(
            address="0xABCDEF1234567890ABCDEF1234567890ABCDEF12",
            balance=Decimal("1000"),
            stablecoin="USDC",
            chain="ethereum",
            first_seen=now - timedelta(days=30),
            last_activity=now,
            is_store_of_value=False,
            source_explorer="etherscan",
        )
        h2 = Holder(
            address="0xabcdef1234567890abcdef1234567890abcdef12",  # lowercase
            balance=Decimal("2000"),
            stablecoin="USDC",
            chain="ethereum",
            first_seen=now - timedelta(days=20),
            last_activity=now,
            is_store_of_value=False,
            source_explorer="bscscan",
        )

        result = aggregator.merge_holder_data([h1, h2])
        assert len(result) == 1

    def test_different_stablecoin_not_merged(self, aggregator):
        """Same address with different stablecoin are not merged."""
        now = datetime.now(timezone.utc)
        h1 = Holder(
            address="0x1111111111111111111111111111111111111111",
            balance=Decimal("1000"),
            stablecoin="USDC",
            chain="ethereum",
            first_seen=now - timedelta(days=30),
            last_activity=now,
            is_store_of_value=False,
            source_explorer="etherscan",
        )
        h2 = Holder(
            address="0x1111111111111111111111111111111111111111",
            balance=Decimal("2000"),
            stablecoin="USDT",  # Different stablecoin
            chain="ethereum",
            first_seen=now - timedelta(days=20),
            last_activity=now,
            is_store_of_value=False,
            source_explorer="etherscan",
        )

        result = aggregator.merge_holder_data([h1, h2])
        assert len(result) == 2

    def test_different_chain_not_merged(self, aggregator):
        """Same address on different chains are not merged."""
        now = datetime.now(timezone.utc)
        h1 = Holder(
            address="0x1111111111111111111111111111111111111111",
            balance=Decimal("1000"),
            stablecoin="USDC",
            chain="ethereum",
            first_seen=now - timedelta(days=30),
            last_activity=now,
            is_store_of_value=False,
            source_explorer="etherscan",
        )
        h2 = Holder(
            address="0x1111111111111111111111111111111111111111",
            balance=Decimal("2000"),
            stablecoin="USDC",
            chain="bsc",  # Different chain
            first_seen=now - timedelta(days=20),
            last_activity=now,
            is_store_of_value=False,
            source_explorer="bscscan",
        )

        result = aggregator.merge_holder_data([h1, h2])
        assert len(result) == 2


class TestAggregate:
    """Tests for aggregate method."""

    def test_aggregate_multiple_explorers(self, aggregator):
        """Aggregate data from multiple explorers."""
        now = datetime.now(timezone.utc)

        explorer1 = ExplorerData(
            explorer_name="etherscan",
            chain="ethereum",
            transactions=[
                Transaction(
                    transaction_hash="0xeth1",
                    block_number=18500000,
                    timestamp=now,
                    from_address="0x1111111111111111111111111111111111111111",
                    to_address="0x2222222222222222222222222222222222222222",
                    amount=Decimal("1000"),
                    stablecoin="USDC",
                    chain="ethereum",
                    activity_type=ActivityType.TRANSACTION,
                    source_explorer="etherscan",
                ),
            ],
            holders=[],
        )

        explorer2 = ExplorerData(
            explorer_name="bscscan",
            chain="bsc",
            transactions=[
                Transaction(
                    transaction_hash="0xbsc1",
                    block_number=30000000,
                    timestamp=now,
                    from_address="0x3333333333333333333333333333333333333333",
                    to_address="0x4444444444444444444444444444444444444444",
                    amount=Decimal("2000"),
                    stablecoin="USDT",
                    chain="bsc",
                    activity_type=ActivityType.TRANSACTION,
                    source_explorer="bscscan",
                ),
            ],
            holders=[],
        )

        result = aggregator.aggregate([explorer1, explorer2])

        assert len(result.transactions) == 2
        assert len(result.explorers_queried) == 2
        assert "etherscan" in result.explorers_queried
        assert "bscscan" in result.explorers_queried

    def test_aggregate_generates_summary_statistics(self, aggregator):
        """Aggregate generates correct summary statistics."""
        now = datetime.now(timezone.utc)

        explorer = ExplorerData(
            explorer_name="etherscan",
            chain="ethereum",
            transactions=[
                Transaction(
                    transaction_hash="0x1",
                    block_number=18500000,
                    timestamp=now,
                    from_address="0x1111111111111111111111111111111111111111",
                    to_address="0x2222222222222222222222222222222222222222",
                    amount=Decimal("1000"),
                    stablecoin="USDC",
                    chain="ethereum",
                    activity_type=ActivityType.TRANSACTION,
                    source_explorer="etherscan",
                ),
                Transaction(
                    transaction_hash="0x2",
                    block_number=18500001,
                    timestamp=now,
                    from_address="0x3333333333333333333333333333333333333333",
                    to_address="0x4444444444444444444444444444444444444444",
                    amount=Decimal("500"),
                    stablecoin="USDC",
                    chain="ethereum",
                    activity_type=ActivityType.TRANSACTION,
                    source_explorer="etherscan",
                ),
            ],
            holders=[
                Holder(
                    address="0x5555555555555555555555555555555555555555",
                    balance=Decimal("10000"),
                    stablecoin="USDC",
                    chain="ethereum",
                    first_seen=now - timedelta(days=60),
                    last_activity=now - timedelta(days=35),
                    is_store_of_value=True,
                    source_explorer="etherscan",
                ),
            ],
        )

        result = aggregator.aggregate([explorer])

        # Check by_stablecoin
        assert "USDC" in result.by_stablecoin
        usdc_stats = result.by_stablecoin["USDC"]
        assert usdc_stats.total_transactions == 2
        assert usdc_stats.total_volume == Decimal("1500")
        # 4 unique addresses from transactions + 1 from holder
        assert usdc_stats.unique_addresses == 5

        # Check by_activity_type
        assert result.by_activity_type.get("transaction") == 2
        assert result.by_activity_type.get("store_of_value") == 1

        # Check by_chain (transactions + holders)
        assert result.by_chain.get("ethereum") == 3  # 2 tx + 1 holder

    def test_aggregate_collects_errors(self, aggregator):
        """Aggregate collects errors from all explorers."""
        explorer1 = ExplorerData(
            explorer_name="etherscan",
            chain="ethereum",
            transactions=[],
            holders=[],
            errors=["Rate limit exceeded"],
        )
        explorer2 = ExplorerData(
            explorer_name="bscscan",
            chain="bsc",
            transactions=[],
            holders=[],
            errors=["API key invalid"],
        )

        result = aggregator.aggregate([explorer1, explorer2])

        assert len(result.errors) == 2
        assert "Rate limit exceeded" in result.errors
        assert "API key invalid" in result.errors

    def test_aggregate_empty_input(self, aggregator):
        """Aggregate with empty input returns empty result."""
        result = aggregator.aggregate([])

        assert len(result.transactions) == 0
        assert len(result.holders) == 0
        assert len(result.explorers_queried) == 0
        assert result.total_records == 0


class TestAggregatedData:
    """Tests for AggregatedData dataclass."""

    def test_total_records_property(self):
        """total_records returns sum of transactions and holders."""
        now = datetime.now(timezone.utc)
        data = AggregatedData(
            transactions=[
                Transaction(
                    transaction_hash="0x1",
                    block_number=18500000,
                    timestamp=now,
                    from_address="0x1111111111111111111111111111111111111111",
                    to_address="0x2222222222222222222222222222222222222222",
                    amount=Decimal("100"),
                    stablecoin="USDC",
                    chain="ethereum",
                    activity_type=ActivityType.TRANSACTION,
                    source_explorer="etherscan",
                ),
            ],
            holders=[
                Holder(
                    address="0x3333333333333333333333333333333333333333",
                    balance=Decimal("1000"),
                    stablecoin="USDC",
                    chain="ethereum",
                    first_seen=now - timedelta(days=30),
                    last_activity=now,
                    is_store_of_value=False,
                    source_explorer="etherscan",
                ),
                Holder(
                    address="0x4444444444444444444444444444444444444444",
                    balance=Decimal("2000"),
                    stablecoin="USDT",
                    chain="bsc",
                    first_seen=now - timedelta(days=20),
                    last_activity=now,
                    is_store_of_value=True,
                    source_explorer="bscscan",
                ),
            ],
        )

        assert data.total_records == 3

    def test_to_dict_serialization(self):
        """to_dict produces valid dictionary for JSON serialization."""
        now = datetime.now(timezone.utc)
        data = AggregatedData(
            transactions=[
                Transaction(
                    transaction_hash="0x1",
                    block_number=18500000,
                    timestamp=now,
                    from_address="0x1111111111111111111111111111111111111111",
                    to_address="0x2222222222222222222222222222222222222222",
                    amount=Decimal("100"),
                    stablecoin="USDC",
                    chain="ethereum",
                    activity_type=ActivityType.TRANSACTION,
                    source_explorer="etherscan",
                ),
            ],
            holders=[],
            by_stablecoin={"USDC": StablecoinSummary(
                total_transactions=1,
                total_volume=Decimal("100"),
                unique_addresses=2,
            )},
            by_activity_type={"transaction": 1},
            by_chain={"ethereum": 1},
        )

        result = data.to_dict()

        assert "summary" in result
        assert "transactions" in result
        assert "holders" in result
        assert result["summary"]["by_stablecoin"]["USDC"]["total_transactions"] == 1
        assert result["summary"]["by_stablecoin"]["USDC"]["total_volume"] == "100"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
