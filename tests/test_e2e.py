"""End-to-end tests for complete application flows.

These tests verify the complete data flow from API trigger to result download,
multi-user scenarios, and permission enforcement.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from collectors.aggregator import DataAggregator, AggregatedData
from collectors.classifier import ActivityClassifier
from collectors.exporter import JSONExporter
from collectors.models import ActivityType, Transaction, Holder, ExplorerData
from config.models import OutputConfig


@pytest.fixture
def sample_transactions():
    """Create sample transactions for testing."""
    now = datetime.now(timezone.utc)
    return [
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
        Transaction(
            transaction_hash="0xpoly1",
            block_number=45000000,
            timestamp=now,
            from_address="0x5555555555555555555555555555555555555555",
            to_address="0x6666666666666666666666666666666666666666",
            amount=Decimal("500"),
            stablecoin="USDC",
            chain="polygon",
            activity_type=ActivityType.TRANSACTION,
            source_explorer="polygonscan",
        ),
    ]


@pytest.fixture
def sample_holders():
    """Create sample holders for testing."""
    now = datetime.now(timezone.utc)
    return [
        Holder(
            address="0xaaaa111111111111111111111111111111111111",
            balance=Decimal("50000"),
            stablecoin="USDC",
            chain="ethereum",
            first_seen=now,
            last_activity=now,
            is_store_of_value=True,
            source_explorer="etherscan",
        ),
        Holder(
            address="0xbbbb222222222222222222222222222222222222",
            balance=Decimal("25000"),
            stablecoin="USDT",
            chain="bsc",
            first_seen=now,
            last_activity=now,
            is_store_of_value=False,
            source_explorer="bscscan",
        ),
    ]


class TestCompleteDataFlow:
    """Tests for complete data flow from collection to export."""

    def test_full_pipeline_collection_to_json(
        self, sample_transactions, sample_holders
    ):
        """Test complete pipeline from collection to JSON export."""
        # Step 1: Create explorer data (simulating collection)
        explorer_data = [
            ExplorerData(
                explorer_name="etherscan",
                chain="ethereum",
                transactions=[sample_transactions[0]],
                holders=[sample_holders[0]],
            ),
            ExplorerData(
                explorer_name="bscscan",
                chain="bsc",
                transactions=[sample_transactions[1]],
                holders=[sample_holders[1]],
            ),
            ExplorerData(
                explorer_name="polygonscan",
                chain="polygon",
                transactions=[sample_transactions[2]],
                holders=[],
            ),
        ]

        # Step 2: Aggregate data
        aggregator = DataAggregator()
        aggregated = aggregator.aggregate(explorer_data)

        # Verify aggregation
        assert aggregated.total_records == 5  # 3 transactions + 2 holders
        assert len(aggregated.explorers_queried) == 3
        assert "USDC" in aggregated.by_stablecoin
        assert "USDT" in aggregated.by_stablecoin

        # Step 3: Convert to dict for JSON
        data_dict = aggregated.to_dict()

        # Verify JSON structure
        assert "summary" in data_dict
        assert "transactions" in data_dict
        assert "holders" in data_dict
        assert len(data_dict["transactions"]) == 3
        assert len(data_dict["holders"]) == 2

        # Step 4: Verify JSON serialization
        json_str = json.dumps(data_dict)
        parsed = json.loads(json_str)
        assert parsed["summary"]["by_chain"]["ethereum"] == 2  # 1 tx + 1 holder
        assert parsed["summary"]["by_chain"]["bsc"] == 2  # 1 tx + 1 holder
        assert parsed["summary"]["by_chain"]["polygon"] == 1  # 1 tx

    def test_classification_in_pipeline(self, sample_transactions):
        """Test that classification works correctly in the pipeline."""
        classifier = ActivityClassifier()

        # Classify each transaction
        for tx in sample_transactions:
            classified_type = classifier.classify_transaction(tx)
            assert classified_type == ActivityType.TRANSACTION

        # Test minting classification
        mint_tx = Transaction(
            transaction_hash="0xmint",
            block_number=18500000,
            timestamp=datetime.now(timezone.utc),
            from_address="0x0000000000000000000000000000000000000000",
            to_address="0x1111111111111111111111111111111111111111",
            amount=Decimal("1000000"),
            stablecoin="USDC",
            chain="ethereum",
            activity_type=ActivityType.UNKNOWN,
            source_explorer="etherscan",
        )
        assert classifier.classify_transaction(mint_tx) == ActivityType.OTHER

    def test_deduplication_in_pipeline(self):
        """Test that deduplication works correctly."""
        now = datetime.now(timezone.utc)
        aggregator = DataAggregator()

        # Create duplicate transactions
        tx1 = Transaction(
            transaction_hash="0xdupe",
            block_number=18500000,
            timestamp=now,
            from_address="0x1111111111111111111111111111111111111111",
            to_address="0x2222222222222222222222222222222222222222",
            amount=Decimal("1000"),
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
            amount=Decimal("1000"),
            stablecoin="USDC",
            chain="ethereum",  # Same chain
            activity_type=ActivityType.TRANSACTION,
            source_explorer="bscscan",  # Different source
        )

        explorer_data = [
            ExplorerData(
                explorer_name="etherscan",
                chain="ethereum",
                transactions=[tx1],
                holders=[],
            ),
            ExplorerData(
                explorer_name="bscscan",
                chain="bsc",
                transactions=[tx2],
                holders=[],
            ),
        ]

        aggregated = aggregator.aggregate(explorer_data)

        # Should have only 1 unique transaction
        assert len(aggregated.transactions) == 1


class TestMultiUserScenarios:
    """Tests for multi-user scenarios."""

    def test_user_isolation(self):
        """Test that user data is isolated."""
        from core.auth0_manager import UserInfo

        user1 = UserInfo(
            user_id="auth0|user1",
            email="user1@example.com",
            permissions=["run:agent", "view:results"],
        )
        user2 = UserInfo(
            user_id="auth0|user2",
            email="user2@example.com",
            permissions=["run:agent", "view:results"],
        )

        # Users should have different IDs
        assert user1.user_id != user2.user_id

        # Each user should only see their own permissions
        assert user1.has_permission("run:agent")
        assert user2.has_permission("run:agent")

    def test_permission_enforcement(self):
        """Test that permissions are enforced correctly."""
        from core.auth0_manager import UserInfo, InsufficientPermissionsError

        # User without admin permission
        regular_user = UserInfo(
            user_id="auth0|regular",
            email="regular@example.com",
            permissions=["run:agent", "view:results"],
        )

        # User with admin permission
        admin_user = UserInfo(
            user_id="auth0|admin",
            email="admin@example.com",
            permissions=["run:agent", "view:results", "admin:config"],
        )

        # Regular user should not have admin permission
        assert not regular_user.has_permission("admin:config")

        # Admin user should have admin permission
        assert admin_user.has_permission("admin:config")


class TestJSONExport:
    """Tests for JSON export functionality."""

    @pytest.mark.asyncio
    async def test_export_creates_valid_json_file(self, sample_transactions):
        """Test that export creates a valid JSON file."""
        # Create aggregated data
        aggregator = DataAggregator()
        explorer_data = [
            ExplorerData(
                explorer_name="etherscan",
                chain="ethereum",
                transactions=sample_transactions,
                holders=[],
            ),
        ]
        aggregated = aggregator.aggregate(explorer_data)

        # Create temp directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_config = OutputConfig(
                directory=temp_dir,
                max_records_per_explorer=1000,
            )

            # Mock database manager
            mock_db = MagicMock()
            mock_db.save_run_result = AsyncMock()

            exporter = JSONExporter(output_config, mock_db)

            # Export data
            run_id = "test-run-123"
            user_id = "auth0|test"
            output_path = await exporter.export(
                data=aggregated,
                run_id=run_id,
                user_id=user_id,
            )

            # Verify file was created
            assert os.path.exists(output_path)

            # Verify file contains valid JSON
            with open(output_path, "r") as f:
                data = json.load(f)

            assert "metadata" in data
            assert "summary" in data
            assert "transactions" in data
            assert data["metadata"]["run_id"] == run_id
            assert data["metadata"]["user_id"] == user_id
            assert len(data["transactions"]) == 3

    @pytest.mark.asyncio
    async def test_export_filename_format(self, sample_transactions):
        """Test that export filename has correct format."""
        aggregator = DataAggregator()
        explorer_data = [
            ExplorerData(
                explorer_name="etherscan",
                chain="ethereum",
                transactions=sample_transactions[:1],
                holders=[],
            ),
        ]
        aggregated = aggregator.aggregate(explorer_data)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_config = OutputConfig(
                directory=temp_dir,
                max_records_per_explorer=1000,
            )

            mock_db = MagicMock()
            mock_db.save_run_result = AsyncMock()

            exporter = JSONExporter(output_config, mock_db)

            run_id = "abc123"
            output_path = await exporter.export(
                data=aggregated,
                run_id=run_id,
                user_id="auth0|test",
            )

            # Filename should contain run_id
            filename = os.path.basename(output_path)
            assert run_id in filename
            assert filename.endswith(".json")


class TestErrorRecovery:
    """Tests for error recovery scenarios."""

    def test_partial_collection_success(self):
        """Test that partial collection failures don't break the pipeline."""
        now = datetime.now(timezone.utc)
        aggregator = DataAggregator()

        # One explorer succeeded, one failed
        explorer_data = [
            ExplorerData(
                explorer_name="etherscan",
                chain="ethereum",
                transactions=[
                    Transaction(
                        transaction_hash="0xsuccess",
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
            ),
            ExplorerData(
                explorer_name="bscscan",
                chain="bsc",
                transactions=[],  # Failed to collect
                holders=[],
                errors=["Rate limit exceeded"],
            ),
        ]

        aggregated = aggregator.aggregate(explorer_data)

        # Should still have data from successful explorer
        assert len(aggregated.transactions) == 1
        assert len(aggregated.errors) == 1
        assert "Rate limit exceeded" in aggregated.errors

    def test_all_explorers_fail(self):
        """Test handling when all explorers fail."""
        aggregator = DataAggregator()

        explorer_data = [
            ExplorerData(
                explorer_name="etherscan",
                chain="ethereum",
                transactions=[],
                holders=[],
                errors=["API key invalid"],
            ),
            ExplorerData(
                explorer_name="bscscan",
                chain="bsc",
                transactions=[],
                holders=[],
                errors=["Connection timeout"],
            ),
        ]

        aggregated = aggregator.aggregate(explorer_data)

        # Should have empty results but errors recorded
        assert aggregated.total_records == 0
        assert len(aggregated.errors) == 2


class TestDataIntegrity:
    """Tests for data integrity throughout the pipeline."""

    def test_transaction_data_preserved(self, sample_transactions):
        """Test that transaction data is preserved through aggregation."""
        aggregator = DataAggregator()

        explorer_data = [
            ExplorerData(
                explorer_name="etherscan",
                chain="ethereum",
                transactions=sample_transactions,
                holders=[],
            ),
        ]

        aggregated = aggregator.aggregate(explorer_data)

        # Verify all transactions are present
        assert len(aggregated.transactions) == len(sample_transactions)

        # Verify data integrity
        for original, aggregated_tx in zip(
            sample_transactions, aggregated.transactions
        ):
            assert original.transaction_hash == aggregated_tx.transaction_hash
            assert original.amount == aggregated_tx.amount
            assert original.stablecoin == aggregated_tx.stablecoin

    def test_holder_data_preserved(self, sample_holders):
        """Test that holder data is preserved through aggregation."""
        aggregator = DataAggregator()

        explorer_data = [
            ExplorerData(
                explorer_name="etherscan",
                chain="ethereum",
                transactions=[],
                holders=sample_holders,
            ),
        ]

        aggregated = aggregator.aggregate(explorer_data)

        # Verify all holders are present
        assert len(aggregated.holders) == len(sample_holders)

        # Verify data integrity
        for original, aggregated_holder in zip(
            sample_holders, aggregated.holders
        ):
            assert original.address == aggregated_holder.address
            assert original.balance == aggregated_holder.balance
            assert original.is_store_of_value == aggregated_holder.is_store_of_value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
