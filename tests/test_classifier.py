"""Unit tests for activity classification logic."""

from datetime import datetime, timezone, timedelta
from decimal import Decimal

import pytest

from collectors.classifier import (
    ActivityClassifier,
    ZERO_ADDRESS,
    STORE_OF_VALUE_THRESHOLD_DAYS,
)
from collectors.models import ActivityType, Transaction, Holder


@pytest.fixture
def classifier():
    """Create an ActivityClassifier instance."""
    return ActivityClassifier()


@pytest.fixture
def base_transaction():
    """Create a base transaction for testing."""
    return Transaction(
        transaction_hash="0x123abc",
        block_number=18500000,
        timestamp=datetime.now(timezone.utc),
        from_address="0x1111111111111111111111111111111111111111",
        to_address="0x2222222222222222222222222222222222222222",
        amount=Decimal("1000.50"),
        stablecoin="USDC",
        chain="ethereum",
        activity_type=ActivityType.UNKNOWN,
        source_explorer="etherscan",
    )


class TestClassifyTransaction:
    """Tests for classify_transaction method."""

    def test_regular_transfer_classified_as_transaction(
        self, classifier, base_transaction
    ):
        """Regular transfer with valid sender, receiver, and amount."""
        result = classifier.classify_transaction(base_transaction)
        assert result == ActivityType.TRANSACTION

    def test_minting_classified_as_other(self, classifier):
        """Transfer from zero address (minting) classified as OTHER."""
        tx = Transaction(
            transaction_hash="0xmint123",
            block_number=18500000,
            timestamp=datetime.now(timezone.utc),
            from_address=ZERO_ADDRESS,
            to_address="0x2222222222222222222222222222222222222222",
            amount=Decimal("1000000"),
            stablecoin="USDT",
            chain="ethereum",
            activity_type=ActivityType.UNKNOWN,
            source_explorer="etherscan",
        )
        result = classifier.classify_transaction(tx)
        assert result == ActivityType.OTHER

    def test_burning_classified_as_other(self, classifier):
        """Transfer to zero address (burning) classified as OTHER."""
        tx = Transaction(
            transaction_hash="0xburn123",
            block_number=18500000,
            timestamp=datetime.now(timezone.utc),
            from_address="0x1111111111111111111111111111111111111111",
            to_address=ZERO_ADDRESS,
            amount=Decimal("500000"),
            stablecoin="USDC",
            chain="bsc",
            activity_type=ActivityType.UNKNOWN,
            source_explorer="bscscan",
        )
        result = classifier.classify_transaction(tx)
        assert result == ActivityType.OTHER

    def test_zero_amount_classified_as_other(self, classifier):
        """Zero amount transfer (approval) classified as OTHER."""
        tx = Transaction(
            transaction_hash="0xapprove123",
            block_number=18500000,
            timestamp=datetime.now(timezone.utc),
            from_address="0x1111111111111111111111111111111111111111",
            to_address="0x2222222222222222222222222222222222222222",
            amount=Decimal("0"),
            stablecoin="USDC",
            chain="polygon",
            activity_type=ActivityType.UNKNOWN,
            source_explorer="polygonscan",
        )
        result = classifier.classify_transaction(tx)
        assert result == ActivityType.OTHER

    def test_case_insensitive_zero_address_check(self, classifier):
        """Zero address check is case-insensitive."""
        # Uppercase zero address
        tx = Transaction(
            transaction_hash="0xmint456",
            block_number=18500000,
            timestamp=datetime.now(timezone.utc),
            from_address="0x0000000000000000000000000000000000000000".upper(),
            to_address="0x2222222222222222222222222222222222222222",
            amount=Decimal("1000"),
            stablecoin="USDC",
            chain="ethereum",
            activity_type=ActivityType.UNKNOWN,
            source_explorer="etherscan",
        )
        result = classifier.classify_transaction(tx)
        assert result == ActivityType.OTHER


class TestCalculateHoldingPeriod:
    """Tests for calculate_holding_period method."""

    def test_no_transactions_returns_zero(self, classifier):
        """No transactions for address returns 0 days."""
        result = classifier.calculate_holding_period(
            "0xunknown", [], datetime.now(timezone.utc)
        )
        assert result == 0

    def test_holding_period_from_last_outgoing(self, classifier):
        """Holding period calculated from last outgoing transaction."""
        address = "0x1111111111111111111111111111111111111111"
        reference_time = datetime.now(timezone.utc)
        last_outgoing_time = reference_time - timedelta(days=45)

        transactions = [
            Transaction(
                transaction_hash="0xout1",
                block_number=18500000,
                timestamp=last_outgoing_time,
                from_address=address,
                to_address="0x2222222222222222222222222222222222222222",
                amount=Decimal("100"),
                stablecoin="USDC",
                chain="ethereum",
                activity_type=ActivityType.TRANSACTION,
                source_explorer="etherscan",
            ),
        ]

        result = classifier.calculate_holding_period(
            address, transactions, reference_time
        )
        assert result == 45

    def test_holding_period_from_first_incoming_when_no_outgoing(
        self, classifier
    ):
        """When no outgoing, use first incoming transaction."""
        address = "0x1111111111111111111111111111111111111111"
        reference_time = datetime.now(timezone.utc)
        first_incoming_time = reference_time - timedelta(days=60)

        transactions = [
            Transaction(
                transaction_hash="0xin1",
                block_number=18500000,
                timestamp=first_incoming_time,
                from_address="0x2222222222222222222222222222222222222222",
                to_address=address,
                amount=Decimal("1000"),
                stablecoin="USDC",
                chain="ethereum",
                activity_type=ActivityType.TRANSACTION,
                source_explorer="etherscan",
            ),
        ]

        result = classifier.calculate_holding_period(
            address, transactions, reference_time
        )
        assert result == 60

    def test_case_insensitive_address_matching(self, classifier):
        """Address matching is case-insensitive."""
        address = "0xABCDEF1234567890ABCDEF1234567890ABCDEF12"
        reference_time = datetime.now(timezone.utc)
        tx_time = reference_time - timedelta(days=20)

        transactions = [
            Transaction(
                transaction_hash="0xout1",
                block_number=18500000,
                timestamp=tx_time,
                from_address=address.lower(),  # lowercase in transaction
                to_address="0x2222222222222222222222222222222222222222",
                amount=Decimal("100"),
                stablecoin="USDC",
                chain="ethereum",
                activity_type=ActivityType.TRANSACTION,
                source_explorer="etherscan",
            ),
        ]

        # Query with uppercase
        result = classifier.calculate_holding_period(
            address.upper(), transactions, reference_time
        )
        assert result == 20


class TestIdentifyStoreOfValue:
    """Tests for identify_store_of_value method."""

    def test_holder_with_long_holding_period_is_sov(self, classifier):
        """Holder with > 30 days holding is store of value."""
        address = "0x1111111111111111111111111111111111111111"
        reference_time = datetime.now(timezone.utc)

        holder = Holder(
            address=address,
            balance=Decimal("50000"),
            stablecoin="USDC",
            chain="ethereum",
            first_seen=reference_time - timedelta(days=60),
            last_activity=reference_time - timedelta(days=35),
            is_store_of_value=False,
            source_explorer="etherscan",
        )

        # Last outgoing was 35 days ago
        transactions = [
            Transaction(
                transaction_hash="0xout1",
                block_number=18500000,
                timestamp=reference_time - timedelta(days=35),
                from_address=address,
                to_address="0x2222222222222222222222222222222222222222",
                amount=Decimal("100"),
                stablecoin="USDC",
                chain="ethereum",
                activity_type=ActivityType.TRANSACTION,
                source_explorer="etherscan",
            ),
        ]

        # Patch the reference time for consistent testing
        result = classifier.identify_store_of_value(holder, transactions)
        assert result is True

    def test_holder_with_short_holding_period_is_not_sov(self, classifier):
        """Holder with < 30 days holding is not store of value."""
        address = "0x1111111111111111111111111111111111111111"
        reference_time = datetime.now(timezone.utc)

        holder = Holder(
            address=address,
            balance=Decimal("50000"),
            stablecoin="USDC",
            chain="ethereum",
            first_seen=reference_time - timedelta(days=20),
            last_activity=reference_time - timedelta(days=5),
            is_store_of_value=False,
            source_explorer="etherscan",
        )

        # Last outgoing was 5 days ago
        transactions = [
            Transaction(
                transaction_hash="0xout1",
                block_number=18500000,
                timestamp=reference_time - timedelta(days=5),
                from_address=address,
                to_address="0x2222222222222222222222222222222222222222",
                amount=Decimal("100"),
                stablecoin="USDC",
                chain="ethereum",
                activity_type=ActivityType.TRANSACTION,
                source_explorer="etherscan",
            ),
        ]

        result = classifier.identify_store_of_value(holder, transactions)
        assert result is False

    def test_exactly_30_days_is_sov(self, classifier):
        """Holder with exactly 30 days holding is store of value."""
        address = "0x1111111111111111111111111111111111111111"
        reference_time = datetime.now(timezone.utc)

        holder = Holder(
            address=address,
            balance=Decimal("50000"),
            stablecoin="USDC",
            chain="ethereum",
            first_seen=reference_time - timedelta(days=30),
            last_activity=reference_time - timedelta(days=30),
            is_store_of_value=False,
            source_explorer="etherscan",
        )

        # Last outgoing was exactly 30 days ago
        transactions = [
            Transaction(
                transaction_hash="0xout1",
                block_number=18500000,
                timestamp=reference_time - timedelta(days=30),
                from_address=address,
                to_address="0x2222222222222222222222222222222222222222",
                amount=Decimal("100"),
                stablecoin="USDC",
                chain="ethereum",
                activity_type=ActivityType.TRANSACTION,
                source_explorer="etherscan",
            ),
        ]

        result = classifier.identify_store_of_value(holder, transactions)
        assert result is True


class TestClassifyHolder:
    """Tests for classify_holder method."""

    def test_classify_holder_updates_flag(self, classifier):
        """classify_holder updates the is_store_of_value flag."""
        address = "0x1111111111111111111111111111111111111111"
        reference_time = datetime.now(timezone.utc)

        holder = Holder(
            address=address,
            balance=Decimal("50000"),
            stablecoin="USDC",
            chain="ethereum",
            first_seen=reference_time - timedelta(days=60),
            last_activity=reference_time - timedelta(days=35),
            is_store_of_value=False,  # Initially False
            source_explorer="etherscan",
        )

        transactions = [
            Transaction(
                transaction_hash="0xout1",
                block_number=18500000,
                timestamp=reference_time - timedelta(days=35),
                from_address=address,
                to_address="0x2222222222222222222222222222222222222222",
                amount=Decimal("100"),
                stablecoin="USDC",
                chain="ethereum",
                activity_type=ActivityType.TRANSACTION,
                source_explorer="etherscan",
            ),
        ]

        result = classifier.classify_holder(holder, transactions)
        assert result.is_store_of_value is True
        assert result is holder  # Same object, modified in place


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
