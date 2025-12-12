"""Activity classifier for stablecoin transactions."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from collectors.models import ActivityType, Holder, Transaction


# Zero address used for minting/burning operations
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

# Store of value threshold in days
STORE_OF_VALUE_THRESHOLD_DAYS = 30


class ActivityClassifier:
    """Classifies stablecoin transaction activity types.

    Classification Logic (from design doc):
    - Transaction: Transfer with both sender and receiver, amount > 0
    - Store of Value: Address holds tokens > 30 days without outgoing
    - Other: Minting, burning, contract interactions, unclassifiable
    """

    def classify_transaction(self, tx: Transaction) -> ActivityType:
        """Classify a transaction based on from/to addresses and amount.

        Args:
            tx: The transaction to classify.

        Returns:
            The classified activity type.

        Classification rules:
        - If from_address is zero address: OTHER (minting)
        - If to_address is zero address: OTHER (burning)
        - If amount is 0: OTHER (contract interaction or approval)
        - Otherwise: TRANSACTION (regular transfer)
        """
        # Check for minting (from zero address)
        if tx.from_address.lower() == ZERO_ADDRESS:
            return ActivityType.OTHER

        # Check for burning (to zero address)
        if tx.to_address.lower() == ZERO_ADDRESS:
            return ActivityType.OTHER

        # Check for zero amount (likely approval or contract interaction)
        if tx.amount == Decimal("0"):
            return ActivityType.OTHER

        # Regular transfer with valid sender, receiver, and amount
        return ActivityType.TRANSACTION

    def identify_store_of_value(
        self, holder: Holder, transactions: list[Transaction]
    ) -> bool:
        """Determine if a holder is using stablecoins as store of value.

        A holder is classified as store of value if they have held
        tokens for more than 30 days without any outgoing transfers.

        Args:
            holder: The holder to evaluate.
            transactions: List of transactions to analyze for the holder.

        Returns:
            True if holder qualifies as store of value, False otherwise.
        """
        holding_period = self.calculate_holding_period(
            holder.address, transactions
        )
        return holding_period >= STORE_OF_VALUE_THRESHOLD_DAYS

    def calculate_holding_period(
        self,
        address: str,
        transactions: list[Transaction],
        reference_time: Optional[datetime] = None,
    ) -> int:
        """Calculate the number of days since last outgoing transfer.

        Args:
            address: The wallet address to check.
            transactions: List of transactions to analyze.
            reference_time: Reference time to calculate from (default: now).

        Returns:
            Number of days since last outgoing transfer, or days since
            first incoming transfer if no outgoing transfers exist.
        """
        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        address_lower = address.lower()

        # Find all outgoing transactions for this address
        outgoing_txs = [
            tx for tx in transactions
            if tx.from_address.lower() == address_lower
        ]

        if outgoing_txs:
            # Find the most recent outgoing transaction
            last_outgoing = max(outgoing_txs, key=lambda tx: tx.timestamp)
            delta = reference_time - last_outgoing.timestamp
            return delta.days

        # No outgoing transactions - check for incoming transactions
        incoming_txs = [
            tx for tx in transactions
            if tx.to_address.lower() == address_lower
        ]

        if incoming_txs:
            # Find the first incoming transaction
            first_incoming = min(incoming_txs, key=lambda tx: tx.timestamp)
            delta = reference_time - first_incoming.timestamp
            return delta.days

        # No transactions found for this address
        return 0

    def classify_holder(
        self, holder: Holder, transactions: list[Transaction]
    ) -> Holder:
        """Classify a holder and update their is_store_of_value flag.

        Args:
            holder: The holder to classify.
            transactions: List of transactions to analyze.

        Returns:
            The holder with updated is_store_of_value flag.
        """
        is_sov = self.identify_store_of_value(holder, transactions)
        holder.is_store_of_value = is_sov
        return holder
