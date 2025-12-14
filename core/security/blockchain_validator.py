"""Blockchain data validation for the stablecoin explorer.

This module provides validation for blockchain-specific data formats
including addresses, transaction hashes, amounts, timestamps, and
block numbers.

Requirements: 4.1, 4.2, 4.3, 4.5, 4.6, 4.7
"""

import logging
import re
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Optional

logger = logging.getLogger(__name__)


class BlockchainValidationError(Exception):
    """Raised when blockchain data validation fails."""

    pass


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    is_valid: bool
    field_name: str
    error_message: Optional[str] = None


class BlockchainDataValidator:
    """Validates blockchain data formats (addresses, hashes, amounts).

    This class provides validation methods for blockchain-specific data
    to ensure data integrity and prevent injection attacks.

    Requirements:
    - 4.1: Address validation (^0x[a-fA-F0-9]{40}$)
    - 4.2: Transaction hash validation (^0x[a-fA-F0-9]{64}$)
    - 4.3: Amount validation (unsigned decimal, max 2^256-1)
    - 4.5: Address normalization to lowercase
    - 4.6: Timestamp validation (genesis to current time)
    - 4.7: Block number validation (positive, not too far in future)
    """

    # Pre-compiled regex patterns for security (anchored, no backtracking)
    ADDRESS_PATTERN = re.compile(r"^0x[a-fA-F0-9]{40}$")
    TX_HASH_PATTERN = re.compile(r"^0x[a-fA-F0-9]{64}$")
    # Amount: digits with optional decimal point and up to 18 fractional digits
    AMOUNT_PATTERN = re.compile(r"^[0-9]+(\.[0-9]{1,18})?$")

    # Maximum amount value (2^256 - 1) - the maximum value for uint256
    MAX_AMOUNT = 2**256 - 1

    # Genesis timestamps for supported chains (Unix timestamps)
    GENESIS_TIMESTAMPS = {
        "ethereum": 1438269973,  # July 30, 2015 (block 0)
        "bsc": 1598671449,  # August 29, 2020 (block 0)
        "polygon": 1590824836,  # May 30, 2020 (block 0)
    }

    # Buffer for block number validation (allow up to 1000 blocks in future)
    BLOCK_NUMBER_FUTURE_BUFFER = 1000

    def validate_address(self, address: str) -> bool:
        """Validate Ethereum-style address format.

        Args:
            address: The address string to validate

        Returns:
            True if address matches ^0x[a-fA-F0-9]{40}$, False otherwise

        Requirements: 4.1
        """
        if not isinstance(address, str):
            return False
        return bool(self.ADDRESS_PATTERN.match(address))

    def validate_tx_hash(self, tx_hash: str) -> bool:
        """Validate transaction hash format.

        Args:
            tx_hash: The transaction hash string to validate

        Returns:
            True if hash matches pattern ^0x[a-fA-F0-9]{64}$, False otherwise

        Requirements: 4.2
        """
        if not isinstance(tx_hash, str):
            return False
        return bool(self.TX_HASH_PATTERN.match(tx_hash))

    def validate_amount(self, amount: str) -> bool:
        """Validate numeric amount format and bounds.

        Validates that the amount:
        - Matches pattern ^[0-9]+(\\.[0-9]{1,18})?$
        - Does not use scientific notation
        - Is not negative
        - Does not exceed 2^256-1

        Args:
            amount: The amount string to validate

        Returns:
            True if amount is valid, False otherwise

        Requirements: 4.3
        """
        if not isinstance(amount, str):
            return False

        # Check pattern match first
        if not self.AMOUNT_PATTERN.match(amount):
            return False

        # Parse and check bounds
        try:
            # Use Decimal for precise parsing
            value = Decimal(amount)

            # Check for negative (should not happen with pattern, but be safe)
            if value < 0:
                return False

            # Check upper bound (2^256 - 1)
            if value > self.MAX_AMOUNT:
                return False

            return True
        except (InvalidOperation, ValueError):
            return False

    def validate_timestamp(self, timestamp: int, chain: str) -> bool:
        """Validate timestamp is within reasonable bounds for chain.

        Args:
            timestamp: Unix timestamp to validate
            chain: Blockchain name (ethereum, bsc, polygon)

        Returns:
            True if timestamp is valid (after genesis, not in future),
            False otherwise

        Requirements: 4.6
        """
        if not isinstance(timestamp, int):
            return False

        # Get genesis timestamp for chain
        genesis = self.GENESIS_TIMESTAMPS.get(chain.lower())
        if genesis is None:
            # Unknown chain - log warning and reject
            logger.warning(f"Unknown chain for timestamp validation: {chain}")
            return False

        # Check not before genesis
        if timestamp < genesis:
            return False

        # Check not in future (with small buffer for clock skew)
        current_time = int(time.time())
        # Allow 5 minutes of clock skew
        if timestamp > current_time + 300:
            return False

        return True

    def validate_block_number(
        self, block_number: int, max_known: Optional[int] = None
    ) -> bool:
        """Validate block number is positive and not too far in future.

        Args:
            block_number: Block number to validate
            max_known: Maximum known block number (optional)

        Returns:
            True if block number is valid, False otherwise

        Requirements: 4.7
        """
        if not isinstance(block_number, int):
            return False

        # Must be positive (block 0 is genesis, so >= 0)
        if block_number < 0:
            return False

        # If max_known provided, check not too far in future
        if max_known is not None:
            if block_number > max_known + self.BLOCK_NUMBER_FUTURE_BUFFER:
                return False

        return True

    def normalize_address(self, address: str) -> str:
        """Normalize address to lowercase.

        This prevents case-based duplicates since Ethereum addresses
        are case-insensitive (except for EIP-55 checksum).

        Args:
            address: The address to normalize

        Returns:
            Lowercase address string

        Requirements: 4.5
        """
        if not isinstance(address, str):
            raise BlockchainValidationError(
                "Address must be a string"
            )
        return address.lower()

    def validate_and_normalize_address(self, address: str) -> Optional[str]:
        """Validate and normalize an address in one operation.

        Args:
            address: The address to validate and normalize

        Returns:
            Normalized (lowercase) address if valid, None otherwise
        """
        if self.validate_address(address):
            return self.normalize_address(address)
        return None

    def validate_record(
        self,
        address: Optional[str] = None,
        tx_hash: Optional[str] = None,
        amount: Optional[str] = None,
        timestamp: Optional[int] = None,
        block_number: Optional[int] = None,
        chain: Optional[str] = None,
        max_known_block: Optional[int] = None,
    ) -> list[ValidationResult]:
        """Validate multiple fields of a blockchain record.

        This is a convenience method for validating multiple fields at once.
        Invalid fields are logged with field name only (not the invalid value)
        per Requirement 4.4.

        Args:
            address: Optional address to validate
            tx_hash: Optional transaction hash to validate
            amount: Optional amount to validate
            timestamp: Optional timestamp to validate (requires chain)
            block_number: Optional block number to validate
            chain: Chain name for timestamp validation
            max_known_block: Max known block for block number validation

        Returns:
            List of ValidationResult objects for each field checked
        """
        results = []

        if address is not None:
            is_valid = self.validate_address(address)
            err = None if is_valid else "Invalid address format"
            results.append(
                ValidationResult(is_valid=is_valid, field_name="address",
                                 error_message=err)
            )
            if not is_valid:
                logger.warning("Validation failed for field: address")

        if tx_hash is not None:
            is_valid = self.validate_tx_hash(tx_hash)
            err = None if is_valid else "Invalid transaction hash format"
            results.append(
                ValidationResult(is_valid=is_valid, field_name="tx_hash",
                                 error_message=err)
            )
            if not is_valid:
                logger.warning("Validation failed for field: tx_hash")

        if amount is not None:
            is_valid = self.validate_amount(amount)
            err = None if is_valid else "Invalid amount format or bounds"
            results.append(
                ValidationResult(is_valid=is_valid, field_name="amount",
                                 error_message=err)
            )
            if not is_valid:
                logger.warning("Validation failed for field: amount")

        if timestamp is not None:
            if chain is None:
                err_msg = "Chain required for timestamp validation"
                results.append(
                    ValidationResult(
                        is_valid=False,
                        field_name="timestamp",
                        error_message=err_msg,
                    )
                )
                logger.warning(
                    "Validation failed for field: timestamp (no chain)"
                )
            else:
                is_valid = self.validate_timestamp(timestamp, chain)
                err = None if is_valid else "Invalid timestamp bounds"
                results.append(
                    ValidationResult(is_valid=is_valid, field_name="timestamp",
                                     error_message=err)
                )
                if not is_valid:
                    logger.warning("Validation failed for field: timestamp")

        if block_number is not None:
            is_valid = self.validate_block_number(
                block_number, max_known_block
            )
            err = None if is_valid else "Invalid block number"
            results.append(
                ValidationResult(
                    is_valid=is_valid,
                    field_name="block_number",
                    error_message=err
                )
            )
            if not is_valid:
                logger.warning("Validation failed for field: block_number")

        return results
