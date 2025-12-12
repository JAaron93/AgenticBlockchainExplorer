"""
Property-based tests for the stablecoin analysis notebook.

These tests use Hypothesis to verify correctness properties defined in the design document.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from decimal import Decimal
from datetime import datetime, timezone, timedelta
import json
import sys
import os

# Add the notebooks directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks'))

from stablecoin_validation import validate_schema


# =============================================================================
# Test Data Strategies
# =============================================================================

# Valid enum values
STABLECOINS = ["USDC", "USDT"]
CHAINS = ["ethereum", "bsc", "polygon"]
ACTIVITY_TYPES = ["transaction", "store_of_value", "other"]


def valid_address():
    """Generate valid Ethereum-style addresses."""
    return st.text(
        alphabet="0123456789abcdef",
        min_size=40,
        max_size=40
    ).map(lambda x: "0x" + x)


def valid_tx_hash():
    """Generate valid transaction hashes."""
    return st.text(
        alphabet="0123456789abcdef",
        min_size=64,
        max_size=64
    ).map(lambda x: "0x" + x)


def valid_decimal_string():
    """Generate valid decimal strings for amounts/balances."""
    return st.decimals(
        min_value=Decimal("0"),
        max_value=Decimal("1000000000"),
        allow_nan=False,
        allow_infinity=False,
        places=6
    ).map(str)


def valid_timestamp():
    """Generate valid ISO8601 timestamps."""
    return st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2025, 12, 31),
        timezones=st.just(timezone.utc)
    ).map(lambda dt: dt.isoformat().replace('+00:00', 'Z'))


@st.composite
def valid_transaction(draw):
    """Generate a valid transaction record."""
    return {
        "transaction_hash": draw(valid_tx_hash()),
        "block_number": draw(st.integers(min_value=0, max_value=100000000)),
        "timestamp": draw(valid_timestamp()),
        "from_address": draw(valid_address()),
        "to_address": draw(valid_address()),
        "amount": draw(valid_decimal_string()),
        "stablecoin": draw(st.sampled_from(STABLECOINS)),
        "chain": draw(st.sampled_from(CHAINS)),
        "activity_type": draw(st.sampled_from(ACTIVITY_TYPES)),
        "source_explorer": draw(st.sampled_from(["etherscan", "bscscan", "polygonscan"])),
        "gas_used": draw(st.one_of(st.none(), st.integers(min_value=0, max_value=1000000))),
        "gas_price": draw(st.one_of(st.none(), valid_decimal_string())),
    }


@st.composite
def valid_holder(draw):
    """Generate a valid holder record."""
    first_seen = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2025, 6, 30),
        timezones=st.just(timezone.utc)
    ))
    # Ensure last_activity >= first_seen by adding a non-negative timedelta
    days_after = draw(st.integers(min_value=0, max_value=365))
    last_activity = first_seen + timedelta(days=days_after)

    return {
        "address": draw(valid_address()),
        "balance": draw(valid_decimal_string()),
        "stablecoin": draw(st.sampled_from(STABLECOINS)),
        "chain": draw(st.sampled_from(CHAINS)),
        "first_seen": first_seen.isoformat().replace('+00:00', 'Z'),
        "last_activity": last_activity.isoformat().replace('+00:00', 'Z'),
        "is_store_of_value": draw(st.booleans()),
        "source_explorer": draw(st.sampled_from(["etherscan", "bscscan", "polygonscan"])),
    }


@st.composite
def valid_metadata(draw):
    """Generate valid metadata."""
    return {
        "run_id": draw(st.uuids()).hex,
        "collection_timestamp": draw(valid_timestamp()),
        "agent_version": "1.0.0",
        "explorers_queried": draw(st.lists(
            st.sampled_from(["etherscan", "bscscan", "polygonscan"]),
            min_size=1,
            max_size=3,
            unique=True
        )),
        "total_records": draw(st.integers(min_value=0, max_value=10000)),
    }


@st.composite
def valid_summary(draw):
    """Generate valid summary."""
    return {
        "by_stablecoin": {
            "USDC": {"transaction_count": draw(st.integers(min_value=0)), "total_volume": "0"},
            "USDT": {"transaction_count": draw(st.integers(min_value=0)), "total_volume": "0"},
        },
        "by_activity_type": {
            "transaction": draw(st.integers(min_value=0)),
            "store_of_value": draw(st.integers(min_value=0)),
            "other": draw(st.integers(min_value=0)),
        },
        "by_chain": {
            "ethereum": draw(st.integers(min_value=0)),
            "bsc": draw(st.integers(min_value=0)),
            "polygon": draw(st.integers(min_value=0)),
        },
    }


@st.composite
def valid_json_export(draw):
    """Generate a complete valid JSON export structure."""
    transactions = draw(st.lists(valid_transaction(), min_size=0, max_size=10))
    holders = draw(st.lists(valid_holder(), min_size=0, max_size=10))

    metadata = draw(valid_metadata())
    metadata["total_records"] = len(transactions) + len(holders)

    return {
        "metadata": metadata,
        "summary": draw(valid_summary()),
        "transactions": transactions,
        "holders": holders,
    }


# =============================================================================
# Property Tests
# =============================================================================

class TestSchemaValidation:
    """Tests for schema validation functions."""

    @settings(max_examples=100)
    @given(data=valid_json_export())
    def test_property_1_schema_validation_round_trip(self, data):
        """
        **Feature: stablecoin-analysis-notebook, Property 1: Schema validation round-trip**

        For any valid JSON data structure (real or generated), parsing and
        validation SHALL succeed and preserve all required fields without data loss.

        **Validates: Requirements 1.2, 8.2**
        """
        # Validate the generated data
        is_valid, errors = validate_schema(data)

        # Assert validation passes for valid data
        assert is_valid, f"Valid data should pass validation. Errors: {errors}"
        assert errors == [], f"No errors expected for valid data. Got: {errors}"

        # Verify all required fields are preserved
        assert "metadata" in data
        assert "summary" in data
        assert "transactions" in data
        assert "holders" in data

        # Verify metadata fields
        assert "run_id" in data["metadata"]
        assert "collection_timestamp" in data["metadata"]
        assert "agent_version" in data["metadata"]
        assert "explorers_queried" in data["metadata"]
        assert "total_records" in data["metadata"]

        # Verify transaction fields if any transactions exist
        for tx in data["transactions"]:
            assert "transaction_hash" in tx
            assert "timestamp" in tx
            assert "amount" in tx
            assert "stablecoin" in tx
            assert "chain" in tx
            assert "activity_type" in tx

        # Verify holder fields if any holders exist
        for holder in data["holders"]:
            assert "address" in holder
            assert "balance" in holder
            assert "stablecoin" in holder
            assert "chain" in holder
            assert "first_seen" in holder
            assert "last_activity" in holder
            assert "is_store_of_value" in holder
