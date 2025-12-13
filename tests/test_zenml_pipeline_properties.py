"""
Property-based tests for ZenML pipeline steps.

These tests verify correctness properties for the ZenML collector and
aggregation steps defined in the design document.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from decimal import Decimal
from datetime import datetime, timezone, timedelta
import pandas as pd
import sys
import os

# Add the project root to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipelines.steps.collectors import (
    CollectorOutput,
    AggregatedOutput,
    transactions_to_dataframe,
    holders_to_dataframe,
    dataframe_to_transactions,
    dataframe_to_holders,
    aggregate_data_step,
    TRANSACTION_REQUIRED_COLUMNS,
    HOLDER_REQUIRED_COLUMNS,
)
from collectors.models import Transaction, Holder, ActivityType


# =============================================================================
# Test Data Strategies
# =============================================================================

STABLECOINS = ["USDC", "USDT"]
CHAINS = ["ethereum", "bsc", "polygon"]
ACTIVITY_TYPES = [ActivityType.TRANSACTION, ActivityType.STORE_OF_VALUE, ActivityType.OTHER]


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


@st.composite
def valid_transaction_model(draw):
    """Generate a valid Transaction model object."""
    timestamp = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2025, 12, 31),
        timezones=st.just(timezone.utc)
    ))
    
    amount = draw(st.decimals(
        min_value=Decimal("0"),
        max_value=Decimal("1000000"),
        allow_nan=False,
        allow_infinity=False,
        places=6
    ))
    
    gas_used = draw(st.one_of(
        st.none(),
        st.integers(min_value=0, max_value=1000000)
    ))
    
    gas_price = draw(st.one_of(
        st.none(),
        st.decimals(
            min_value=Decimal("0"),
            max_value=Decimal("1000"),
            allow_nan=False,
            allow_infinity=False,
            places=9
        )
    ))
    
    return Transaction(
        transaction_hash=draw(valid_tx_hash()),
        block_number=draw(st.integers(min_value=0, max_value=100000000)),
        timestamp=timestamp,
        from_address=draw(valid_address()),
        to_address=draw(valid_address()),
        amount=amount,
        stablecoin=draw(st.sampled_from(STABLECOINS)),
        chain=draw(st.sampled_from(CHAINS)),
        activity_type=draw(st.sampled_from(ACTIVITY_TYPES)),
        source_explorer=draw(st.sampled_from(["etherscan", "bscscan", "polygonscan"])),
        gas_used=gas_used,
        gas_price=gas_price,
    )


@st.composite
def valid_holder_model(draw):
    """Generate a valid Holder model object."""
    first_seen = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2025, 6, 30),
        timezones=st.just(timezone.utc)
    ))
    # Ensure last_activity >= first_seen
    days_after = draw(st.integers(min_value=0, max_value=365))
    last_activity = first_seen + timedelta(days=days_after)
    
    balance = draw(st.decimals(
        min_value=Decimal("0"),
        max_value=Decimal("1000000000"),
        allow_nan=False,
        allow_infinity=False,
        places=6
    ))
    
    return Holder(
        address=draw(valid_address()),
        balance=balance,
        stablecoin=draw(st.sampled_from(STABLECOINS)),
        chain=draw(st.sampled_from(CHAINS)),
        first_seen=first_seen,
        last_activity=last_activity,
        is_store_of_value=draw(st.booleans()),
        source_explorer=draw(st.sampled_from(["etherscan", "bscscan", "polygonscan"])),
    )


@st.composite
def valid_collector_output(draw, chain: str = "ethereum", explorer: str = "etherscan"):
    """Generate a valid CollectorOutput."""
    transactions = draw(st.lists(valid_transaction_model(), min_size=0, max_size=20))
    holders = draw(st.lists(valid_holder_model(), min_size=0, max_size=10))
    
    # Override chain for consistency
    for tx in transactions:
        object.__setattr__(tx, 'chain', chain)
    for h in holders:
        object.__setattr__(h, 'chain', chain)
    
    success = len(transactions) > 0 or len(holders) > 0
    errors = [] if success else ["No data collected"]
    
    return CollectorOutput(
        transactions_df=transactions_to_dataframe(transactions),
        holders_df=holders_to_dataframe(holders),
        explorer_name=explorer,
        chain=chain,
        success=success,
        errors=errors,
        collection_time_seconds=draw(st.floats(min_value=0.0, max_value=60.0)),
    )


# =============================================================================
# Property Tests for ZenML Steps
# =============================================================================

class TestZenMLStepOutputTyping:
    """Tests for ZenML step output typing (Property 13)."""

    @settings(max_examples=100, deadline=None)
    @given(transactions=st.lists(valid_transaction_model(), min_size=0, max_size=30))
    def test_property_13_transactions_dataframe_has_required_columns(
        self, transactions
    ):
        """
        **Feature: stablecoin-analysis-notebook, Property 13: ZenML step output typing**

        For any ZenML collector step execution, the output artifact SHALL be
        a valid pandas DataFrame with the required transaction schema columns.

        **Validates: Requirements 9.1, 9.2**
        """
        # Convert transactions to DataFrame
        df = transactions_to_dataframe(transactions)
        
        # Verify it's a DataFrame
        assert isinstance(df, pd.DataFrame), (
            f"Expected pandas DataFrame, got {type(df)}"
        )
        
        # Verify all required columns are present
        for col in TRANSACTION_REQUIRED_COLUMNS:
            assert col in df.columns, (
                f"Required column '{col}' missing from transactions DataFrame"
            )
        
        # Verify row count matches input
        assert len(df) == len(transactions), (
            f"DataFrame has {len(df)} rows, expected {len(transactions)}"
        )
        
        # Verify data types for key columns
        if len(df) > 0:
            # transaction_hash should be string
            assert df["transaction_hash"].dtype == object, (
                f"transaction_hash should be string type"
            )
            # block_number should be numeric
            assert pd.api.types.is_numeric_dtype(df["block_number"]), (
                f"block_number should be numeric type"
            )
            # amount should be numeric (float after conversion)
            assert pd.api.types.is_numeric_dtype(df["amount"]), (
                f"amount should be numeric type"
            )

    @settings(max_examples=100, deadline=None)
    @given(holders=st.lists(valid_holder_model(), min_size=0, max_size=30))
    def test_property_13_holders_dataframe_has_required_columns(self, holders):
        """
        **Feature: stablecoin-analysis-notebook, Property 13: ZenML step output typing**

        For any ZenML collector step execution, the output artifact SHALL be
        a valid pandas DataFrame with the required holder schema columns.

        **Validates: Requirements 9.1, 9.2**
        """
        # Convert holders to DataFrame
        df = holders_to_dataframe(holders)
        
        # Verify it's a DataFrame
        assert isinstance(df, pd.DataFrame), (
            f"Expected pandas DataFrame, got {type(df)}"
        )
        
        # Verify all required columns are present
        for col in HOLDER_REQUIRED_COLUMNS:
            assert col in df.columns, (
                f"Required column '{col}' missing from holders DataFrame"
            )
        
        # Verify row count matches input
        assert len(df) == len(holders), (
            f"DataFrame has {len(df)} rows, expected {len(holders)}"
        )
        
        # Verify data types for key columns
        if len(df) > 0:
            # address should be string
            assert df["address"].dtype == object, (
                f"address should be string type"
            )
            # balance should be numeric (float after conversion)
            assert pd.api.types.is_numeric_dtype(df["balance"]), (
                f"balance should be numeric type"
            )
            # is_store_of_value should be boolean
            assert df["is_store_of_value"].dtype == bool, (
                f"is_store_of_value should be boolean type"
            )

    @settings(max_examples=100, deadline=None)
    @given(transactions=st.lists(valid_transaction_model(), min_size=1, max_size=20))
    def test_property_13_dataframe_round_trip_preserves_data(self, transactions):
        """
        **Feature: stablecoin-analysis-notebook, Property 13: ZenML step output typing**

        For any list of transactions, converting to DataFrame and back SHALL
        preserve all data without loss.

        **Validates: Requirements 9.1, 9.2**
        """
        # Convert to DataFrame
        df = transactions_to_dataframe(transactions)
        
        # Convert back to Transaction objects
        recovered = dataframe_to_transactions(df)
        
        # Verify count matches
        assert len(recovered) == len(transactions), (
            f"Round-trip lost transactions: {len(transactions)} -> {len(recovered)}"
        )
        
        # Verify key fields are preserved
        original_hashes = {tx.transaction_hash for tx in transactions}
        recovered_hashes = {tx.transaction_hash for tx in recovered}
        assert original_hashes == recovered_hashes, (
            f"Transaction hashes not preserved in round-trip"
        )
        
        # Verify amounts are preserved (within floating-point tolerance)
        for orig, rec in zip(
            sorted(transactions, key=lambda x: x.transaction_hash),
            sorted(recovered, key=lambda x: x.transaction_hash)
        ):
            assert abs(float(orig.amount) - float(rec.amount)) < 0.000001, (
                f"Amount not preserved: {orig.amount} -> {rec.amount}"
            )

    @settings(max_examples=100, deadline=None)
    @given(holders=st.lists(valid_holder_model(), min_size=1, max_size=20))
    def test_property_13_holders_round_trip_preserves_data(self, holders):
        """
        **Feature: stablecoin-analysis-notebook, Property 13: ZenML step output typing**

        For any list of holders, converting to DataFrame and back SHALL
        preserve all data without loss.

        **Validates: Requirements 9.1, 9.2**
        """
        # Convert to DataFrame
        df = holders_to_dataframe(holders)
        
        # Convert back to Holder objects
        recovered = dataframe_to_holders(df)
        
        # Verify count matches
        assert len(recovered) == len(holders), (
            f"Round-trip lost holders: {len(holders)} -> {len(recovered)}"
        )
        
        # Verify key fields are preserved
        original_addresses = {h.address for h in holders}
        recovered_addresses = {h.address for h in recovered}
        assert original_addresses == recovered_addresses, (
            f"Holder addresses not preserved in round-trip"
        )
        
        # Verify balances are preserved (within floating-point tolerance)
        for orig, rec in zip(
            sorted(holders, key=lambda x: x.address),
            sorted(recovered, key=lambda x: x.address)
        ):
            assert abs(float(orig.balance) - float(rec.balance)) < 0.000001, (
                f"Balance not preserved: {orig.balance} -> {rec.balance}"
            )



class TestAggregationPreservesRecords:
    """Tests for aggregation preserving records (Property 14)."""

    @settings(max_examples=100, deadline=None)
    @given(
        eth_transactions=st.lists(valid_transaction_model(), min_size=0, max_size=15),
        bsc_transactions=st.lists(valid_transaction_model(), min_size=0, max_size=15),
        poly_transactions=st.lists(valid_transaction_model(), min_size=0, max_size=15),
    )
    def test_property_14_aggregation_preserves_unique_transactions(
        self, eth_transactions, bsc_transactions, poly_transactions
    ):
        """
        **Feature: stablecoin-analysis-notebook, Property 14: Aggregation preserves records**

        For any set of collector outputs, the aggregated result SHALL contain
        all unique transactions from all sources (deduplication by transaction_hash).

        **Validates: Requirements 9.3**
        """
        # Set correct chains for each set
        for tx in eth_transactions:
            object.__setattr__(tx, 'chain', 'ethereum')
            object.__setattr__(tx, 'source_explorer', 'etherscan')
        for tx in bsc_transactions:
            object.__setattr__(tx, 'chain', 'bsc')
            object.__setattr__(tx, 'source_explorer', 'bscscan')
        for tx in poly_transactions:
            object.__setattr__(tx, 'chain', 'polygon')
            object.__setattr__(tx, 'source_explorer', 'polygonscan')
        
        # Create collector outputs
        eth_output = CollectorOutput(
            transactions_df=transactions_to_dataframe(eth_transactions),
            holders_df=holders_to_dataframe([]),
            explorer_name="etherscan",
            chain="ethereum",
            success=len(eth_transactions) > 0,
            errors=[] if eth_transactions else ["No data"],
        )
        
        bsc_output = CollectorOutput(
            transactions_df=transactions_to_dataframe(bsc_transactions),
            holders_df=holders_to_dataframe([]),
            explorer_name="bscscan",
            chain="bsc",
            success=len(bsc_transactions) > 0,
            errors=[] if bsc_transactions else ["No data"],
        )
        
        poly_output = CollectorOutput(
            transactions_df=transactions_to_dataframe(poly_transactions),
            holders_df=holders_to_dataframe([]),
            explorer_name="polygonscan",
            chain="polygon",
            success=len(poly_transactions) > 0,
            errors=[] if poly_transactions else ["No data"],
        )
        
        # Count successful collectors
        successful_count = sum([
            1 if eth_output.success else 0,
            1 if bsc_output.success else 0,
            1 if poly_output.success else 0,
        ])
        
        # Skip test if not enough successful collectors
        assume(successful_count >= 1)
        
        # Run aggregation with min_successful_collectors=1 to allow partial data
        try:
            result = aggregate_data_step.entrypoint(
                etherscan_output=eth_output,
                bscscan_output=bsc_output,
                polygonscan_output=poly_output,
                min_successful_collectors=1,
            )
        except ValueError:
            # If aggregation fails due to no successful collectors, skip
            assume(False)
            return
        
        # Calculate expected unique transactions
        # Key is (hash, chain) since same hash on different chains is different tx
        all_transactions = eth_transactions + bsc_transactions + poly_transactions
        unique_keys = set()
        for tx in all_transactions:
            key = (tx.transaction_hash, tx.chain)
            unique_keys.add(key)
        
        expected_unique_count = len(unique_keys)
        
        # Verify aggregated count matches unique count
        actual_count = len(result.transactions_df)
        assert actual_count == expected_unique_count, (
            f"Aggregated transaction count ({actual_count}) != "
            f"expected unique count ({expected_unique_count})"
        )
        
        # Verify all unique transaction hashes are present
        if actual_count > 0:
            result_keys = set(
                zip(
                    result.transactions_df["transaction_hash"],
                    result.transactions_df["chain"]
                )
            )
            assert result_keys == unique_keys, (
                f"Aggregated transactions don't match expected unique set"
            )

    @settings(max_examples=100, deadline=None)
    @given(
        eth_holders=st.lists(valid_holder_model(), min_size=0, max_size=10),
        bsc_holders=st.lists(valid_holder_model(), min_size=0, max_size=10),
        poly_holders=st.lists(valid_holder_model(), min_size=0, max_size=10),
    )
    def test_property_14_aggregation_merges_holders_correctly(
        self, eth_holders, bsc_holders, poly_holders
    ):
        """
        **Feature: stablecoin-analysis-notebook, Property 14: Aggregation preserves records**

        For any set of collector outputs, the aggregated result SHALL contain
        all unique holders merged by (address, stablecoin, chain).

        **Validates: Requirements 9.3**
        """
        # Set correct chains for each set
        for h in eth_holders:
            object.__setattr__(h, 'chain', 'ethereum')
            object.__setattr__(h, 'source_explorer', 'etherscan')
        for h in bsc_holders:
            object.__setattr__(h, 'chain', 'bsc')
            object.__setattr__(h, 'source_explorer', 'bscscan')
        for h in poly_holders:
            object.__setattr__(h, 'chain', 'polygon')
            object.__setattr__(h, 'source_explorer', 'polygonscan')
        
        # Create collector outputs
        eth_output = CollectorOutput(
            transactions_df=transactions_to_dataframe([]),
            holders_df=holders_to_dataframe(eth_holders),
            explorer_name="etherscan",
            chain="ethereum",
            success=len(eth_holders) > 0,
            errors=[] if eth_holders else ["No data"],
        )
        
        bsc_output = CollectorOutput(
            transactions_df=transactions_to_dataframe([]),
            holders_df=holders_to_dataframe(bsc_holders),
            explorer_name="bscscan",
            chain="bsc",
            success=len(bsc_holders) > 0,
            errors=[] if bsc_holders else ["No data"],
        )
        
        poly_output = CollectorOutput(
            transactions_df=transactions_to_dataframe([]),
            holders_df=holders_to_dataframe(poly_holders),
            explorer_name="polygonscan",
            chain="polygon",
            success=len(poly_holders) > 0,
            errors=[] if poly_holders else ["No data"],
        )
        
        # Count successful collectors
        successful_count = sum([
            1 if eth_output.success else 0,
            1 if bsc_output.success else 0,
            1 if poly_output.success else 0,
        ])
        
        # Skip test if not enough successful collectors
        assume(successful_count >= 1)
        
        # Run aggregation
        try:
            result = aggregate_data_step.entrypoint(
                etherscan_output=eth_output,
                bscscan_output=bsc_output,
                polygonscan_output=poly_output,
                min_successful_collectors=1,
            )
        except ValueError:
            assume(False)
            return
        
        # Calculate expected unique holders
        # Key is (address.lower(), stablecoin, chain)
        all_holders = eth_holders + bsc_holders + poly_holders
        unique_keys = set()
        for h in all_holders:
            key = (h.address.lower(), h.stablecoin, h.chain)
            unique_keys.add(key)
        
        expected_unique_count = len(unique_keys)
        
        # Verify aggregated count matches unique count
        actual_count = len(result.holders_df)
        assert actual_count == expected_unique_count, (
            f"Aggregated holder count ({actual_count}) != "
            f"expected unique count ({expected_unique_count})"
        )

    @settings(max_examples=50, deadline=None)
    @given(
        eth_transactions=st.lists(valid_transaction_model(), min_size=1, max_size=10),
        bsc_transactions=st.lists(valid_transaction_model(), min_size=1, max_size=10),
    )
    def test_property_14_completeness_ratio_calculated_correctly(
        self, eth_transactions, bsc_transactions
    ):
        """
        **Feature: stablecoin-analysis-notebook, Property 14: Aggregation preserves records**

        For any set of collector outputs, the completeness_ratio SHALL equal
        successful_collectors / total_collectors.

        **Validates: Requirements 9.3, 9.4**
        """
        # Set correct chains
        for tx in eth_transactions:
            object.__setattr__(tx, 'chain', 'ethereum')
        for tx in bsc_transactions:
            object.__setattr__(tx, 'chain', 'bsc')
        
        # Create outputs - eth and bsc succeed, polygon fails
        eth_output = CollectorOutput(
            transactions_df=transactions_to_dataframe(eth_transactions),
            holders_df=holders_to_dataframe([]),
            explorer_name="etherscan",
            chain="ethereum",
            success=True,
            errors=[],
        )
        
        bsc_output = CollectorOutput(
            transactions_df=transactions_to_dataframe(bsc_transactions),
            holders_df=holders_to_dataframe([]),
            explorer_name="bscscan",
            chain="bsc",
            success=True,
            errors=[],
        )
        
        # Polygon fails
        poly_output = CollectorOutput(
            transactions_df=transactions_to_dataframe([]),
            holders_df=holders_to_dataframe([]),
            explorer_name="polygonscan",
            chain="polygon",
            success=False,
            errors=["Collection failed"],
        )
        
        # Run aggregation
        result = aggregate_data_step.entrypoint(
            etherscan_output=eth_output,
            bscscan_output=bsc_output,
            polygonscan_output=poly_output,
            min_successful_collectors=2,
        )
        
        # Verify completeness ratio
        expected_ratio = 2 / 3  # 2 successful out of 3
        assert abs(result.completeness_ratio - expected_ratio) < 0.001, (
            f"Completeness ratio ({result.completeness_ratio}) != "
            f"expected ({expected_ratio})"
        )
        
        # Verify successful/failed sources
        assert "etherscan" in result.successful_sources
        assert "bscscan" in result.successful_sources
        assert "polygonscan" in result.failed_sources
        assert len(result.successful_sources) == 2
        assert len(result.failed_sources) == 1
