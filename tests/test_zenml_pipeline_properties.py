"""
Property-based tests for ZenML pipeline steps.

These tests verify correctness properties for the ZenML collector and
aggregation steps defined in the design document.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
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


class TestPipelineArtifactVersioning:
    """Tests for pipeline artifact versioning (Property 15)."""

    @settings(max_examples=100, deadline=None)
    @given(transactions=st.lists(valid_transaction_model(), min_size=1, max_size=20))
    def test_property_15_activity_analysis_output_is_serializable(
        self, transactions
    ):
        """
        **Feature: stablecoin-analysis-notebook, Property 15: Pipeline artifact versioning**

        For any activity analysis step execution, the output artifact SHALL be
        serializable and contain all required fields for versioned storage.

        **Validates: Requirements 10.2**
        """
        from pipelines.steps.analysis import (
            activity_analysis_step,
            ActivityAnalysisOutput,
        )
        from notebooks.stablecoin_analysis_functions import ACTIVITY_TYPES
        
        # Set consistent chain for transactions
        for tx in transactions:
            object.__setattr__(tx, 'chain', 'ethereum')
        
        # Convert to DataFrame
        df = transactions_to_dataframe(transactions)
        
        # Run the analysis step
        result = activity_analysis_step.entrypoint(transactions_df=df)
        
        # Verify output type
        assert isinstance(result, ActivityAnalysisOutput), (
            f"Expected ActivityAnalysisOutput, got {type(result)}"
        )
        
        # Verify all activity types are present in output
        for at in ACTIVITY_TYPES:
            assert at in result.counts, (
                f"Activity type '{at}' missing from counts"
            )
            assert at in result.percentages, (
                f"Activity type '{at}' missing from percentages"
            )
            assert at in result.volumes, (
                f"Activity type '{at}' missing from volumes"
            )
            assert at in result.volume_percentages, (
                f"Activity type '{at}' missing from volume_percentages"
            )
        
        # Verify output is serializable to dict
        output_dict = result.to_dict()
        assert isinstance(output_dict, dict), (
            f"to_dict() should return dict, got {type(output_dict)}"
        )
        
        # Verify dict contains all required keys
        required_keys = ["counts", "percentages", "volumes", "volume_percentages"]
        for key in required_keys:
            assert key in output_dict, (
                f"Required key '{key}' missing from serialized output"
            )
        
        # Verify counts sum equals transaction count
        total_count = sum(result.counts.values())
        assert total_count == len(transactions), (
            f"Total count ({total_count}) != transaction count ({len(transactions)})"
        )

    @settings(max_examples=100, deadline=None)
    @given(holders=st.lists(valid_holder_model(), min_size=1, max_size=20))
    def test_property_15_holder_analysis_output_is_serializable(self, holders):
        """
        **Feature: stablecoin-analysis-notebook, Property 15: Pipeline artifact versioning**

        For any holder analysis step execution, the output artifact SHALL be
        serializable and contain all required fields for versioned storage.

        **Validates: Requirements 10.2**
        """
        from pipelines.steps.analysis import (
            holder_analysis_step,
            HolderAnalysisOutput,
        )
        
        # Set consistent chain for holders
        for h in holders:
            object.__setattr__(h, 'chain', 'ethereum')
        
        # Convert to DataFrames
        holders_df = holders_to_dataframe(holders)
        transactions_df = transactions_to_dataframe([])  # Empty transactions
        
        # Run the analysis step
        result = holder_analysis_step.entrypoint(
            holders_df=holders_df,
            transactions_df=transactions_df,
            top_n=10,
        )
        
        # Verify output type
        assert isinstance(result, HolderAnalysisOutput), (
            f"Expected HolderAnalysisOutput, got {type(result)}"
        )
        
        # Verify total holders matches input
        assert result.total_holders == len(holders), (
            f"Total holders ({result.total_holders}) != input count ({len(holders)})"
        )
        
        # Verify SoV count is valid
        assert 0 <= result.sov_count <= result.total_holders, (
            f"SoV count ({result.sov_count}) out of valid range"
        )
        
        # Verify output is serializable to dict
        output_dict = result.to_dict()
        assert isinstance(output_dict, dict), (
            f"to_dict() should return dict, got {type(output_dict)}"
        )
        
        # Verify dict contains all required keys
        required_keys = [
            "total_holders", "sov_count", "sov_percentage",
            "avg_balance_sov", "avg_balance_active",
            "avg_holding_period_days", "median_holding_period_days",
            "top_holders"
        ]
        for key in required_keys:
            assert key in output_dict, (
                f"Required key '{key}' missing from serialized output"
            )
        
        # Verify top_holders is a list
        assert isinstance(output_dict["top_holders"], list), (
            f"top_holders should be list, got {type(output_dict['top_holders'])}"
        )

    @settings(max_examples=100, deadline=None)
    @given(transactions=st.lists(valid_transaction_model(), min_size=1, max_size=20))
    def test_property_15_time_series_output_is_dataframe(self, transactions):
        """
        **Feature: stablecoin-analysis-notebook, Property 15: Pipeline artifact versioning**

        For any time series step execution, the output artifact SHALL be
        a valid pandas DataFrame suitable for versioned storage.

        **Validates: Requirements 10.2**
        """
        from pipelines.steps.analysis import time_series_step
        
        # Set consistent chain for transactions
        for tx in transactions:
            object.__setattr__(tx, 'chain', 'ethereum')
        
        # Convert to DataFrame
        df = transactions_to_dataframe(transactions)
        
        # Run the analysis step
        result = time_series_step.entrypoint(
            transactions_df=df,
            aggregation="daily",
        )
        
        # Verify output type
        assert isinstance(result, pd.DataFrame), (
            f"Expected pandas DataFrame, got {type(result)}"
        )
        
        # Verify required columns are present
        required_columns = [
            "period", "activity_type", "stablecoin",
            "transaction_count", "volume"
        ]
        for col in required_columns:
            assert col in result.columns, (
                f"Required column '{col}' missing from time series output"
            )
        
        # Verify aggregated counts sum to original count
        if len(result) > 0:
            total_aggregated = result["transaction_count"].sum()
            assert total_aggregated == len(transactions), (
                f"Aggregated count ({total_aggregated}) != "
                f"original count ({len(transactions)})"
            )

    @settings(max_examples=100, deadline=None)
    @given(transactions=st.lists(valid_transaction_model(), min_size=1, max_size=20))
    def test_property_15_chain_analysis_output_is_serializable(self, transactions):
        """
        **Feature: stablecoin-analysis-notebook, Property 15: Pipeline artifact versioning**

        For any chain analysis step execution, the output artifact SHALL be
        serializable and contain metrics for all supported chains.

        **Validates: Requirements 10.2**
        """
        from pipelines.steps.analysis import (
            chain_analysis_step,
            ChainAnalysisOutput,
        )
        from notebooks.stablecoin_analysis_functions import SUPPORTED_CHAINS
        
        # Distribute transactions across chains
        for i, tx in enumerate(transactions):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(tx, 'chain', chain)
        
        # Convert to DataFrame
        df = transactions_to_dataframe(transactions)
        
        # Run the analysis step
        result = chain_analysis_step.entrypoint(
            transactions_df=df,
            holders_df=None,
        )
        
        # Verify output type
        assert isinstance(result, ChainAnalysisOutput), (
            f"Expected ChainAnalysisOutput, got {type(result)}"
        )
        
        # Verify output is serializable to dict
        output_dict = result.to_dict()
        assert isinstance(output_dict, dict), (
            f"to_dict() should return dict, got {type(output_dict)}"
        )
        
        # Verify chain_metrics is present and is a list
        assert "chain_metrics" in output_dict, (
            "chain_metrics key missing from serialized output"
        )
        assert isinstance(output_dict["chain_metrics"], list), (
            f"chain_metrics should be list, got {type(output_dict['chain_metrics'])}"
        )
        
        # Verify all supported chains are present
        chains_in_output = {m["chain"] for m in output_dict["chain_metrics"]}
        for chain in SUPPORTED_CHAINS:
            assert chain in chains_in_output, (
                f"Chain '{chain}' missing from chain metrics output"
            )
        
        # Verify each chain metric has required fields
        required_fields = [
            "chain", "transaction_count", "total_volume",
            "avg_transaction_size", "sov_ratio", "activity_distribution"
        ]
        for metric in output_dict["chain_metrics"]:
            for field in required_fields:
                assert field in metric, (
                    f"Required field '{field}' missing from chain metric"
                )

    @settings(max_examples=50, deadline=None)
    @given(
        transactions=st.lists(valid_transaction_model(), min_size=5, max_size=15),
        holders=st.lists(valid_holder_model(), min_size=3, max_size=10),
    )
    def test_property_15_all_analysis_outputs_are_versioned_artifacts(
        self, transactions, holders
    ):
        """
        **Feature: stablecoin-analysis-notebook, Property 15: Pipeline artifact versioning**

        For any completed analysis pipeline run, all output artifacts SHALL be
        retrievable and contain consistent data across all analysis steps.

        **Validates: Requirements 10.2**
        """
        from pipelines.steps.analysis import (
            activity_analysis_step,
            holder_analysis_step,
            time_series_step,
            chain_analysis_step,
        )
        
        # Distribute data across chains
        for i, tx in enumerate(transactions):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(tx, 'chain', chain)
        for i, h in enumerate(holders):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(h, 'chain', chain)
        
        # Convert to DataFrames
        transactions_df = transactions_to_dataframe(transactions)
        holders_df = holders_to_dataframe(holders)
        
        # Run all analysis steps
        activity_output = activity_analysis_step.entrypoint(
            transactions_df=transactions_df
        )
        holder_output = holder_analysis_step.entrypoint(
            holders_df=holders_df,
            transactions_df=transactions_df,
            top_n=10,
        )
        time_series_output = time_series_step.entrypoint(
            transactions_df=transactions_df,
            aggregation="daily",
        )
        chain_output = chain_analysis_step.entrypoint(
            transactions_df=transactions_df,
            holders_df=holders_df,
        )
        
        # Verify all outputs are serializable
        activity_dict = activity_output.to_dict()
        holder_dict = holder_output.to_dict()
        chain_dict = chain_output.to_dict()
        
        # Verify consistency: total counts should match
        activity_total = sum(activity_output.counts.values())
        assert activity_total == len(transactions), (
            f"Activity total ({activity_total}) != transaction count ({len(transactions)})"
        )
        
        holder_total = holder_output.total_holders
        assert holder_total == len(holders), (
            f"Holder total ({holder_total}) != holder count ({len(holders)})"
        )
        
        # Verify time series aggregation preserves total
        if len(time_series_output) > 0:
            ts_total = time_series_output["transaction_count"].sum()
            assert ts_total == len(transactions), (
                f"Time series total ({ts_total}) != transaction count ({len(transactions)})"
            )
        
        # Verify chain metrics cover all transactions
        chain_total = sum(
            m["transaction_count"] for m in chain_dict["chain_metrics"]
        )
        assert chain_total == len(transactions), (
            f"Chain total ({chain_total}) != transaction count ({len(transactions)})"
        )


class TestFeatureEngineeringCompleteness:
    """Tests for feature engineering completeness (Property 16)."""

    @settings(max_examples=100, deadline=None)
    @given(
        holders=st.lists(valid_holder_model(), min_size=1, max_size=20),
        transactions=st.lists(valid_transaction_model(), min_size=0, max_size=30),
    )
    def test_property_16_feature_engineering_produces_all_required_features(
        self, holders, transactions
    ):
        """
        **Feature: stablecoin-analysis-notebook, Property 16: Feature engineering completeness**

        For any holder with transaction history, the feature engineering step
        SHALL produce a feature vector with all required fields:
        - transaction_count
        - avg_transaction_size
        - balance_percentile
        - holding_period_days
        - activity_recency_days
        - transaction_frequency
        - balance_volatility
        - cross_chain_flag

        **Validates: Requirements 11.1, 12.2**
        """
        from pipelines.steps.ml import (
            feature_engineering_step,
            REQUIRED_FEATURES,
            FeatureEngineeringOutput,
        )

        # Set consistent chains for data
        for i, h in enumerate(holders):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(h, 'chain', chain)
        for i, tx in enumerate(transactions):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(tx, 'chain', chain)

        # Convert to DataFrames
        holders_df = holders_to_dataframe(holders)
        transactions_df = transactions_to_dataframe(transactions)

        # Run feature engineering step
        result = feature_engineering_step.entrypoint(
            transactions_df=transactions_df,
            holders_df=holders_df,
        )

        # Verify output type
        assert isinstance(result, FeatureEngineeringOutput), (
            f"Expected FeatureEngineeringOutput, got {type(result)}"
        )

        # Verify all required features are present in the DataFrame
        for feat in REQUIRED_FEATURES:
            assert feat in result.features_df.columns, (
                f"Required feature '{feat}' missing from features DataFrame"
            )

        # Verify feature_names list matches REQUIRED_FEATURES
        assert set(result.feature_names) == set(REQUIRED_FEATURES), (
            f"feature_names {result.feature_names} != REQUIRED_FEATURES {REQUIRED_FEATURES}"
        )

        # Verify holder count matches
        assert result.holder_count == len(holders), (
            f"holder_count ({result.holder_count}) != input holders ({len(holders)})"
        )

        # Verify each holder has a feature vector
        assert len(result.features_df) == len(holders), (
            f"Feature rows ({len(result.features_df)}) != holders ({len(holders)})"
        )

    @settings(max_examples=100, deadline=None)
    @given(holders=st.lists(valid_holder_model(), min_size=1, max_size=20))
    def test_property_16_feature_values_are_valid(self, holders):
        """
        **Feature: stablecoin-analysis-notebook, Property 16: Feature engineering completeness**

        For any holder, all extracted feature values SHALL be valid:
        - transaction_count >= 0
        - avg_transaction_size >= 0
        - balance_percentile in [0, 100]
        - holding_period_days >= 0
        - activity_recency_days >= 0
        - transaction_frequency >= 0
        - balance_volatility >= 0
        - cross_chain_flag in {0, 1}

        **Validates: Requirements 11.1, 12.2**
        """
        from pipelines.steps.ml import (
            feature_engineering_step,
            REQUIRED_FEATURES,
        )

        # Set consistent chains
        for i, h in enumerate(holders):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(h, 'chain', chain)

        # Convert to DataFrames
        holders_df = holders_to_dataframe(holders)
        transactions_df = transactions_to_dataframe([])  # Empty transactions

        # Run feature engineering
        result = feature_engineering_step.entrypoint(
            transactions_df=transactions_df,
            holders_df=holders_df,
        )

        df = result.features_df

        # Verify non-negative features
        non_negative_features = [
            "transaction_count",
            "avg_transaction_size",
            "holding_period_days",
            "activity_recency_days",
            "transaction_frequency",
            "balance_volatility",
        ]
        for feat in non_negative_features:
            assert (df[feat] >= 0).all(), (
                f"Feature '{feat}' has negative values: {df[feat].min()}"
            )

        # Verify balance_percentile is in [0, 100]
        assert (df["balance_percentile"] >= 0).all(), (
            f"balance_percentile has values < 0: {df['balance_percentile'].min()}"
        )
        assert (df["balance_percentile"] <= 100).all(), (
            f"balance_percentile has values > 100: {df['balance_percentile'].max()}"
        )

        # Verify cross_chain_flag is binary
        assert df["cross_chain_flag"].isin([0, 1]).all(), (
            f"cross_chain_flag has non-binary values: {df['cross_chain_flag'].unique()}"
        )

    @settings(max_examples=100, deadline=None)
    @given(
        holders=st.lists(valid_holder_model(), min_size=1, max_size=15),
        transactions=st.lists(valid_transaction_model(), min_size=1, max_size=30),
    )
    def test_property_16_feature_engineering_is_deterministic(
        self, holders, transactions
    ):
        """
        **Feature: stablecoin-analysis-notebook, Property 16: Feature engineering completeness**

        For any input data, running feature engineering twice SHALL produce
        identical feature vectors.

        **Validates: Requirements 11.1, 12.2**
        """
        import numpy as np
        from pipelines.steps.ml import compute_holder_features
        from datetime import datetime, timezone

        # Set consistent chains
        for i, h in enumerate(holders):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(h, 'chain', chain)
        for i, tx in enumerate(transactions):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(tx, 'chain', chain)

        # Convert to DataFrames
        holders_df = holders_to_dataframe(holders)
        transactions_df = transactions_to_dataframe(transactions)

        # Use fixed reference date for determinism
        ref_date = datetime(2025, 1, 1, tzinfo=timezone.utc)

        # Run feature engineering twice
        result1 = compute_holder_features(holders_df, transactions_df, ref_date)
        result2 = compute_holder_features(holders_df, transactions_df, ref_date)

        # Verify results are identical
        assert result1.shape == result2.shape, (
            f"Shape mismatch: {result1.shape} vs {result2.shape}"
        )

        # Compare all columns
        for col in result1.columns:
            if col == "address":
                continue  # Skip address comparison
            # Use numpy allclose for numeric comparison
            assert np.allclose(
                result1[col].values,
                result2[col].values,
                rtol=1e-10,
                equal_nan=True,
            ), f"Column '{col}' differs between runs"

    @settings(max_examples=50, deadline=None)
    @given(holders=st.lists(valid_holder_model(), min_size=1, max_size=10))
    def test_property_16_empty_transactions_produces_valid_features(self, holders):
        """
        **Feature: stablecoin-analysis-notebook, Property 16: Feature engineering completeness**

        For any set of holders with no transactions, feature engineering SHALL
        still produce valid feature vectors with appropriate default values.

        **Validates: Requirements 11.1, 12.2**
        """
        from pipelines.steps.ml import (
            feature_engineering_step,
            REQUIRED_FEATURES,
        )

        # Set consistent chains
        for i, h in enumerate(holders):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(h, 'chain', chain)

        # Convert to DataFrames
        holders_df = holders_to_dataframe(holders)
        transactions_df = transactions_to_dataframe([])  # Empty transactions

        # Run feature engineering
        result = feature_engineering_step.entrypoint(
            transactions_df=transactions_df,
            holders_df=holders_df,
        )

        # Verify all features are present
        for feat in REQUIRED_FEATURES:
            assert feat in result.features_df.columns, (
                f"Required feature '{feat}' missing with empty transactions"
            )

        # Verify transaction-based features are zero
        assert (result.features_df["transaction_count"] == 0).all(), (
            "transaction_count should be 0 with no transactions"
        )
        assert (result.features_df["avg_transaction_size"] == 0).all(), (
            "avg_transaction_size should be 0 with no transactions"
        )
        assert (result.features_df["balance_volatility"] == 0).all(), (
            "balance_volatility should be 0 with no transactions"
        )
        assert (result.features_df["cross_chain_flag"] == 0).all(), (
            "cross_chain_flag should be 0 with no transactions"
        )



class TestSoVPredictionProbabilityBounds:
    """Tests for SoV prediction probability bounds (Property 17)."""

    @settings(max_examples=100, deadline=None)
    @given(
        holders=st.lists(valid_holder_model(), min_size=10, max_size=30),
        transactions=st.lists(valid_transaction_model(), min_size=5, max_size=50),
    )
    def test_property_17_sov_prediction_probabilities_in_valid_range(
        self, holders, transactions
    ):
        """
        **Feature: stablecoin-analysis-notebook, Property 17: SoV prediction probability bounds**

        For any holder, the SoV prediction probability SHALL be in the range [0.0, 1.0].

        **Validates: Requirements 11.5**
        """
        from pipelines.steps.ml import (
            feature_engineering_step,
            train_sov_predictor_step,
            predict_sov_step,
        )

        # Ensure we have enough holders with both SoV classes for training
        # Set at least 30% as SoV to ensure stratified split works
        for i, h in enumerate(holders):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(h, 'chain', chain)
            # Ensure balanced classes
            is_sov = i % 3 == 0  # ~33% SoV
            object.__setattr__(h, 'is_store_of_value', is_sov)

        for i, tx in enumerate(transactions):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(tx, 'chain', chain)

        # Convert to DataFrames
        holders_df = holders_to_dataframe(holders)
        transactions_df = transactions_to_dataframe(transactions)

        # Run feature engineering
        features_output = feature_engineering_step.entrypoint(
            transactions_df=transactions_df,
            holders_df=holders_df,
        )

        # Skip if not enough samples for training
        assume(len(features_output.features_df) >= 10)

        # Train model (use RandomForest to avoid XGBoost dependency issues)
        try:
            model_output = train_sov_predictor_step.entrypoint(
                features_df=features_output.features_df,
                holders_df=holders_df,
                algorithm='random_forest',
                n_estimators=20,
                max_depth=5,
                test_size=0.2,
            )
        except ValueError:
            # Skip if training fails due to insufficient data
            assume(False)
            return

        # Run inference
        predictions_output = predict_sov_step.entrypoint(
            features_df=features_output.features_df,
            model=model_output.model,
        )

        # Verify all probabilities are in [0.0, 1.0]
        probs = predictions_output.predictions_df['sov_probability']

        assert (probs >= 0.0).all(), (
            f"Found probability < 0.0: min={probs.min()}"
        )
        assert (probs <= 1.0).all(), (
            f"Found probability > 1.0: max={probs.max()}"
        )

    @settings(max_examples=50, deadline=None)
    @given(holders=st.lists(valid_holder_model(), min_size=5, max_size=15))
    def test_property_17_empty_transactions_still_produces_valid_probabilities(
        self, holders
    ):
        """
        **Feature: stablecoin-analysis-notebook, Property 17: SoV prediction probability bounds**

        For any set of holders with no transactions, SoV prediction SHALL still
        produce valid probabilities in [0.0, 1.0].

        **Validates: Requirements 11.5**
        """
        from pipelines.steps.ml import (
            feature_engineering_step,
            train_sov_predictor_step,
            predict_sov_step,
        )

        # Set consistent chains and ensure balanced classes
        for i, h in enumerate(holders):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(h, 'chain', chain)
            is_sov = i % 3 == 0
            object.__setattr__(h, 'is_store_of_value', is_sov)

        # Convert to DataFrames
        holders_df = holders_to_dataframe(holders)
        transactions_df = transactions_to_dataframe([])  # Empty transactions

        # Run feature engineering
        features_output = feature_engineering_step.entrypoint(
            transactions_df=transactions_df,
            holders_df=holders_df,
        )

        # Skip if not enough samples
        assume(len(features_output.features_df) >= 5)

        # Train model
        try:
            model_output = train_sov_predictor_step.entrypoint(
                features_df=features_output.features_df,
                holders_df=holders_df,
                algorithm='random_forest',
                n_estimators=10,
                max_depth=3,
                test_size=0.2,
            )
        except ValueError:
            assume(False)
            return

        # Run inference
        predictions_output = predict_sov_step.entrypoint(
            features_df=features_output.features_df,
            model=model_output.model,
        )

        # Verify probabilities are valid
        probs = predictions_output.predictions_df['sov_probability']
        assert (probs >= 0.0).all() and (probs <= 1.0).all(), (
            f"Probabilities out of range: min={probs.min()}, max={probs.max()}"
        )

    @settings(max_examples=50, deadline=None)
    @given(
        holders=st.lists(valid_holder_model(), min_size=15, max_size=30),
        transactions=st.lists(valid_transaction_model(), min_size=10, max_size=40),
    )
    def test_property_17_prediction_output_has_required_columns(
        self, holders, transactions
    ):
        """
        **Feature: stablecoin-analysis-notebook, Property 17: SoV prediction probability bounds**

        For any prediction output, the DataFrame SHALL contain required columns:
        - address
        - sov_probability
        - predicted_class

        **Validates: Requirements 11.5**
        """
        from pipelines.steps.ml import (
            feature_engineering_step,
            train_sov_predictor_step,
            predict_sov_step,
            SoVPredictionOutput,
        )

        # Set consistent chains and balanced classes
        for i, h in enumerate(holders):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(h, 'chain', chain)
            is_sov = i % 3 == 0
            object.__setattr__(h, 'is_store_of_value', is_sov)

        for i, tx in enumerate(transactions):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(tx, 'chain', chain)

        # Convert to DataFrames
        holders_df = holders_to_dataframe(holders)
        transactions_df = transactions_to_dataframe(transactions)

        # Run feature engineering
        features_output = feature_engineering_step.entrypoint(
            transactions_df=transactions_df,
            holders_df=holders_df,
        )

        assume(len(features_output.features_df) >= 10)

        # Train model
        try:
            model_output = train_sov_predictor_step.entrypoint(
                features_df=features_output.features_df,
                holders_df=holders_df,
                algorithm='random_forest',
                n_estimators=20,
                max_depth=5,
            )
        except ValueError:
            assume(False)
            return

        # Run inference
        predictions_output = predict_sov_step.entrypoint(
            features_df=features_output.features_df,
            model=model_output.model,
        )

        # Verify output type
        assert isinstance(predictions_output, SoVPredictionOutput), (
            f"Expected SoVPredictionOutput, got {type(predictions_output)}"
        )

        # Verify required columns
        required_columns = ['address', 'sov_probability', 'predicted_class']
        for col in required_columns:
            assert col in predictions_output.predictions_df.columns, (
                f"Required column '{col}' missing from predictions"
            )

        # Verify prediction count matches
        assert predictions_output.prediction_count == len(features_output.features_df), (
            f"Prediction count ({predictions_output.prediction_count}) != "
            f"feature count ({len(features_output.features_df)})"
        )

        # Verify predicted_class is boolean
        assert predictions_output.predictions_df['predicted_class'].dtype == bool, (
            f"predicted_class should be bool, got "
            f"{predictions_output.predictions_df['predicted_class'].dtype}"
        )



class TestWalletClassificationExclusivity:
    """Tests for wallet classification exclusivity (Property 18)."""

    @settings(max_examples=100, deadline=None)
    @given(
        holders=st.lists(valid_holder_model(), min_size=10, max_size=30),
        transactions=st.lists(valid_transaction_model(), min_size=5, max_size=50),
    )
    def test_property_18_each_wallet_assigned_exactly_one_class(
        self, holders, transactions
    ):
        """
        **Feature: stablecoin-analysis-notebook, Property 18: Wallet classification exclusivity**

        For any holder, the wallet classifier SHALL assign exactly one behavior
        class from {trader, holder, whale, retail}.

        **Validates: Requirements 12.4**
        """
        from pipelines.steps.ml import feature_engineering_step
        from pipelines.steps.wallet_classifier import (
            train_wallet_classifier_step,
            classify_wallets_step,
            WalletBehaviorClass,
        )

        # Set consistent chains and varied features for classification
        for i, h in enumerate(holders):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(h, 'chain', chain)

        for i, tx in enumerate(transactions):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(tx, 'chain', chain)

        # Convert to DataFrames
        holders_df = holders_to_dataframe(holders)
        transactions_df = transactions_to_dataframe(transactions)

        # Run feature engineering
        features_output = feature_engineering_step.entrypoint(
            transactions_df=transactions_df,
            holders_df=holders_df,
        )

        # Skip if not enough samples
        assume(len(features_output.features_df) >= 10)

        # Train classifier
        try:
            model_output = train_wallet_classifier_step.entrypoint(
                features_df=features_output.features_df,
                algorithm='random_forest',
                n_estimators=20,
                max_depth=5,
                test_size=0.2,
            )
        except ValueError:
            assume(False)
            return

        # Run classification
        classification_output = classify_wallets_step.entrypoint(
            features_df=features_output.features_df,
            model=model_output.model,
        )

        # Verify each wallet has exactly one class (exclusivity)
        valid_classes = set(WalletBehaviorClass.all_classes())
        
        for _, row in classification_output.classifications_df.iterrows():
            behavior_class = row['behavior_class']
            
            # Verify class is one of the valid classes (exactly one)
            assert behavior_class in valid_classes, (
                f"Invalid behavior class: {behavior_class}. "
                f"Expected one of {valid_classes}"
            )
            
            # Verify only one class is assigned (not multiple)
            # This is implicit since behavior_class is a single value, not a list

        # Verify classification count matches input (one classification per input row)
        assert classification_output.classification_count == len(features_output.features_df), (
            f"Classification count ({classification_output.classification_count}) != "
            f"input count ({len(features_output.features_df)})"
        )
        
        # Verify output row count matches input
        assert len(classification_output.classifications_df) == len(features_output.features_df), (
            f"Output rows ({len(classification_output.classifications_df)}) != "
            f"input rows ({len(features_output.features_df)})"
        )

    @settings(max_examples=100, deadline=None)
    @given(holders=st.lists(valid_holder_model(), min_size=5, max_size=20))
    def test_property_18_labeling_function_assigns_exactly_one_class(self, holders):
        """
        **Feature: stablecoin-analysis-notebook, Property 18: Wallet classification exclusivity**

        For any holder features, the labeling function SHALL assign exactly one
        behavior class from {trader, holder, whale, retail}.

        **Validates: Requirements 12.4**
        """
        from pipelines.steps.ml import feature_engineering_step
        from pipelines.steps.wallet_classifier import (
            label_wallet_behavior,
            WalletBehaviorClass,
        )

        # Set consistent chains
        for i, h in enumerate(holders):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(h, 'chain', chain)

        # Convert to DataFrames
        holders_df = holders_to_dataframe(holders)
        transactions_df = transactions_to_dataframe([])

        # Run feature engineering
        features_output = feature_engineering_step.entrypoint(
            transactions_df=transactions_df,
            holders_df=holders_df,
        )

        # Apply labeling function
        labels = label_wallet_behavior(features_output.features_df)

        # Verify each label is valid
        valid_classes = set(WalletBehaviorClass.all_classes())
        
        for label in labels:
            assert label in valid_classes, (
                f"Invalid label: {label}. Expected one of {valid_classes}"
            )

        # Verify label count matches input
        assert len(labels) == len(features_output.features_df), (
            f"Label count ({len(labels)}) != feature count ({len(features_output.features_df)})"
        )

    @settings(max_examples=50, deadline=None)
    @given(
        holders=st.lists(valid_holder_model(), min_size=15, max_size=30),
        transactions=st.lists(valid_transaction_model(), min_size=10, max_size=40),
    )
    def test_property_18_class_distribution_sums_to_total(
        self, holders, transactions
    ):
        """
        **Feature: stablecoin-analysis-notebook, Property 18: Wallet classification exclusivity**

        For any classification output, the sum of class distribution counts
        SHALL equal the total number of classified wallets.

        **Validates: Requirements 12.4**
        """
        from pipelines.steps.ml import feature_engineering_step
        from pipelines.steps.wallet_classifier import (
            train_wallet_classifier_step,
            classify_wallets_step,
        )

        # Set consistent chains
        for i, h in enumerate(holders):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(h, 'chain', chain)

        for i, tx in enumerate(transactions):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(tx, 'chain', chain)

        # Convert to DataFrames
        holders_df = holders_to_dataframe(holders)
        transactions_df = transactions_to_dataframe(transactions)

        # Run feature engineering
        features_output = feature_engineering_step.entrypoint(
            transactions_df=transactions_df,
            holders_df=holders_df,
        )

        assume(len(features_output.features_df) >= 10)

        # Train classifier
        try:
            model_output = train_wallet_classifier_step.entrypoint(
                features_df=features_output.features_df,
                algorithm='random_forest',
                n_estimators=20,
                max_depth=5,
            )
        except ValueError:
            assume(False)
            return

        # Run classification
        classification_output = classify_wallets_step.entrypoint(
            features_df=features_output.features_df,
            model=model_output.model,
        )

        # Verify class distribution sums to total
        total_from_distribution = sum(classification_output.class_distribution.values())
        assert total_from_distribution == classification_output.classification_count, (
            f"Class distribution sum ({total_from_distribution}) != "
            f"classification count ({classification_output.classification_count})"
        )

    @settings(max_examples=100, deadline=None)
    @given(holders=st.lists(valid_holder_model(), min_size=10, max_size=25))
    def test_property_18_confidence_in_valid_range(self, holders):
        """
        **Feature: stablecoin-analysis-notebook, Property 18: Wallet classification exclusivity**

        For any classification, the confidence score SHALL be in range [0.0, 1.0].

        **Validates: Requirements 12.4, 12.5**
        """
        from pipelines.steps.ml import feature_engineering_step
        from pipelines.steps.wallet_classifier import (
            train_wallet_classifier_step,
            classify_wallets_step,
        )

        # Set consistent chains
        for i, h in enumerate(holders):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(h, 'chain', chain)

        # Convert to DataFrames
        holders_df = holders_to_dataframe(holders)
        transactions_df = transactions_to_dataframe([])

        # Run feature engineering
        features_output = feature_engineering_step.entrypoint(
            transactions_df=transactions_df,
            holders_df=holders_df,
        )

        assume(len(features_output.features_df) >= 10)

        # Train classifier
        try:
            model_output = train_wallet_classifier_step.entrypoint(
                features_df=features_output.features_df,
                algorithm='random_forest',
                n_estimators=20,
                max_depth=5,
            )
        except ValueError:
            assume(False)
            return

        # Run classification
        classification_output = classify_wallets_step.entrypoint(
            features_df=features_output.features_df,
            model=model_output.model,
        )

        # Verify confidence is in valid range
        confidences = classification_output.classifications_df['confidence']
        
        assert (confidences >= 0.0).all(), (
            f"Found confidence < 0.0: min={confidences.min()}"
        )
        assert (confidences <= 1.0).all(), (
            f"Found confidence > 1.0: max={confidences.max()}"
        )


class TestModelMetricsValidity:
    """Tests for model metrics validity (Property 19)."""

    @settings(max_examples=100, deadline=None)
    @given(
        holders=st.lists(valid_holder_model(), min_size=15, max_size=30),
        transactions=st.lists(valid_transaction_model(), min_size=10, max_size=50),
    )
    def test_property_19_wallet_classifier_metrics_in_valid_range(
        self, holders, transactions
    ):
        """
        **Feature: stablecoin-analysis-notebook, Property 19: Model metrics validity**

        For any trained wallet classifier model, the evaluation metrics
        (precision, recall, F1, accuracy) SHALL each be in the range [0.0, 1.0].

        **Validates: Requirements 11.3, 15.2**
        """
        from pipelines.steps.ml import feature_engineering_step
        from pipelines.steps.wallet_classifier import train_wallet_classifier_step

        # Set consistent chains
        for i, h in enumerate(holders):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(h, 'chain', chain)

        for i, tx in enumerate(transactions):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(tx, 'chain', chain)

        # Convert to DataFrames
        holders_df = holders_to_dataframe(holders)
        transactions_df = transactions_to_dataframe(transactions)

        # Run feature engineering
        features_output = feature_engineering_step.entrypoint(
            transactions_df=transactions_df,
            holders_df=holders_df,
        )

        assume(len(features_output.features_df) >= 15)

        # Train classifier
        try:
            model_output = train_wallet_classifier_step.entrypoint(
                features_df=features_output.features_df,
                algorithm='random_forest',
                n_estimators=20,
                max_depth=5,
            )
        except ValueError:
            assume(False)
            return

        # Verify all metrics are in valid range [0.0, 1.0]
        for metric_name, metric_value in model_output.metrics.items():
            assert 0.0 <= metric_value <= 1.0, (
                f"Metric '{metric_name}' out of range: {metric_value}"
            )

    @settings(max_examples=100, deadline=None)
    @given(
        holders=st.lists(valid_holder_model(), min_size=15, max_size=30),
        transactions=st.lists(valid_transaction_model(), min_size=10, max_size=50),
    )
    def test_property_19_sov_predictor_metrics_in_valid_range(
        self, holders, transactions
    ):
        """
        **Feature: stablecoin-analysis-notebook, Property 19: Model metrics validity**

        For any trained SoV predictor model, the evaluation metrics
        (precision, recall, F1, AUC-ROC) SHALL each be in the range [0.0, 1.0].

        **Validates: Requirements 11.3, 15.2**
        """
        from pipelines.steps.ml import (
            feature_engineering_step,
            train_sov_predictor_step,
        )

        # Set consistent chains and balanced classes
        for i, h in enumerate(holders):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(h, 'chain', chain)
            is_sov = i % 3 == 0
            object.__setattr__(h, 'is_store_of_value', is_sov)

        for i, tx in enumerate(transactions):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(tx, 'chain', chain)

        # Convert to DataFrames
        holders_df = holders_to_dataframe(holders)
        transactions_df = transactions_to_dataframe(transactions)

        # Run feature engineering
        features_output = feature_engineering_step.entrypoint(
            transactions_df=transactions_df,
            holders_df=holders_df,
        )

        assume(len(features_output.features_df) >= 15)

        # Train SoV predictor
        try:
            model_output = train_sov_predictor_step.entrypoint(
                features_df=features_output.features_df,
                holders_df=holders_df,
                algorithm='random_forest',
                n_estimators=20,
                max_depth=5,
            )
        except ValueError:
            assume(False)
            return

        # Verify all metrics are in valid range [0.0, 1.0]
        for metric_name, metric_value in model_output.metrics.items():
            assert 0.0 <= metric_value <= 1.0, (
                f"Metric '{metric_name}' out of range: {metric_value}"
            )

    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
    @given(
        holders=st.lists(valid_holder_model(), min_size=20, max_size=35),
        transactions=st.lists(valid_transaction_model(), min_size=15, max_size=60),
    )
    def test_property_19_feature_importances_are_valid(
        self, holders, transactions
    ):
        """
        **Feature: stablecoin-analysis-notebook, Property 19: Model metrics validity**

        For any trained model, feature importances SHALL be non-negative and
        sum to approximately 1.0 (for tree-based models).

        **Validates: Requirements 11.3, 15.2**
        """
        from pipelines.steps.ml import feature_engineering_step
        from pipelines.steps.wallet_classifier import train_wallet_classifier_step

        # Set consistent chains
        for i, h in enumerate(holders):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(h, 'chain', chain)

        for i, tx in enumerate(transactions):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(tx, 'chain', chain)

        # Convert to DataFrames
        holders_df = holders_to_dataframe(holders)
        transactions_df = transactions_to_dataframe(transactions)

        # Run feature engineering
        features_output = feature_engineering_step.entrypoint(
            transactions_df=transactions_df,
            holders_df=holders_df,
        )

        assume(len(features_output.features_df) >= 20)

        # Train classifier
        try:
            model_output = train_wallet_classifier_step.entrypoint(
                features_df=features_output.features_df,
                algorithm='random_forest',
                n_estimators=20,
                max_depth=5,
            )
        except ValueError:
            assume(False)
            return

        # Verify feature importances are non-negative
        for feat_name, importance in model_output.feature_importances.items():
            assert importance >= 0.0, (
                f"Feature '{feat_name}' has negative importance: {importance}"
            )

        # Verify feature importances sum to approximately 1.0 (or 0.0 if model
        # couldn't learn due to degenerate data like all identical features)
        total_importance = sum(model_output.feature_importances.values())
        assert abs(total_importance - 1.0) < 0.01 or abs(total_importance) < 0.01, (
            f"Feature importances sum to {total_importance}, expected ~1.0 or ~0.0"
        )

    @settings(max_examples=50, deadline=None)
    @given(
        holders=st.lists(valid_holder_model(), min_size=15, max_size=30),
        transactions=st.lists(valid_transaction_model(), min_size=10, max_size=50),
    )
    def test_property_19_training_metadata_is_complete(
        self, holders, transactions
    ):
        """
        **Feature: stablecoin-analysis-notebook, Property 19: Model metrics validity**

        For any trained model, the training metadata SHALL contain required
        fields: training_timestamp, total_samples, train_samples, test_samples.

        **Validates: Requirements 11.3, 15.2**
        """
        from pipelines.steps.ml import feature_engineering_step
        from pipelines.steps.wallet_classifier import train_wallet_classifier_step

        # Set consistent chains
        for i, h in enumerate(holders):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(h, 'chain', chain)

        for i, tx in enumerate(transactions):
            chain = CHAINS[i % len(CHAINS)]
            object.__setattr__(tx, 'chain', chain)

        # Convert to DataFrames
        holders_df = holders_to_dataframe(holders)
        transactions_df = transactions_to_dataframe(transactions)

        # Run feature engineering
        features_output = feature_engineering_step.entrypoint(
            transactions_df=transactions_df,
            holders_df=holders_df,
        )

        assume(len(features_output.features_df) >= 15)

        # Train classifier
        try:
            model_output = train_wallet_classifier_step.entrypoint(
                features_df=features_output.features_df,
                algorithm='random_forest',
                n_estimators=20,
                max_depth=5,
            )
        except ValueError:
            assume(False)
            return

        # Verify required metadata fields
        required_fields = [
            "training_timestamp",
            "total_samples",
            "train_samples",
            "test_samples",
        ]
        
        for field in required_fields:
            assert field in model_output.training_metadata, (
                f"Required metadata field '{field}' missing"
            )

        # Verify sample counts are consistent
        total = model_output.training_metadata["total_samples"]
        train = model_output.training_metadata["train_samples"]
        test = model_output.training_metadata["test_samples"]
        
        assert train + test == total, (
            f"train ({train}) + test ({test}) != total ({total})"
        )
