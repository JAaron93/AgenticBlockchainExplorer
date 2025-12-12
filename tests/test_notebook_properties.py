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
from stablecoin_analysis_functions import get_time_series_totals


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


def compute_summary_from_data(transactions: list) -> dict:
    """
    Compute summary statistics from transaction data.
    
    This ensures the summary is consistent with the actual transactions.
    
    Note: This function is intended ONLY for Hypothesis-generated test data.
    Input transactions MUST conform to the following constraints:
    - tx["stablecoin"] must be in STABLECOINS ("USDC", "USDT")
    - tx["activity_type"] must be in ACTIVITY_TYPES ("transaction", "store_of_value", "other")
    - tx["chain"] must be in CHAINS ("ethereum", "bsc", "polygon")
    - tx["amount"] must be a valid decimal string (parseable by Decimal())
    
    These constraints are guaranteed by the valid_transaction() Hypothesis strategy.
    """
    # Initialize counters
    by_stablecoin = {
        coin: {"transaction_count": 0, "total_volume": Decimal("0")}
        for coin in STABLECOINS
    }
    by_activity_type = {at: 0 for at in ACTIVITY_TYPES}
    by_chain = {chain: 0 for chain in CHAINS}
    
    # Aggregate from transactions
    # Note: No defensive validation here as inputs are guaranteed by Hypothesis strategies
    for tx in transactions:
        coin = tx["stablecoin"]
        activity = tx["activity_type"]
        chain = tx["chain"]
        amount = Decimal(tx["amount"])
        
        by_stablecoin[coin]["transaction_count"] += 1
        by_stablecoin[coin]["total_volume"] += amount
        by_activity_type[activity] += 1
        by_chain[chain] += 1
    
    # Convert Decimal volumes to strings for JSON compatibility
    for coin in by_stablecoin:
        by_stablecoin[coin]["total_volume"] = str(
            by_stablecoin[coin]["total_volume"]
        )
    
    return {
        "by_stablecoin": by_stablecoin,
        "by_activity_type": by_activity_type,
        "by_chain": by_chain,
    }


@st.composite
def valid_json_export(draw):
    """Generate a complete valid JSON export structure."""
    transactions = draw(st.lists(valid_transaction(), min_size=0, max_size=10))
    holders = draw(st.lists(valid_holder(), min_size=0, max_size=10))

    metadata = draw(valid_metadata())
    metadata["total_records"] = len(transactions) + len(holders)
    
    # Compute summary from actual transaction data for consistency
    summary = compute_summary_from_data(transactions)

    return {
        "metadata": metadata,
        "summary": summary,
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


class TestActivityAnalysis:
    """Tests for activity type analysis functions."""

    @settings(max_examples=100, deadline=None)
    @given(transactions=st.lists(valid_transaction(), min_size=0, max_size=50))
    def test_property_2_grouping_preserves_totals(self, transactions):
        """
        **Feature: stablecoin-analysis-notebook, Property 2: Grouping preserves totals**

        For any transactions DataFrame, grouping by any dimension (activity_type,
        stablecoin, chain) and summing counts SHALL equal the total transaction count,
        and summing volumes SHALL equal the total volume.

        **Validates: Requirements 2.1, 3.1, 6.1**
        """
        import pandas as pd
        from stablecoin_analysis_functions import analyze_activity_types

        # Convert transactions to DataFrame
        if not transactions:
            df = pd.DataFrame(columns=[
                "transaction_hash", "block_number", "timestamp",
                "from_address", "to_address", "amount", "stablecoin",
                "chain", "activity_type", "source_explorer"
            ])
        else:
            df = pd.DataFrame(transactions)
            # Convert amount strings to Decimal
            df["amount"] = df["amount"].apply(Decimal)

        total_count = len(df)
        total_volume = df["amount"].sum() if not df.empty else Decimal("0")

        # Test grouping by activity_type
        breakdown = analyze_activity_types(df)
        sum_counts = sum(breakdown.counts.values())
        sum_volumes = sum(breakdown.volumes.values())

        assert sum_counts == total_count, (
            f"Sum of counts by activity_type ({sum_counts}) "
            f"!= total count ({total_count})"
        )
        assert sum_volumes == total_volume, (
            f"Sum of volumes by activity_type ({sum_volumes}) "
            f"!= total volume ({total_volume})"
        )

        # Test grouping by stablecoin preserves totals
        if not df.empty:
            by_stablecoin_count = df.groupby("stablecoin").size().sum()
            by_stablecoin_volume = df.groupby("stablecoin")["amount"].sum().sum()
            assert by_stablecoin_count == total_count, (
                f"Sum of counts by stablecoin ({by_stablecoin_count}) "
                f"!= total count ({total_count})"
            )
            assert by_stablecoin_volume == total_volume, (
                f"Sum of volumes by stablecoin ({by_stablecoin_volume}) "
                f"!= total volume ({total_volume})"
            )

        # Test grouping by chain preserves totals
        if not df.empty:
            by_chain_count = df.groupby("chain").size().sum()
            by_chain_volume = df.groupby("chain")["amount"].sum().sum()
            assert by_chain_count == total_count, (
                f"Sum of counts by chain ({by_chain_count}) "
                f"!= total count ({total_count})"
            )
            assert by_chain_volume == total_volume, (
                f"Sum of volumes by chain ({by_chain_volume}) "
                f"!= total volume ({total_volume})"
            )

    @settings(max_examples=100, deadline=None)
    @given(transactions=st.lists(valid_transaction(), min_size=1, max_size=50))
    def test_property_3_percentages_sum_to_100(self, transactions):
        """
        **Feature: stablecoin-analysis-notebook, Property 3: Percentages sum to 100**

        For any percentage breakdown calculation (activity types, holder
        classifications), the sum of all percentages SHALL equal 100%
        (within floating-point tolerance).

        **Validates: Requirements 2.1, 4.1**
        """
        import pandas as pd
        from stablecoin_analysis_functions import analyze_activity_types

        # Convert transactions to DataFrame
        df = pd.DataFrame(transactions)
        # Convert amount strings to Decimal
        df["amount"] = df["amount"].apply(Decimal)

        # Analyze activity types
        breakdown = analyze_activity_types(df)

        # Sum of count percentages should equal 100%
        # (count is always > 0 when transactions list is non-empty)
        sum_count_pct = sum(breakdown.percentages.values())
        assert abs(sum_count_pct - 100.0) < 0.01, (
            f"Sum of count percentages ({sum_count_pct}) != 100%"
        )

        # Sum of volume percentages should equal 100% when total volume > 0
        # When total volume is 0, percentages are all 0% (mathematically correct)
        total_volume = sum(breakdown.volumes.values())
        sum_volume_pct = sum(breakdown.volume_percentages.values())
        if total_volume > 0:
            assert abs(sum_volume_pct - 100.0) < 0.01, (
                f"Sum of volume percentages ({sum_volume_pct}) != 100%"
            )
        else:
            # When total volume is 0, all percentages should be 0
            assert sum_volume_pct == 0.0, (
                f"When total volume is 0, volume percentages should sum to 0, "
                f"got {sum_volume_pct}"
            )


class TestStablecoinComparison:
    """Tests for stablecoin comparison analysis functions."""

    @settings(max_examples=100, deadline=None)
    @given(transactions=st.lists(valid_transaction(), min_size=0, max_size=50))
    def test_property_4_volume_calculation_consistency(self, transactions):
        """
        **Feature: stablecoin-analysis-notebook, Property 4: Volume calculation consistency**

        For any transactions DataFrame, the sum of volumes by activity type
        SHALL equal the sum of volumes by stablecoin SHALL equal the sum of
        volumes by chain SHALL equal total volume.

        **Validates: Requirements 2.3, 3.1, 6.1**
        """
        import pandas as pd
        from stablecoin_analysis_functions import (
            calculate_volume_by_dimension,
        )

        # Convert transactions to DataFrame
        if not transactions:
            df = pd.DataFrame(columns=[
                "transaction_hash", "block_number", "timestamp",
                "from_address", "to_address", "amount", "stablecoin",
                "chain", "activity_type", "source_explorer"
            ])
        else:
            df = pd.DataFrame(transactions)
            # Convert amount strings to Decimal
            df["amount"] = df["amount"].apply(Decimal)

        # Calculate total volume directly
        total_volume = Decimal("0")
        if not df.empty:
            for amt in df["amount"].dropna():
                total_volume += amt

        # Calculate volumes by each dimension
        vol_by_activity = calculate_volume_by_dimension(df, "activity_type")
        vol_by_stablecoin = calculate_volume_by_dimension(df, "stablecoin")
        vol_by_chain = calculate_volume_by_dimension(df, "chain")

        # Sum volumes for each dimension
        sum_by_activity = sum(vol_by_activity.values())
        sum_by_stablecoin = sum(vol_by_stablecoin.values())
        sum_by_chain = sum(vol_by_chain.values())

        # All sums should equal total volume
        assert sum_by_activity == total_volume, (
            f"Sum of volumes by activity_type ({sum_by_activity}) "
            f"!= total volume ({total_volume})"
        )
        assert sum_by_stablecoin == total_volume, (
            f"Sum of volumes by stablecoin ({sum_by_stablecoin}) "
            f"!= total volume ({total_volume})"
        )
        assert sum_by_chain == total_volume, (
            f"Sum of volumes by chain ({sum_by_chain}) "
            f"!= total volume ({total_volume})"
        )

        # Cross-check: all dimension sums should be equal
        assert sum_by_activity == sum_by_stablecoin == sum_by_chain, (
            f"Volume sums differ: activity={sum_by_activity}, "
            f"stablecoin={sum_by_stablecoin}, chain={sum_by_chain}"
        )

    @settings(max_examples=100, deadline=None)
    @given(transactions=st.lists(valid_transaction(), min_size=1, max_size=50))
    def test_property_5_average_calculation_correctness(self, transactions):
        """
        **Feature: stablecoin-analysis-notebook, Property 5: Average calculation correctness**

        For any non-empty group of transactions, the calculated average
        transaction size SHALL equal the sum of amounts divided by the
        count of transactions.

        **Validates: Requirements 3.3, 6.3**
        """
        import pandas as pd
        from stablecoin_analysis_functions import (
            calculate_average_transaction_size,
            analyze_by_stablecoin,
        )

        # Convert transactions to DataFrame
        df = pd.DataFrame(transactions)
        # Convert amount strings to Decimal
        df["amount"] = df["amount"].apply(Decimal)

        # Test overall average
        total_sum = sum(df["amount"].dropna())
        count = len(df["amount"].dropna())
        expected_avg = total_sum / count if count > 0 else Decimal("0")

        avg_result = calculate_average_transaction_size(df)
        assert "total" in avg_result, "Expected 'total' key in result"
        assert avg_result["total"] == expected_avg, (
            f"Overall average ({avg_result['total']}) != "
            f"expected ({expected_avg})"
        )

        # Test average by stablecoin
        avg_by_stablecoin = calculate_average_transaction_size(
            df, group_by="stablecoin"
        )
        for coin in df["stablecoin"].unique():
            coin_df = df[df["stablecoin"] == coin]
            coin_sum = sum(coin_df["amount"].dropna())
            coin_count = len(coin_df["amount"].dropna())
            expected_coin_avg = (
                coin_sum / coin_count if coin_count > 0 else Decimal("0")
            )
            assert coin in avg_by_stablecoin, (
                f"Expected stablecoin '{coin}' in result"
            )
            assert avg_by_stablecoin[coin] == expected_coin_avg, (
                f"Average for {coin} ({avg_by_stablecoin[coin]}) != "
                f"expected ({expected_coin_avg})"
            )

        # Test via analyze_by_stablecoin function
        comparison = analyze_by_stablecoin(df)
        for coin in df["stablecoin"].unique():
            coin_df = df[df["stablecoin"] == coin]
            coin_sum = sum(coin_df["amount"].dropna())
            coin_count = len(coin_df)
            expected_coin_avg = (
                coin_sum / coin_count if coin_count > 0 else Decimal("0")
            )
            assert comparison.by_stablecoin[coin].avg_transaction_size == \
                expected_coin_avg, (
                    f"analyze_by_stablecoin avg for {coin} "
                    f"({comparison.by_stablecoin[coin].avg_transaction_size}) "
                    f"!= expected ({expected_coin_avg})"
                )


class TestTimeSeriesAnalysis:
    """Tests for time series analysis functions."""

    @settings(max_examples=100, deadline=None)
    @given(
        transactions=st.lists(valid_transaction(), min_size=0, max_size=50),
        aggregation=st.sampled_from(["daily", "weekly", "monthly"])
    )
    def test_property_8_time_aggregation_preserves_totals(
        self, transactions, aggregation
    ):
        """
        **Feature: stablecoin-analysis-notebook, Property 8: Time aggregation
        preserves totals**

        For any transactions DataFrame and aggregation period
        (daily/weekly/monthly), the sum of aggregated counts SHALL equal
        total transaction count.

        **Validates: Requirements 5.1, 5.4**
        """
        import pandas as pd
        from stablecoin_analysis_functions import get_time_series_totals

        # Convert transactions to DataFrame
        if not transactions:
            df = pd.DataFrame(columns=[
                "transaction_hash", "block_number", "timestamp",
                "from_address", "to_address", "amount", "stablecoin",
                "chain", "activity_type", "source_explorer"
            ])
        else:
            df = pd.DataFrame(transactions)
            # Convert amount strings to Decimal
            df["amount"] = df["amount"].apply(Decimal)

        # Get time series with totals
        result = get_time_series_totals(df, aggregation)

        # Verify total count is preserved
        aggregated_count = result.aggregated_df["transaction_count"].sum()
        assert aggregated_count == result.total_count, (
            f"Sum of aggregated counts ({aggregated_count}) "
            f"!= total count ({result.total_count}) "
            f"for aggregation '{aggregation}'"
        )

        # Verify total volume is preserved
        aggregated_volume = Decimal("0")
        for vol in result.aggregated_df["volume"]:
            if isinstance(vol, Decimal):
                aggregated_volume += vol
            else:
                aggregated_volume += Decimal(str(vol))

        assert aggregated_volume == result.total_volume, (
            f"Sum of aggregated volumes ({aggregated_volume}) "
            f"!= total volume ({result.total_volume}) "
            f"for aggregation '{aggregation}'"
        )


class TestHolderAnalysis:
    """Tests for holder behavior analysis functions."""

    @settings(max_examples=100, deadline=None)
    @given(holders=st.lists(valid_holder(), min_size=0, max_size=50))
    def test_property_6_holder_classification_consistency(self, holders):
        """
        **Feature: stablecoin-analysis-notebook, Property 6: Holder classification consistency**

        For any holders DataFrame, the count of is_store_of_value=True plus
        is_store_of_value=False SHALL equal total holder count.

        **Validates: Requirements 4.1**
        """
        import pandas as pd
        from stablecoin_analysis_functions import analyze_holders

        # Convert holders to DataFrame
        if not holders:
            df = pd.DataFrame(columns=[
                "address", "balance", "stablecoin", "chain",
                "first_seen", "last_activity", "is_store_of_value",
                "source_explorer", "holding_period_days"
            ])
        else:
            df = pd.DataFrame(holders)
            # Convert balance strings to Decimal
            df["balance"] = df["balance"].apply(Decimal)
            # Add holding_period_days if not present
            if "holding_period_days" not in df.columns:
                df["holding_period_days"] = (
                    pd.to_datetime(df["last_activity"], format='ISO8601') - 
                    pd.to_datetime(df["first_seen"], format='ISO8601')
                ).dt.days

        total_holders = len(df)

        # Count SoV and non-SoV holders directly
        if not df.empty:
            sov_true_count = df["is_store_of_value"].sum()
            sov_false_count = (~df["is_store_of_value"]).sum()
        else:
            sov_true_count = 0
            sov_false_count = 0

        # Verify counts add up to total
        assert sov_true_count + sov_false_count == total_holders, (
            f"SoV True ({sov_true_count}) + SoV False ({sov_false_count}) "
            f"!= total holders ({total_holders})"
        )

        # Verify via analyze_holders function
        metrics = analyze_holders(df)
        assert metrics.total_holders == total_holders, (
            f"analyze_holders total ({metrics.total_holders}) "
            f"!= expected ({total_holders})"
        )
        assert metrics.sov_count == sov_true_count, (
            f"analyze_holders sov_count ({metrics.sov_count}) "
            f"!= expected ({sov_true_count})"
        )

        # Verify percentage calculation
        if total_holders > 0:
            expected_pct = sov_true_count / total_holders * 100.0
            assert abs(metrics.sov_percentage - expected_pct) < 0.01, (
                f"SoV percentage ({metrics.sov_percentage}) "
                f"!= expected ({expected_pct})"
            )

    @settings(max_examples=100, deadline=None)
    @given(
        holders=st.lists(valid_holder(), min_size=0, max_size=50),
        n=st.integers(min_value=1, max_value=20)
    )
    def test_property_7_top_n_ordering_correctness(self, holders, n):
        """
        **Feature: stablecoin-analysis-notebook, Property 7: Top-N ordering correctness**

        For any holders DataFrame and N <= total holders, the top N holders
        by balance SHALL be sorted in descending order by balance.

        **Validates: Requirements 4.4**
        """
        import pandas as pd
        from stablecoin_analysis_functions import get_top_holders

        # Convert holders to DataFrame
        if not holders:
            df = pd.DataFrame(columns=[
                "address", "balance", "stablecoin", "chain",
                "first_seen", "last_activity", "is_store_of_value",
                "source_explorer"
            ])
        else:
            df = pd.DataFrame(holders)
            # Convert balance strings to Decimal
            df["balance"] = df["balance"].apply(Decimal)

        # Get top N holders
        top_holders = get_top_holders(df, n=n)

        # Verify the result count
        expected_count = min(n, len(df))
        assert len(top_holders) == expected_count, (
            f"Expected {expected_count} top holders, got {len(top_holders)}"
        )

        # Verify descending order by balance
        if len(top_holders) > 1:
            for i in range(len(top_holders) - 1):
                assert top_holders[i].balance >= top_holders[i + 1].balance, (
                    f"Top holders not in descending order: "
                    f"holder[{i}].balance ({top_holders[i].balance}) < "
                    f"holder[{i+1}].balance ({top_holders[i + 1].balance})"
                )

        # Verify that returned holders have the highest balances
        if not df.empty and len(top_holders) > 0:
            # Get all balances sorted descending
            all_balances = sorted(df["balance"].tolist(), reverse=True)
            top_n_balances = all_balances[:expected_count]

            # The returned balances should match the top N balances
            returned_balances = sorted(
                [h.balance for h in top_holders], reverse=True
            )
            assert returned_balances == top_n_balances, (
                f"Returned balances {returned_balances} don't match "
                f"expected top {expected_count} balances {top_n_balances}"
            )


# =============================================================================
# Sample Data Generator Tests
# =============================================================================

class TestSampleDataGenerator:
    """Tests for sample data generation functions."""

    @settings(max_examples=100, deadline=None)
    @given(
        num_transactions=st.integers(min_value=0, max_value=100),
        num_holders=st.integers(min_value=0, max_value=50),
        sov_ratio=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        seed=st.integers(min_value=0, max_value=1000000),
    )
    def test_property_9_sample_data_schema_compliance(
        self, num_transactions, num_holders, sov_ratio, seed
    ):
        """
        **Feature: stablecoin-analysis-notebook, Property 9: Sample data schema compliance**

        For any sample data configuration, generated data SHALL pass the same
        schema validation as real exported data.

        **Validates: Requirements 8.2**
        """
        from sample_data_generator import SampleDataConfig, generate_sample_json

        # Create configuration
        config = SampleDataConfig(
            num_transactions=num_transactions,
            num_holders=num_holders,
            sov_ratio=sov_ratio,
            seed=seed,
        )

        # Generate sample data as JSON
        json_data = generate_sample_json(config)

        # Validate against schema
        is_valid, errors = validate_schema(json_data)

        assert is_valid, (
            f"Generated sample data failed schema validation. "
            f"Config: num_transactions={num_transactions}, "
            f"num_holders={num_holders}, sov_ratio={sov_ratio}. "
            f"Errors: {errors}"
        )

        # Verify all required top-level fields are present
        assert "metadata" in json_data
        assert "summary" in json_data
        assert "transactions" in json_data
        assert "holders" in json_data

        # Verify metadata fields
        assert "run_id" in json_data["metadata"]
        assert "collection_timestamp" in json_data["metadata"]
        assert "agent_version" in json_data["metadata"]
        assert "explorers_queried" in json_data["metadata"]
        assert "total_records" in json_data["metadata"]

        # Verify transaction count matches
        assert len(json_data["transactions"]) == num_transactions, (
            f"Expected {num_transactions} transactions, "
            f"got {len(json_data['transactions'])}"
        )

        # Verify holder count matches
        assert len(json_data["holders"]) == num_holders, (
            f"Expected {num_holders} holders, "
            f"got {len(json_data['holders'])}"
        )

    @settings(max_examples=100, deadline=None)
    @given(
        num_transactions=st.integers(min_value=0, max_value=100),
        num_holders=st.integers(min_value=0, max_value=50),
        sov_ratio=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        seed=st.integers(min_value=0, max_value=1000000),
    )
    def test_property_10_sample_data_respects_configuration(
        self, num_transactions, num_holders, sov_ratio, seed
    ):
        """
        **Feature: stablecoin-analysis-notebook, Property 10: Sample data respects configuration**

        For any sample data configuration specifying N transactions and M holders,
        generated data SHALL contain exactly N transactions and M holders.

        **Validates: Requirements 8.4**
        """
        from sample_data_generator import SampleDataConfig, generate_sample_data

        # Create configuration
        config = SampleDataConfig(
            num_transactions=num_transactions,
            num_holders=num_holders,
            sov_ratio=sov_ratio,
            seed=seed,
        )

        # Generate sample data
        data = generate_sample_data(config)

        # Verify transaction count
        assert len(data.transactions_df) == num_transactions, (
            f"Expected {num_transactions} transactions, "
            f"got {len(data.transactions_df)}"
        )

        # Verify holder count
        assert len(data.holders_df) == num_holders, (
            f"Expected {num_holders} holders, "
            f"got {len(data.holders_df)}"
        )

        # Verify is_sample_data flag is set
        assert data.is_sample_data is True, (
            "Generated data should have is_sample_data=True"
        )

        # Verify SoV ratio is approximately correct (within tolerance)
        if num_holders > 0:
            actual_sov_count = data.holders_df["is_store_of_value"].sum()
            expected_sov_count = int(num_holders * sov_ratio)
            
            # Allow for rounding differences
            assert actual_sov_count == expected_sov_count, (
                f"Expected {expected_sov_count} SoV holders "
                f"(ratio={sov_ratio}), got {actual_sov_count}"
            )

        # Verify metadata total_records is correct
        expected_total = num_transactions + num_holders
        assert data.metadata["total_records"] == expected_total, (
            f"Expected total_records={expected_total}, "
            f"got {data.metadata['total_records']}"
        )



# =============================================================================
# Confidence and Conclusion Tests
# =============================================================================

class TestConfidenceCalculation:
    """Tests for confidence calculation functions."""

    @settings(max_examples=100, deadline=None)
    @given(
        num_transactions=st.integers(min_value=0, max_value=2000),
        num_holders=st.integers(min_value=0, max_value=100),
        seed=st.integers(min_value=0, max_value=1000000),
    )
    def test_property_11_confidence_calculation_bounds(
        self, num_transactions, num_holders, seed
    ):
        """
        **Feature: stablecoin-analysis-notebook, Property 11: Confidence calculation bounds**

        For any dataset, the calculated confidence indicator SHALL be a valid
        ConfidenceLevel enum value (HIGH, MEDIUM, or LOW) based on the formula:
        confidence_score = 0.6 × min(sample_size/1000, 1.0) + 0.4 × completeness_percent,
        mapped via ConfidenceLevel.from_score() where HIGH (score ≥ 0.85),
        MEDIUM (0.50 ≤ score < 0.85), LOW (score < 0.50).

        **Validates: Requirements 7.3**
        """
        from sample_data_generator import SampleDataConfig, generate_sample_data
        from stablecoin_analysis_functions import (
            calculate_confidence, ConfidenceLevel, SUPPORTED_CHAIN_COUNT
        )

        # Generate sample data
        config = SampleDataConfig(
            num_transactions=num_transactions,
            num_holders=num_holders,
            seed=seed,
        )
        data = generate_sample_data(config)

        # Calculate confidence
        metrics = calculate_confidence(data.transactions_df)

        # Verify confidence_level is a valid enum value
        assert isinstance(metrics.confidence_level, ConfidenceLevel), (
            f"confidence_level should be ConfidenceLevel enum, "
            f"got {type(metrics.confidence_level)}"
        )
        assert metrics.confidence_level in [
            ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW
        ], f"Invalid confidence level: {metrics.confidence_level}"

        # Verify confidence_score is in valid range [0.0, 1.0]
        assert 0.0 <= metrics.confidence_score <= 1.0, (
            f"confidence_score should be in [0.0, 1.0], "
            f"got {metrics.confidence_score}"
        )

        # Verify the formula is correctly applied
        expected_normalized_sample = min(num_transactions / 1000, 1.0)
        expected_score = (
            0.6 * expected_normalized_sample +
            0.4 * metrics.completeness_percent
        )
        assert abs(metrics.confidence_score - expected_score) < 0.001, (
            f"Confidence score {metrics.confidence_score} doesn't match "
            f"expected {expected_score} from formula"
        )

        # Verify threshold mapping
        if metrics.confidence_score >= 0.85:
            assert metrics.confidence_level == ConfidenceLevel.HIGH, (
                f"Score {metrics.confidence_score} >= 0.85 should be HIGH, "
                f"got {metrics.confidence_level}"
            )
        elif metrics.confidence_score >= 0.50:
            assert metrics.confidence_level == ConfidenceLevel.MEDIUM, (
                f"Score {metrics.confidence_score} in [0.50, 0.85) should be MEDIUM, "
                f"got {metrics.confidence_level}"
            )
        else:
            assert metrics.confidence_level == ConfidenceLevel.LOW, (
                f"Score {metrics.confidence_score} < 0.50 should be LOW, "
                f"got {metrics.confidence_level}"
            )

        # Verify component metrics are in valid ranges
        assert 0.0 <= metrics.field_completeness <= 1.0, (
            f"field_completeness should be in [0.0, 1.0], "
            f"got {metrics.field_completeness}"
        )
        assert 0.0 <= metrics.chain_coverage <= 1.0, (
            f"chain_coverage should be in [0.0, 1.0], "
            f"got {metrics.chain_coverage}"
        )
        assert 0 <= metrics.chains_with_data <= SUPPORTED_CHAIN_COUNT, (
            f"chains_with_data should be in [0, {SUPPORTED_CHAIN_COUNT}], "
            f"got {metrics.chains_with_data}"
        )
        assert 0.0 <= metrics.completeness_percent <= 1.0, (
            f"completeness_percent should be in [0.0, 1.0], "
            f"got {metrics.completeness_percent}"
        )


class TestErrorDetection:
    """Tests for error detection and warning generation."""

    @settings(max_examples=100, deadline=None)
    @given(
        num_transactions=st.integers(min_value=10, max_value=100),
        num_errors=st.integers(min_value=1, max_value=10),
        seed=st.integers(min_value=0, max_value=1000000),
    )
    def test_property_12_error_detection_completeness(
        self, num_transactions, num_errors, seed
    ):
        """
        **Feature: stablecoin-analysis-notebook, Property 12: Error detection completeness**

        For any JSON data containing an "errors" array with non-empty entries,
        the data quality warnings SHALL include at least one warning.

        **Validates: Requirements 7.4**
        """
        from sample_data_generator import SampleDataConfig, generate_sample_data
        from stablecoin_analysis_functions import (
            calculate_confidence, get_data_quality_warnings
        )

        # Generate sample data
        config = SampleDataConfig(
            num_transactions=num_transactions,
            num_holders=10,
            seed=seed,
        )
        data = generate_sample_data(config)

        # Add errors to the data
        errors = [f"Test error {i}" for i in range(num_errors)]

        # Calculate confidence
        confidence = calculate_confidence(data.transactions_df)

        # Get warnings with errors
        warnings = get_data_quality_warnings(errors, confidence)

        # Verify at least one warning is generated when errors are present
        assert len(warnings) >= 1, (
            f"Expected at least 1 warning when {num_errors} errors present, "
            f"got {len(warnings)} warnings"
        )

        # Verify the warning mentions the errors
        error_warning_found = any(
            "error" in w.lower() for w in warnings
        )
        assert error_warning_found, (
            f"Expected a warning mentioning errors, but none found. "
            f"Warnings: {warnings}"
        )

    @settings(max_examples=50, deadline=None)
    @given(
        num_transactions=st.integers(min_value=0, max_value=50),
        seed=st.integers(min_value=0, max_value=1000000),
    )
    def test_small_sample_size_warning(self, num_transactions, seed):
        """
        Test that small sample sizes generate appropriate warnings.
        """
        from sample_data_generator import SampleDataConfig, generate_sample_data
        from stablecoin_analysis_functions import (
            calculate_confidence, get_data_quality_warnings
        )

        # Generate sample data with small sample size
        config = SampleDataConfig(
            num_transactions=num_transactions,
            num_holders=5,
            seed=seed,
        )
        data = generate_sample_data(config)

        # Calculate confidence
        confidence = calculate_confidence(data.transactions_df)

        # Get warnings
        warnings = get_data_quality_warnings([], confidence)

        # If sample size < 100, should have a warning
        if num_transactions < 100:
            sample_warning_found = any(
                "sample size" in w.lower() for w in warnings
            )
            assert sample_warning_found, (
                f"Expected sample size warning for {num_transactions} transactions, "
                f"but none found. Warnings: {warnings}"
            )
