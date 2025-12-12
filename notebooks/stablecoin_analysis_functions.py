"""
Analysis functions for stablecoin analysis notebook.

This module provides functions to analyze stablecoin transaction data,
including activity type breakdown, holder metrics, and chain comparisons.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional

import pandas as pd


# Activity types
ACTIVITY_TYPES = ["transaction", "store_of_value", "other"]

# Supported stablecoins
SUPPORTED_STABLECOINS = ["USDC", "USDT"]

# Supported chains
SUPPORTED_CHAINS = ["ethereum", "bsc", "polygon"]


@dataclass
class ActivityBreakdown:
    """Activity type distribution metrics."""
    counts: Dict[str, int]
    percentages: Dict[str, float]
    volumes: Dict[str, Decimal]
    volume_percentages: Dict[str, float]


def analyze_activity_types(df: pd.DataFrame) -> ActivityBreakdown:
    """
    Calculate activity type distribution from transactions DataFrame.

    Args:
        df: Transactions DataFrame with 'activity_type' and 'amount' columns

    Returns:
        ActivityBreakdown with counts, percentages, volumes, and
        volume percentages by activity type

    Requirements: 2.1, 2.3
    """
    # Validate required columns
    required_columns = {'activity_type', 'amount'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Validate activity types
    if not df.empty:
        invalid_types = (
            set(df['activity_type'].dropna().unique()) - set(ACTIVITY_TYPES)
        )
        if invalid_types:
            raise ValueError(
                f"DataFrame contains invalid activity types: {invalid_types}"
            )

    # Initialize with zeros for all activity types
    counts: Dict[str, int] = {at: 0 for at in ACTIVITY_TYPES}
    volumes: Dict[str, Decimal] = {at: Decimal("0") for at in ACTIVITY_TYPES}
    percentages: Dict[str, float] = {at: 0.0 for at in ACTIVITY_TYPES}
    volume_pcts: Dict[str, float] = {at: 0.0 for at in ACTIVITY_TYPES}
    # Handle empty DataFrame
    if df.empty:
        return ActivityBreakdown(
            counts=counts,
            percentages=percentages,
            volumes=volumes,
            volume_percentages=volume_pcts,
        )

    # Calculate counts by activity type
    count_series = df.groupby("activity_type").size()
    for activity_type in ACTIVITY_TYPES:
        if activity_type in count_series.index:
            counts[activity_type] = int(count_series[activity_type])

    # Use total rows in DataFrame for percentage calculation.
    # This ensures percentages are based on all transactions, including
    # any with unknown activity types (schema validation prevents this).
    total_rows = len(df)

    # Calculate percentages based on total rows
    for activity_type in ACTIVITY_TYPES:
        if total_rows > 0:
            percentages[activity_type] = (
                counts[activity_type] / total_rows * 100.0
            )

    # Calculate volumes by activity type
    # Group by activity_type and sum amounts
    for activity_type in ACTIVITY_TYPES:
        mask = df["activity_type"] == activity_type
        if mask.any():
            # Sum amounts, handling Decimal values
            activity_amounts = df.loc[mask, "amount"]
            # Filter out None/NaN values
            valid_amounts = activity_amounts.dropna()
            if len(valid_amounts) > 0:
                total = Decimal("0")
                for a in valid_amounts:
                    if isinstance(a, Decimal):
                        total += a
                    else:
                        total += Decimal(str(a))
                volumes[activity_type] = total

    # Calculate total volume for volume percentages
    total_volume = sum(volumes.values())

    # Calculate volume percentages
    for activity_type in ACTIVITY_TYPES:
        if total_volume > 0:
            volume_pcts[activity_type] = float(
                volumes[activity_type] / total_volume * 100
            )

    return ActivityBreakdown(
        counts=counts,
        percentages=percentages,
        volumes=volumes,
        volume_percentages=volume_pcts,
    )


@dataclass
class StablecoinMetrics:
    """Metrics for a single stablecoin."""
    stablecoin: str
    transaction_count: int
    total_volume: Decimal
    avg_transaction_size: Decimal
    activity_distribution: Dict[str, float]
    sov_ratio: float  # store_of_value holders / total holders


@dataclass
class StablecoinComparison:
    """Comparison metrics across stablecoins."""
    by_stablecoin: Dict[str, StablecoinMetrics]
    total_transactions: int
    total_volume: Decimal


def analyze_by_stablecoin(
    transactions_df: pd.DataFrame,
    holders_df: Optional[pd.DataFrame] = None,
) -> StablecoinComparison:
    """
    Analyze transactions grouped by stablecoin type.

    Calculates activity distribution, average transaction size, and
    store-of-value ratio for each stablecoin.

    Args:
        transactions_df: DataFrame with transaction data
        holders_df: Optional DataFrame with holder data for SoV ratio

    Returns:
        StablecoinComparison with metrics for each stablecoin

    Requirements: 3.1, 3.3, 3.4
    """
    required_cols = {'stablecoin', 'amount', 'activity_type'}
    if not required_cols.issubset(transactions_df.columns):
        missing = required_cols - set(transactions_df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Initialize metrics for all supported stablecoins
    metrics: Dict[str, StablecoinMetrics] = {}

    for coin in SUPPORTED_STABLECOINS:
        metrics[coin] = StablecoinMetrics(
            stablecoin=coin,
            transaction_count=0,
            total_volume=Decimal("0"),
            avg_transaction_size=Decimal("0"),
            activity_distribution={at: 0.0 for at in ACTIVITY_TYPES},
            sov_ratio=0.0,
        )

    # Calculate total metrics
    total_transactions = len(transactions_df)
    total_volume = Decimal("0")
    if not transactions_df.empty:
        for amt in transactions_df["amount"].dropna():
            if isinstance(amt, Decimal):
                total_volume += amt
            else:
                total_volume += Decimal(str(amt))

    # Group by stablecoin and calculate metrics
    for coin in SUPPORTED_STABLECOINS:
        coin_mask = transactions_df["stablecoin"] == coin
        coin_df = transactions_df[coin_mask]

        if coin_df.empty:
            continue

        # Transaction count
        tx_count = len(coin_df)
        metrics[coin].transaction_count = tx_count

        # Total volume
        coin_volume = Decimal("0")
        for amt in coin_df["amount"].dropna():
            if isinstance(amt, Decimal):
                coin_volume += amt
            else:
                coin_volume += Decimal(str(amt))
        metrics[coin].total_volume = coin_volume

        # Average transaction size
        if tx_count > 0:
            metrics[coin].avg_transaction_size = coin_volume / tx_count

        # Activity distribution (percentages within this stablecoin)
        activity_counts = coin_df.groupby("activity_type").size()
        for at in ACTIVITY_TYPES:
            if at in activity_counts.index:
                metrics[coin].activity_distribution[at] = (
                    activity_counts[at] / tx_count * 100.0
                )

    # Calculate SoV ratio from holders if provided
    if holders_df is not None and not holders_df.empty:
        if 'stablecoin' in holders_df.columns and \
           'is_store_of_value' in holders_df.columns:
            for coin in SUPPORTED_STABLECOINS:
                coin_holders = holders_df[holders_df["stablecoin"] == coin]
                if len(coin_holders) > 0:
                    sov_count = coin_holders["is_store_of_value"].sum()
                    metrics[coin].sov_ratio = sov_count / len(coin_holders)

    return StablecoinComparison(
        by_stablecoin=metrics,
        total_transactions=total_transactions,
        total_volume=total_volume,
    )


def calculate_volume_by_dimension(
    df: pd.DataFrame,
    dimension: str,
) -> Dict[str, Decimal]:
    """
    Calculate total volume grouped by a dimension.

    Args:
        df: Transactions DataFrame with 'amount' column
        dimension: Column name to group by (e.g., 'activity_type',
                   'stablecoin', 'chain')

    Returns:
        Dictionary mapping dimension values to total volumes

    Requirements: 2.3, 3.1, 6.1
    """
    if dimension not in df.columns:
        raise ValueError(f"Column '{dimension}' not found in DataFrame")
    if 'amount' not in df.columns:
        raise ValueError("Column 'amount' not found in DataFrame")

    result: Dict[str, Decimal] = {}

    if df.empty:
        return result

    for value in df[dimension].unique():
        mask = df[dimension] == value
        volume = Decimal("0")
        for amt in df.loc[mask, "amount"].dropna():
            if isinstance(amt, Decimal):
                volume += amt
            else:
                volume += Decimal(str(amt))
        result[value] = volume

    return result


def calculate_average_transaction_size(
    df: pd.DataFrame,
    group_by: Optional[str] = None,
) -> Dict[str, Decimal]:
    """
    Calculate average transaction size, optionally grouped by a dimension.

    Args:
        df: Transactions DataFrame with 'amount' column
        group_by: Optional column name to group by

    Returns:
        Dictionary mapping group values to average transaction sizes.
        If group_by is None, returns {"total": average}.

    Requirements: 3.3, 6.3
    """
    if 'amount' not in df.columns:
        raise ValueError("Column 'amount' not found in DataFrame")

    result: Dict[str, Decimal] = {}

    if df.empty:
        if group_by is None:
            return {"total": Decimal("0")}
        return result

    if group_by is None:
        # Calculate overall average
        total = Decimal("0")
        count = 0
        for amt in df["amount"].dropna():
            if isinstance(amt, Decimal):
                total += amt
            else:
                total += Decimal(str(amt))
            count += 1
        if count > 0:
            result["total"] = total / count
        else:
            result["total"] = Decimal("0")
    else:
        if group_by not in df.columns:
            raise ValueError(f"Column '{group_by}' not found in DataFrame")

        for value in df[group_by].unique():
            mask = df[group_by] == value
            group_df = df.loc[mask, "amount"].dropna()
            total = Decimal("0")
            count = 0
            for amt in group_df:
                if isinstance(amt, Decimal):
                    total += amt
                else:
                    total += Decimal(str(amt))
                count += 1
            if count > 0:
                result[value] = total / count
            else:
                result[value] = Decimal("0")

    return result
