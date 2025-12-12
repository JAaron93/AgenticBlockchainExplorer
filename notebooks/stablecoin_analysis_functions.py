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
            # Validate is_store_of_value is boolean or numeric
            if not (pd.api.types.is_bool_dtype(holders_df['is_store_of_value']) or 
                    pd.api.types.is_numeric_dtype(holders_df['is_store_of_value'])):
                raise ValueError(
                    f"Column 'is_store_of_value' must be boolean or numeric, "
                    f"got {holders_df['is_store_of_value'].dtype}"
                )
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

    for value in df[dimension].dropna().unique():
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

        for value in df[group_by].dropna().unique():
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


# =============================================================================
# Holder Analysis Functions
# =============================================================================


@dataclass
class HolderMetrics:
    """Holder behavior metrics.

    Attributes:
        total_holders: Total number of holders in the dataset
        sov_count: Number of holders classified as store_of_value
        sov_percentage: Percentage of holders that are store_of_value
        avg_balance_sov: Average balance of store_of_value holders
        avg_balance_active: Average balance of active (non-SoV) holders
        avg_holding_period_days: Average holding period for SoV holders
        median_holding_period_days: Median holding period for SoV holders
    """
    total_holders: int
    sov_count: int
    sov_percentage: float
    avg_balance_sov: Decimal
    avg_balance_active: Decimal
    avg_holding_period_days: float
    median_holding_period_days: float


@dataclass
class TopHolder:
    """Information about a top holder.

    Attributes:
        address: Wallet address
        balance: Current token balance
        stablecoin: Token type (USDC/USDT)
        chain: Blockchain network
        is_store_of_value: SoV classification
    """
    address: str
    balance: Decimal
    stablecoin: str
    chain: str
    is_store_of_value: bool


def analyze_holders(
    holders_df: pd.DataFrame,
    transactions_df: Optional[pd.DataFrame] = None,
) -> HolderMetrics:
    """
    Analyze holder behavior patterns.

    Calculates SoV percentage, average balances, and holding periods.

    Args:
        holders_df: DataFrame with holder data including 'is_store_of_value',
                    'balance', and 'holding_period_days' columns
        transactions_df: Optional DataFrame with transaction data (unused
                         in current implementation, reserved for future use)

    Returns:
        HolderMetrics with calculated statistics

    Requirements: 4.1, 4.3
    """
    # Validate required columns
    required_cols = {'is_store_of_value', 'balance'}
    if not required_cols.issubset(holders_df.columns):
        missing = required_cols - set(holders_df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Handle empty DataFrame
    if holders_df.empty:
        return HolderMetrics(
            total_holders=0,
            sov_count=0,
            sov_percentage=0.0,
            avg_balance_sov=Decimal("0"),
            avg_balance_active=Decimal("0"),
            avg_holding_period_days=0.0,
            median_holding_period_days=0.0,
        )

    total_holders = len(holders_df)

    # Count SoV holders - handle both boolean and numeric types
    sov_mask = holders_df["is_store_of_value"].astype(bool)
    sov_count = int(sov_mask.sum())

    # Calculate SoV percentage
    sov_percentage = (sov_count / total_holders * 100.0) if total_holders > 0 else 0.0

    # Calculate average balance for SoV holders
    sov_holders = holders_df[sov_mask]
    avg_balance_sov = Decimal("0")
    if len(sov_holders) > 0:
        total_sov_balance = Decimal("0")
        valid_balances = sov_holders["balance"].dropna()
        for bal in valid_balances:
            if isinstance(bal, Decimal):
                total_sov_balance += bal
            else:
                total_sov_balance += Decimal(str(bal))
        avg_balance_sov = total_sov_balance / len(sov_holders)

    # Calculate average balance for active (non-SoV) holders
    active_holders = holders_df[~sov_mask]
    avg_balance_active = Decimal("0")
    if len(active_holders) > 0:
        total_active_balance = Decimal("0")
        for bal in active_holders["balance"].dropna():
            if isinstance(bal, Decimal):
                total_active_balance += bal
            else:
                total_active_balance += Decimal(str(bal))
        avg_balance_active = total_active_balance / len(active_holders)

    # Calculate holding period statistics for SoV holders
    avg_holding_period_days = 0.0
    median_holding_period_days = 0.0

    if "holding_period_days" in holders_df.columns and len(sov_holders) > 0:
        sov_holding_periods = sov_holders["holding_period_days"].dropna()
        if len(sov_holding_periods) > 0:
            avg_holding_period_days = float(sov_holding_periods.mean())
            median_holding_period_days = float(sov_holding_periods.median())

    return HolderMetrics(
        total_holders=total_holders,
        sov_count=sov_count,
        sov_percentage=sov_percentage,
        avg_balance_sov=avg_balance_sov,
        avg_balance_active=avg_balance_active,
        avg_holding_period_days=avg_holding_period_days,
        median_holding_period_days=median_holding_period_days,
    )


def get_top_holders(
    holders_df: pd.DataFrame,
    n: int = 10,
) -> list:
    """
    Get top N holders by balance globally (across all stablecoins and chains).

    Args:
        holders_df: DataFrame with holder data
        n: Number of top holders to return (default: 10)

    Returns:
        List of TopHolder objects sorted by descending balance

    Requirements: 4.4
    """
    # Validate required columns
    required_cols = {
        'address', 'balance', 'stablecoin', 'chain', 'is_store_of_value'
    }
    if not required_cols.issubset(holders_df.columns):
        missing = required_cols - set(holders_df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    if holders_df.empty or n <= 0:
        return []

    # Convert balance to Decimal for proper sorting if not already
    df = holders_df.copy()
    if not df.empty:
        df["_balance_decimal"] = df["balance"].apply(
            lambda x: (
                x if isinstance(x, Decimal)
                else Decimal("0") if pd.isna(x)
                else Decimal(str(x))
            )
        )

    # Sort by balance descending and take top N
    sorted_df = df.sort_values("_balance_decimal", ascending=False).head(n)

    # Convert to list of TopHolder objects
    top_holders = []
    for _, row in sorted_df.iterrows():
        top_holders.append(TopHolder(
            address=row["address"],
            balance=row["_balance_decimal"],
            stablecoin=row["stablecoin"],
            chain=row["chain"],
            is_store_of_value=bool(row["is_store_of_value"]),
        ))

    return top_holders


# =============================================================================
# Time Series Analysis Functions
# =============================================================================


# Valid aggregation periods
AGGREGATION_PERIODS = ["daily", "weekly", "monthly"]


@dataclass
class TimeSeriesResult:
    """Time series aggregation result.

    Attributes:
        aggregated_df: DataFrame with time-aggregated data
        aggregation: The aggregation period used
        total_count: Total transaction count (should match sum of aggregated)
        total_volume: Total volume (should match sum of aggregated)
    """
    aggregated_df: pd.DataFrame
    aggregation: str
    total_count: int
    total_volume: Decimal


def analyze_time_series(
    df: pd.DataFrame,
    aggregation: str = "daily",
) -> pd.DataFrame:
    """
    Create time-series aggregations from transactions DataFrame.

    Aggregates transaction counts and volumes by time period, grouped by
    activity type and stablecoin.

    Args:
        df: Transactions DataFrame with 'timestamp', 'amount',
            'activity_type', and 'stablecoin' columns
        aggregation: Aggregation period - "daily", "weekly", or "monthly"

    Returns:
        DataFrame with columns:
        - period: The time period (date)
        - activity_type: Activity type
        - stablecoin: Stablecoin type
        - transaction_count: Number of transactions in period
        - volume: Total volume in period

    Requirements: 5.1, 5.4
    """
    # Validate aggregation parameter
    if aggregation not in AGGREGATION_PERIODS:
        raise ValueError(
            f"Invalid aggregation '{aggregation}'. "
            f"Must be one of: {AGGREGATION_PERIODS}"
        )

    # Validate required columns
    required_cols = {'timestamp', 'amount', 'activity_type', 'stablecoin'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Handle empty DataFrame
    if df.empty:
        return pd.DataFrame(columns=[
            'period', 'activity_type', 'stablecoin',
            'transaction_count', 'volume'
        ])

    # Create a working copy
    work_df = df.copy()

    # Ensure timestamp is datetime
    _ensure_datetime(work_df, 'timestamp')

    # Create period column based on aggregation
    if aggregation == "daily":
        work_df['period'] = work_df['timestamp'].dt.date
    elif aggregation == "weekly":
        # Use Monday as start of week
        work_df['period'] = (
            work_df['timestamp'].dt.to_period('W-MON').dt.start_time.dt.date
        )
    elif aggregation == "monthly":
        work_df['period'] = (
            work_df['timestamp'].dt.to_period('M').dt.start_time.dt.date
        )

    # Group by period, activity_type, and stablecoin
    grouped = work_df.groupby(
        ['period', 'activity_type', 'stablecoin'],
        as_index=False
    ).agg(
        transaction_count=('timestamp', 'count'),
        volume=('amount', _sum_decimal_amounts)
    )

    # Sort by period
    grouped = grouped.sort_values('period').reset_index(drop=True)

    return grouped


def _sum_decimal_amounts(amounts: pd.Series) -> Decimal:
    """
    Sum a series of amounts, handling Decimal conversion.

    Args:
        amounts: Series of amounts (Decimal or numeric)

    Returns:
        Sum as Decimal
    """
    total = Decimal("0")
    for amt in amounts.dropna():
        if isinstance(amt, Decimal):
            total += amt
        else:
            total += Decimal(str(amt))
    return total


def _ensure_datetime(df: pd.DataFrame, column: str = 'timestamp') -> None:
    """
    Ensure a column is datetime type, converting if necessary.

    Modifies the DataFrame in place.

    Args:
        df: DataFrame to modify
        column: Column name to convert
    """
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        df[column] = pd.to_datetime(df[column], format='ISO8601', utc=True)


def get_time_series_totals(
    df: pd.DataFrame,
    aggregation: str = "daily",
) -> TimeSeriesResult:
    """
    Get time series aggregation with totals for verification.

    This function wraps analyze_time_series and includes total counts
    and volumes for property testing verification.

    Args:
        df: Transactions DataFrame
        aggregation: Aggregation period

    Returns:
        TimeSeriesResult with aggregated DataFrame and totals

    Requirements: 5.1, 5.4
    """
    aggregated_df = analyze_time_series(df, aggregation)

    # Calculate totals from original data
    total_count = len(df)
    total_volume = Decimal("0")
    if not df.empty and 'amount' in df.columns:
        for amt in df['amount'].dropna():
            if isinstance(amt, Decimal):
                total_volume += amt
            else:
                total_volume += Decimal(str(amt))

    return TimeSeriesResult(
        aggregated_df=aggregated_df,
        aggregation=aggregation,
        total_count=total_count,
        total_volume=total_volume,
    )


def aggregate_time_series_by_activity(
    df: pd.DataFrame,
    aggregation: str = "daily",
) -> pd.DataFrame:
    """
    Aggregate time series by activity type only (for line charts).

    Args:
        df: Transactions DataFrame
        aggregation: Aggregation period

    Returns:
        DataFrame with period, activity_type, transaction_count, volume

    Requirements: 5.2
    """
    # Validate aggregation parameter
    if aggregation not in AGGREGATION_PERIODS:
        raise ValueError(
            f"Invalid aggregation '{aggregation}'. "
            f"Must be one of: {AGGREGATION_PERIODS}"
        )

    # Validate required columns
    required_cols = {'timestamp', 'amount', 'activity_type'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Handle empty DataFrame
    if df.empty:
        return pd.DataFrame(columns=[
            'period', 'activity_type', 'transaction_count', 'volume'
        ])

    # Create a working copy
    work_df = df.copy()

    # Ensure timestamp is datetime
    _ensure_datetime(work_df, 'timestamp')

    # Create period column based on aggregation
    if aggregation == "daily":
        work_df['period'] = work_df['timestamp'].dt.date
    elif aggregation == "weekly":
        work_df['period'] = (
            work_df['timestamp'].dt.to_period('W-MON').dt.start_time.dt.date
        )
    elif aggregation == "monthly":
        work_df['period'] = (
            work_df['timestamp'].dt.to_period('M').dt.start_time.dt.date
        )

    # Group by period and activity_type
    grouped = work_df.groupby(
        ['period', 'activity_type'],
        as_index=False
    ).agg(
        transaction_count=('timestamp', 'count'),
        volume=('amount', _sum_decimal_amounts)
    )

    return grouped.sort_values('period').reset_index(drop=True)


def aggregate_time_series_by_stablecoin(
    df: pd.DataFrame,
    aggregation: str = "daily",
) -> pd.DataFrame:
    """
    Aggregate time series by stablecoin only (for line charts).

    Args:
        df: Transactions DataFrame
        aggregation: Aggregation period

    Returns:
        DataFrame with period, stablecoin, transaction_count, volume

    Requirements: 5.3
    """
    # Validate aggregation parameter
    if aggregation not in AGGREGATION_PERIODS:
        raise ValueError(
            f"Invalid aggregation '{aggregation}'. "
            f"Must be one of: {AGGREGATION_PERIODS}"
        )

    # Validate required columns
    required_cols = {'timestamp', 'amount', 'stablecoin'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Handle empty DataFrame
    if df.empty:
        return pd.DataFrame(columns=[
            'period', 'stablecoin', 'transaction_count', 'volume'
        ])

    # Create a working copy
    work_df = df.copy()

    # Ensure timestamp is datetime
    _ensure_datetime(work_df, 'timestamp')

    # Create period column based on aggregation
    if aggregation == "daily":
        work_df['period'] = work_df['timestamp'].dt.date
    elif aggregation == "weekly":
        work_df['period'] = (
            work_df['timestamp'].dt.to_period('W-MON').dt.start_time.dt.date
        )
    elif aggregation == "monthly":
        work_df['period'] = (
            work_df['timestamp'].dt.to_period('M').dt.start_time.dt.date
        )

    # Group by period and stablecoin
    grouped = work_df.groupby(
        ['period', 'stablecoin'],
        as_index=False
    ).agg(
        transaction_count=('timestamp', 'count'),
        volume=('amount', _sum_decimal_amounts)
    )

    return grouped.sort_values('period').reset_index(drop=True)
