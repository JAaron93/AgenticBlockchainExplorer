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


# =============================================================================
# Chain Comparison Analysis Functions
# =============================================================================


@dataclass
class ChainMetrics:
    """Per-chain analysis metrics.

    Attributes:
        chain: Blockchain network name (ethereum, bsc, polygon)
        transaction_count: Total number of transactions on this chain
        total_volume: Total transaction volume on this chain
        avg_transaction_size: Average transaction size on this chain
        avg_gas_cost: Average gas cost in native token (ETH/BNB/MATIC),
                      None if no gas data available
        sov_ratio: Ratio of store_of_value holders to total holders
        activity_distribution: Percentage distribution by activity type
        excluded_gas_count: Number of transactions excluded from gas calculation
    """
    chain: str
    transaction_count: int
    total_volume: Decimal
    avg_transaction_size: Decimal
    avg_gas_cost: Optional[Decimal]
    sov_ratio: float
    activity_distribution: Dict[str, float]
    excluded_gas_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "chain": self.chain,
            "transaction_count": self.transaction_count,
            "total_volume": str(self.total_volume),
            "avg_transaction_size": str(self.avg_transaction_size),
            "avg_gas_cost": str(self.avg_gas_cost) if self.avg_gas_cost else None,
            "sov_ratio": self.sov_ratio,
            "activity_distribution": self.activity_distribution,
            "excluded_gas_count": self.excluded_gas_count,
        }


def analyze_by_chain(
    transactions_df: pd.DataFrame,
    holders_df: Optional[pd.DataFrame] = None,
) -> list:
    """
    Analyze transactions grouped by blockchain chain.

    Calculates transaction count, volume, average transaction size,
    average gas cost, and store-of-value ratio for each chain.

    Gas cost is computed as: gas_cost = gas_used × gas_price (in wei),
    converted to native token units (ETH/BNB/MATIC) by dividing by 10^18.
    Transactions with null gas_used or gas_price fields are excluded from
    gas cost calculations.

    Args:
        transactions_df: DataFrame with transaction data including 'chain',
                         'amount', 'activity_type', 'gas_used', 'gas_price'
        holders_df: Optional DataFrame with holder data for SoV ratio

    Returns:
        List of ChainMetrics objects, one per chain

    Requirements: 6.1, 6.3, 6.4
    """
    # Validate required columns
    required_cols = {'chain', 'amount', 'activity_type'}
    if not required_cols.issubset(transactions_df.columns):
        missing = required_cols - set(transactions_df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Initialize metrics for all supported chains
    chain_metrics: list = []

    for chain in SUPPORTED_CHAINS:
        chain_mask = transactions_df['chain'] == chain
        chain_df = transactions_df[chain_mask]

        # Initialize default values
        tx_count = 0
        total_volume = Decimal("0")
        avg_tx_size = Decimal("0")
        avg_gas_cost: Optional[Decimal] = None
        sov_ratio = 0.0
        activity_dist: Dict[str, float] = {at: 0.0 for at in ACTIVITY_TYPES}
        excluded_gas_count = 0

        if not chain_df.empty:
            # Transaction count
            tx_count = len(chain_df)

            # Total volume
            for amt in chain_df['amount'].dropna():
                if isinstance(amt, Decimal):
                    total_volume += amt
                else:
                    total_volume += Decimal(str(amt))

            # Average transaction size
            if tx_count > 0:
                avg_tx_size = total_volume / tx_count

            # Activity distribution (percentages within this chain)
            activity_counts = chain_df.groupby('activity_type').size()
            for at in ACTIVITY_TYPES:
                if at in activity_counts.index:
                    activity_dist[at] = activity_counts[at] / tx_count * 100.0

            # Calculate average gas cost
            # Gas cost = gas_used × gas_price (in wei)
            # Convert to native token by dividing by 10^18
            if 'gas_used' in chain_df.columns and 'gas_price' in chain_df.columns:
                # Filter transactions with valid gas data
                gas_mask = (
                    chain_df['gas_used'].notna() &
                    chain_df['gas_price'].notna()
                )
                valid_gas_df = chain_df[gas_mask]
                excluded_gas_count = len(chain_df) - len(valid_gas_df)

                if not valid_gas_df.empty:
                    total_gas_cost = Decimal("0")
                    gas_count = 0

                    for _, row in valid_gas_df.iterrows():
                        gas_used = row['gas_used']
                        gas_price = row['gas_price']

                        # Convert to Decimal if needed
                        if not isinstance(gas_used, (int, float, Decimal)):
                            try:
                                gas_used = Decimal(str(gas_used))
                            except (ValueError, TypeError):
                                continue
                        else:
                            gas_used = Decimal(str(gas_used))

                        if not isinstance(gas_price, (int, float, Decimal)):
                            try:
                                gas_price = Decimal(str(gas_price))
                            except (ValueError, TypeError):
                                continue
                        else:
                            gas_price = Decimal(str(gas_price))

                        # Calculate gas cost in wei, then convert to native token
                        gas_cost_wei = gas_used * gas_price
                        gas_cost_native = gas_cost_wei / Decimal("1000000000000000000")
                        total_gas_cost += gas_cost_native
                        gas_count += 1

                    if gas_count > 0:
                        avg_gas_cost = total_gas_cost / gas_count
            else:
                # No gas columns, all transactions excluded
                excluded_gas_count = tx_count

        # Calculate SoV ratio from holders if provided
        if holders_df is not None and not holders_df.empty:
            if 'chain' in holders_df.columns and \
               'is_store_of_value' in holders_df.columns:
                # Validate is_store_of_value is boolean or numeric
                if not (pd.api.types.is_bool_dtype(holders_df['is_store_of_value']) or 
                        pd.api.types.is_numeric_dtype(holders_df['is_store_of_value'])):
                    raise ValueError(
                        f"Column 'is_store_of_value' must be boolean or numeric, "
                        f"got {holders_df['is_store_of_value'].dtype}"
                    )
                chain_holders = holders_df[holders_df['chain'] == chain]
                if len(chain_holders) > 0:
                    sov_count = chain_holders['is_store_of_value'].sum()
                    sov_ratio = float(sov_count) / len(chain_holders)

        chain_metrics.append(ChainMetrics(
            chain=chain,
            transaction_count=tx_count,
            total_volume=total_volume,
            avg_transaction_size=avg_tx_size,
            avg_gas_cost=avg_gas_cost,
            sov_ratio=sov_ratio,
            activity_distribution=activity_dist,
            excluded_gas_count=excluded_gas_count,
        ))

    return chain_metrics


def get_chain_activity_distribution(
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Get activity type distribution per chain for visualization.

    Returns a DataFrame suitable for creating stacked bar charts.

    Args:
        transactions_df: DataFrame with 'chain' and 'activity_type' columns

    Returns:
        DataFrame with columns: chain, activity_type, count, percentage

    Requirements: 6.2
    """
    if 'chain' not in transactions_df.columns:
        raise ValueError("Column 'chain' not found in DataFrame")
    if 'activity_type' not in transactions_df.columns:
        raise ValueError("Column 'activity_type' not found in DataFrame")

    if transactions_df.empty:
        return pd.DataFrame(columns=['chain', 'activity_type', 'count', 'percentage'])

    # Group by chain and activity_type
    grouped = transactions_df.groupby(
        ['chain', 'activity_type'],
        as_index=False
    ).size()
    grouped.columns = ['chain', 'activity_type', 'count']

    # Calculate percentage within each chain
    chain_totals = grouped.groupby('chain')['count'].transform('sum')
    grouped['percentage'] = (grouped['count'] / chain_totals * 100.0).round(2)

    # Ensure all chain/activity combinations exist
    result_rows = []
    for chain in SUPPORTED_CHAINS:
        for at in ACTIVITY_TYPES:
            mask = (grouped['chain'] == chain) & (grouped['activity_type'] == at)
            if mask.any():
                row = grouped[mask].iloc[0]
                result_rows.append({
                    'chain': chain,
                    'activity_type': at,
                    'count': int(row['count']),
                    'percentage': float(row['percentage']),
                })
            else:
                result_rows.append({
                    'chain': chain,
                    'activity_type': at,
                    'count': 0,
                    'percentage': 0.0,
                })

    return pd.DataFrame(result_rows)



# =============================================================================
# Conclusion Generation Functions
# =============================================================================


from enum import Enum
from typing import List


class ConfidenceLevel(str, Enum):
    """Confidence level for analysis conclusions.
    
    Inherits from str to enable JSON serialization as string values.
    Use .value for string representation, e.g., ConfidenceLevel.HIGH.value == "high"
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    
    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        """Map confidence score to level based on thresholds.
        
        Args:
            score: Confidence score between 0.0 and 1.0
            
        Returns:
            ConfidenceLevel.HIGH if score >= 0.85
            ConfidenceLevel.MEDIUM if 0.50 <= score < 0.85
            ConfidenceLevel.LOW if score < 0.50
        """
        if score >= 0.85:
            return cls.HIGH
        elif score >= 0.50:
            return cls.MEDIUM
        else:
            return cls.LOW


# System constant: supported chains (fixed requirement)
SUPPORTED_CHAIN_COUNT: int = 3  # ethereum, bsc, polygon


@dataclass
class ConfidenceMetrics:
    """Metrics used for confidence calculation.
    
    Attributes:
        sample_size: Number of transactions in the dataset
        field_completeness: Percentage of non-null required fields (0.0-1.0)
        chain_coverage: chains_with_data / SUPPORTED_CHAIN_COUNT (0.0-1.0)
        chains_with_data: Count of unique chains present in dataset
        completeness_percent: Combined completeness (0.7*field + 0.3*chain)
        confidence_score: Final confidence score (0.0-1.0)
        confidence_level: Mapped confidence level (HIGH/MEDIUM/LOW)
    """
    sample_size: int
    field_completeness: float
    chain_coverage: float
    chains_with_data: int
    completeness_percent: float
    confidence_score: float
    confidence_level: ConfidenceLevel
    
    def to_dict(self) -> dict:
        """Serialize to dictionary with confidence_level as string value."""
        return {
            "sample_size": self.sample_size,
            "field_completeness": self.field_completeness,
            "chain_coverage": self.chain_coverage,
            "chains_with_data": self.chains_with_data,
            "completeness_percent": self.completeness_percent,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level.value,
        }


@dataclass
class Conclusion:
    """Analysis conclusion with confidence.
    
    Attributes:
        finding: Short description of the finding
        value: The value or result of the finding
        confidence: Confidence level for this conclusion
        explanation: Detailed explanation of the finding
    """
    finding: str
    value: str
    confidence: ConfidenceLevel
    explanation: str
    
    def to_dict(self) -> dict:
        """Serialize to dictionary with confidence as string value."""
        return {
            "finding": self.finding,
            "value": self.value,
            "confidence": self.confidence.value,
            "explanation": self.explanation,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Conclusion":
        """Deserialize from dictionary, mapping string to ConfidenceLevel."""
        return cls(
            finding=data["finding"],
            value=data["value"],
            confidence=ConfidenceLevel(data["confidence"]),
            explanation=data["explanation"],
        )


def calculate_field_completeness(transactions_df: pd.DataFrame) -> float:
    """
    Calculate the percentage of non-null required fields.
    
    Required fields: transaction_hash, timestamp, amount, stablecoin,
                     chain, activity_type
    
    Args:
        transactions_df: DataFrame with transaction data
    
    Returns:
        Percentage of non-null required fields (0.0 to 1.0)
    """
    if transactions_df.empty:
        return 0.0
    
    required_fields = [
        "transaction_hash", "timestamp", "amount",
        "stablecoin", "chain", "activity_type"
    ]
    
    # Count non-null values for each required field
    total_cells = 0
    non_null_cells = 0
    
    for field in required_fields:
        if field in transactions_df.columns:
            total_cells += len(transactions_df)
            non_null_cells += transactions_df[field].notna().sum()
        else:
            # Field is missing entirely
            total_cells += len(transactions_df)
    
    if total_cells == 0:
        return 0.0
    
    return non_null_cells / total_cells


def calculate_confidence(
    transactions_df: pd.DataFrame,
) -> ConfidenceMetrics:
    """
    Calculate confidence level based on data quality.
    
    Formula:
        field_completeness = non_null_required_fields / total_required_fields
        chain_coverage = chains_with_data / 3
        completeness_percent = 0.7 * field_completeness + 0.3 * chain_coverage
        normalized_sample_size = min(sample_size / 1000, 1.0)
        confidence_score = 0.6 * normalized_sample_size + 0.4 * completeness_percent
    
    Thresholds:
        HIGH: score >= 0.85
        MEDIUM: 0.50 <= score < 0.85
        LOW: score < 0.50
    
    Args:
        transactions_df: DataFrame with transaction data
    
    Returns:
        ConfidenceMetrics with all component values and final confidence_level
    
    Requirements: 7.3
    """
    sample_size = len(transactions_df)
    
    # Calculate field completeness
    field_completeness = calculate_field_completeness(transactions_df)
    
    # Calculate chain coverage
    if not transactions_df.empty and "chain" in transactions_df.columns:
        chains_with_data = transactions_df["chain"].dropna().nunique()
    else:
        chains_with_data = 0
    
    chain_coverage = chains_with_data / SUPPORTED_CHAIN_COUNT
    
    # Calculate combined completeness
    completeness_percent = 0.7 * field_completeness + 0.3 * chain_coverage
    
    # Calculate normalized sample size
    normalized_sample_size = min(sample_size / 1000, 1.0)
    
    # Calculate final confidence score
    confidence_score = 0.6 * normalized_sample_size + 0.4 * completeness_percent
    
    # Map to confidence level
    confidence_level = ConfidenceLevel.from_score(confidence_score)
    
    return ConfidenceMetrics(
        sample_size=sample_size,
        field_completeness=field_completeness,
        chain_coverage=chain_coverage,
        chains_with_data=chains_with_data,
        completeness_percent=completeness_percent,
        confidence_score=confidence_score,
        confidence_level=confidence_level,
    )


def generate_conclusions(
    activity_breakdown: ActivityBreakdown,
    holder_metrics: HolderMetrics,
    chain_metrics: List[ChainMetrics],
    confidence_metrics: ConfidenceMetrics,
    errors: List[str],
) -> List[Conclusion]:
    """
    Generate summary conclusions from analysis results.
    
    Calculates overall transaction vs SoV ratio and identifies key findings.
    
    Args:
        activity_breakdown: Activity type distribution metrics
        holder_metrics: Holder behavior metrics
        chain_metrics: Per-chain analysis metrics
        confidence_metrics: Data quality confidence metrics
        errors: List of data collection errors
    
    Returns:
        List of Conclusion objects with findings and confidence levels
    
    Requirements: 7.1, 7.2
    """
    conclusions: List[Conclusion] = []
    confidence = confidence_metrics.confidence_level
    
    # 1. Dominant usage pattern (transaction vs store_of_value)
    tx_count = activity_breakdown.counts.get("transaction", 0)
    sov_count = activity_breakdown.counts.get("store_of_value", 0)
    total_activity = tx_count + sov_count
    
    if total_activity > 0:
        tx_ratio = tx_count / total_activity
        sov_ratio = sov_count / total_activity
        
        if tx_ratio > sov_ratio:
            dominant = "Transaction"
            ratio_str = f"{tx_ratio:.1%} transactions vs {sov_ratio:.1%} store-of-value"
        else:
            dominant = "Store of Value"
            ratio_str = f"{sov_ratio:.1%} store-of-value vs {tx_ratio:.1%} transactions"
        
        conclusions.append(Conclusion(
            finding="Dominant Usage Pattern",
            value=dominant,
            confidence=confidence,
            explanation=f"Based on activity type distribution: {ratio_str}",
        ))
    
    # 2. Holder behavior pattern
    if holder_metrics.total_holders > 0:
        sov_pct = holder_metrics.sov_percentage
        active_pct = 100.0 - sov_pct
        
        if sov_pct > active_pct:
            holder_pattern = "Store of Value"
            pattern_str = f"{sov_pct:.1f}% holders are store-of-value"
        else:
            holder_pattern = "Active Transactors"
            pattern_str = f"{active_pct:.1f}% holders are active transactors"
        
        conclusions.append(Conclusion(
            finding="Holder Behavior Pattern",
            value=holder_pattern,
            confidence=confidence,
            explanation=pattern_str,
        ))
    
    # 3. Chain with highest transaction activity
    if chain_metrics:
        highest_chain = max(chain_metrics, key=lambda c: c.transaction_count)
        if highest_chain.transaction_count > 0:
            conclusions.append(Conclusion(
                finding="Most Active Chain",
                value=highest_chain.chain.capitalize(),
                confidence=confidence,
                explanation=(
                    f"{highest_chain.chain.capitalize()} has "
                    f"{highest_chain.transaction_count:,} transactions"
                ),
            ))
    
    # 4. Chain with highest SoV ratio
    if chain_metrics:
        chains_with_sov = [c for c in chain_metrics if c.sov_ratio > 0]
        if chains_with_sov:
            highest_sov_chain = max(chains_with_sov, key=lambda c: c.sov_ratio)
            conclusions.append(Conclusion(
                finding="Highest Store-of-Value Ratio",
                value=highest_sov_chain.chain.capitalize(),
                confidence=confidence,
                explanation=(
                    f"{highest_sov_chain.chain.capitalize()} has "
                    f"{highest_sov_chain.sov_ratio:.1%} store-of-value ratio"
                ),
            ))
    
    # 5. Data quality warning if errors present
    if errors:
        conclusions.append(Conclusion(
            finding="Data Quality Warning",
            value=f"{len(errors)} error(s) detected",
            confidence=ConfidenceLevel.LOW,
            explanation=(
                "Data collection encountered errors that may affect "
                "the accuracy of conclusions. Review the errors list "
                "for details."
            ),
        ))
    
    return conclusions


def get_data_quality_warnings(
    errors: List[str],
    confidence_metrics: ConfidenceMetrics,
) -> List[str]:
    """
    Generate data quality warnings based on errors and confidence.
    
    Args:
        errors: List of data collection errors
        confidence_metrics: Data quality confidence metrics
    
    Returns:
        List of warning messages
    
    Requirements: 7.4
    """
    warnings: List[str] = []
    
    # Add warnings for collection errors
    if errors:
        warnings.append(
            f"Data collection reported {len(errors)} error(s). "
            "Results may be incomplete."
        )
        # Add first few errors as specific warnings
        for error in errors[:3]:
            warnings.append(f"Collection error: {error}")
        if len(errors) > 3:
            warnings.append(f"... and {len(errors) - 3} more error(s)")
    
    # Add warnings for low confidence
    if confidence_metrics.confidence_level == ConfidenceLevel.LOW:
        warnings.append(
            f"Low confidence score ({confidence_metrics.confidence_score:.2f}). "
            "Consider collecting more data for reliable conclusions."
        )
    
    # Add warnings for incomplete chain coverage
    if confidence_metrics.chains_with_data < SUPPORTED_CHAIN_COUNT:
        missing_count = SUPPORTED_CHAIN_COUNT - confidence_metrics.chains_with_data
        warnings.append(
            f"Data from {missing_count} chain(s) is missing. "
            "Cross-chain analysis may be incomplete."
        )
    
    # Add warnings for low field completeness
    if confidence_metrics.field_completeness < 0.9:
        pct = confidence_metrics.field_completeness * 100
        warnings.append(
            f"Field completeness is {pct:.1f}%. "
            "Some records have missing required fields."
        )
    
    # Add warnings for small sample size
    if confidence_metrics.sample_size < 100:
        warnings.append(
            f"Small sample size ({confidence_metrics.sample_size} transactions). "
            "Statistical conclusions may not be representative."
        )
    
    return warnings
