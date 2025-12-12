"""
Analysis functions for stablecoin analysis notebook.

This module provides functions to analyze stablecoin transaction data,
including activity type breakdown, holder metrics, and chain comparisons.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict

import pandas as pd


# Activity types
ACTIVITY_TYPES = ["transaction", "store_of_value", "other"]


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

    # Calculate total count for percentages
    total_count = sum(counts.values())

    # Calculate percentages
    for activity_type in ACTIVITY_TYPES:
        if total_count > 0:
            percentages[activity_type] = (
                counts[activity_type] / total_count * 100.0
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
