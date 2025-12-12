"""
Data loading functions for stablecoin analysis notebook.

This module provides functions to load JSON export files and convert
them to pandas DataFrames for analysis.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from stablecoin_validation import validate_schema

logger = logging.getLogger(__name__)


@dataclass
class LoadedData:
    """Container for loaded and validated stablecoin data."""
    metadata: dict
    transactions_df: pd.DataFrame
    holders_df: pd.DataFrame
    summary: dict
    errors: List[str]
    is_sample_data: bool = False


def _safe_decimal(value) -> Optional[Decimal]:
    """
    Safely convert a value to Decimal, returning None on failure.
    
    Handles None, empty strings, and malformed inputs gracefully.
    """
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as e:
        logger.warning(f"Failed to convert '{value}' to Decimal: {e}")
        return None


def _parse_timestamp(ts_str: str) -> Optional[datetime]:
    """
    Parse ISO8601 timestamp string to datetime.
    
    Returns None if parsing fails.
    """
    if ts_str is None or ts_str == "":
        return None
    try:
        return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    except (ValueError, AttributeError) as e:
        logger.warning(f"Failed to parse timestamp '{ts_str}': {e}")
        return None


def _convert_transactions_to_df(transactions: list) -> pd.DataFrame:
    """Convert transactions list to pandas DataFrame with proper types."""
    if not transactions:
        return pd.DataFrame(columns=[
            "transaction_hash", "block_number", "timestamp", "from_address",
            "to_address", "amount", "stablecoin", "chain", "activity_type",
            "source_explorer", "gas_used", "gas_price"
        ])

    df = pd.DataFrame(transactions)

    # Convert amount to Decimal with defensive handling
    df["amount"] = df["amount"].apply(_safe_decimal)
    
    # Log warning if any amounts failed to convert
    null_amounts = df["amount"].isna().sum()
    if null_amounts > 0:
        logger.warning(
            f"Found {null_amounts} transaction(s) with invalid amount values"
        )

    # Parse timestamps with defensive handling
    df["timestamp"] = df["timestamp"].apply(_parse_timestamp)

    # Convert gas_price to Decimal if present
    if "gas_price" in df.columns:
        df["gas_price"] = df["gas_price"].apply(_safe_decimal)

    return df


def _convert_holders_to_df(holders: list) -> pd.DataFrame:
    """Convert holders list to pandas DataFrame with proper types."""
    if not holders:
        return pd.DataFrame(columns=[
            "address", "balance", "stablecoin", "chain", "first_seen",
            "last_activity", "is_store_of_value", "source_explorer",
            "holding_period_days"
        ])

    df = pd.DataFrame(holders)

    # Convert balance to Decimal with defensive handling
    df["balance"] = df["balance"].apply(_safe_decimal)
    
    # Log warning if any balances failed to convert
    null_balances = df["balance"].isna().sum()
    if null_balances > 0:
        logger.warning(
            f"Found {null_balances} holder(s) with invalid balance values"
        )

    # Parse timestamps with defensive handling
    df["first_seen"] = df["first_seen"].apply(_parse_timestamp)
    df["last_activity"] = df["last_activity"].apply(_parse_timestamp)

    # Calculate holding period in days, clamping negatives to zero
    # Handle cases where timestamps might be None
    valid_timestamps = (
        df["first_seen"].notna() & df["last_activity"].notna()
    )
    
    df["holding_period_days"] = 0  # Default value
    
    if valid_timestamps.any():
        delta_days = (
            df.loc[valid_timestamps, "last_activity"] -
            df.loc[valid_timestamps, "first_seen"]
        ).dt.days
        
        # Flag any invalid rows where last_activity < first_seen
        invalid_mask = delta_days < 0
        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            logger.warning(
                f"Found {invalid_count} holder(s) with "
                "last_activity < first_seen. "
                "Clamping holding_period_days to 0 for these rows."
            )
        
        # Clamp negative values to zero
        df.loc[valid_timestamps, "holding_period_days"] = delta_days.clip(
            lower=0
        )

    return df


def load_json_file(file_path: Union[str, Path]) -> LoadedData:
    """
    Load and validate JSON export file.

    Args:
        file_path: Path to the JSON export file

    Returns:
        LoadedData container with parsed data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
        ValueError: If schema validation fails
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path, 'r') as f:
        data = json.load(f)

    # Validate schema
    is_valid, validation_errors = validate_schema(data)
    if not is_valid:
        raise ValueError(
            "Schema validation failed:\n" +
            "\n".join(f"  - {e}" for e in validation_errors)
        )

    # Convert to DataFrames
    transactions_df = _convert_transactions_to_df(
        data.get("transactions", [])
    )
    holders_df = _convert_holders_to_df(data.get("holders", []))

    # Extract errors from data if present
    data_errors = data.get("errors", [])

    return LoadedData(
        metadata=data["metadata"],
        transactions_df=transactions_df,
        holders_df=holders_df,
        summary=data["summary"],
        errors=data_errors,
        is_sample_data=False,
    )


def load_json_data(data: dict) -> LoadedData:
    """
    Load and validate JSON data from a dictionary (for testing).

    Args:
        data: Parsed JSON data dictionary

    Returns:
        LoadedData container with parsed data

    Raises:
        ValueError: If schema validation fails
    """
    # Validate schema
    is_valid, validation_errors = validate_schema(data)
    if not is_valid:
        raise ValueError(
            "Schema validation failed:\n" +
            "\n".join(f"  - {e}" for e in validation_errors)
        )

    # Convert to DataFrames
    transactions_df = _convert_transactions_to_df(
        data.get("transactions", [])
    )
    holders_df = _convert_holders_to_df(data.get("holders", []))

    # Extract errors from data if present
    data_errors = data.get("errors", [])

    return LoadedData(
        metadata=data["metadata"],
        transactions_df=transactions_df,
        holders_df=holders_df,
        summary=data["summary"],
        errors=data_errors,
        is_sample_data=False,
    )
