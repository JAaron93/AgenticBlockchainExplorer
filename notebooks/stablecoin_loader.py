"""
Data loading functions for stablecoin analysis notebook.

This module provides functions to load JSON export files and convert
them to pandas DataFrames for analysis.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from stablecoin_validation import validate_schema


@dataclass
class LoadedData:
    """Container for loaded and validated stablecoin data."""
    metadata: dict
    transactions_df: pd.DataFrame
    holders_df: pd.DataFrame
    summary: dict
    errors: List[str]
    is_sample_data: bool = False


def _parse_timestamp(ts_str: str) -> datetime:
    """Parse ISO8601 timestamp string to datetime."""
    return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))


def _convert_transactions_to_df(transactions: list) -> pd.DataFrame:
    """Convert transactions list to pandas DataFrame with proper types."""
    if not transactions:
        return pd.DataFrame(columns=[
            "transaction_hash", "block_number", "timestamp", "from_address",
            "to_address", "amount", "stablecoin", "chain", "activity_type",
            "source_explorer", "gas_used", "gas_price"
        ])

    df = pd.DataFrame(transactions)

    # Convert amount to Decimal
    df["amount"] = df["amount"].apply(Decimal)

    # Parse timestamps
    df["timestamp"] = df["timestamp"].apply(_parse_timestamp)

    # Convert gas_price to Decimal if present
    if "gas_price" in df.columns:
        df["gas_price"] = df["gas_price"].apply(
            lambda x: Decimal(x) if x is not None else None
        )

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

    # Convert balance to Decimal
    df["balance"] = df["balance"].apply(Decimal)

    # Parse timestamps
    df["first_seen"] = df["first_seen"].apply(_parse_timestamp)
    df["last_activity"] = df["last_activity"].apply(_parse_timestamp)

    # Calculate holding period in days
    df["holding_period_days"] = (
        df["last_activity"] - df["first_seen"]
    ).dt.days

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
            f"Schema validation failed:\n" +
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
            f"Schema validation failed:\n" +
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
