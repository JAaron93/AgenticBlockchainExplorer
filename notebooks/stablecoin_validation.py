"""
Schema validation functions for stablecoin analysis notebook.

This module provides validation functions for JSON export data from the
blockchain explorer data collection agents.
"""

from datetime import datetime
from decimal import Decimal
from typing import Tuple, List, Any

# Valid enum values
SUPPORTED_STABLECOINS = ["USDC", "USDT"]
SUPPORTED_CHAINS = ["ethereum", "bsc", "polygon"]
ACTIVITY_TYPES = ["transaction", "store_of_value", "other"]


def _validate_iso8601(value: str, field_name: str) -> List[str]:
    """Validate ISO8601 timestamp format."""
    errors = []
    try:
        datetime.fromisoformat(value.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        errors.append(f"Invalid ISO8601 timestamp in '{field_name}': {value}")
    return errors


def _validate_decimal_string(value: str, field_name: str) -> List[str]:
    """Validate decimal string format."""
    errors = []
    try:
        dec_val = Decimal(value)
        if dec_val < 0:
            errors.append(f"Negative value in '{field_name}': {value}")
    except Exception:
        errors.append(f"Invalid decimal string in '{field_name}': {value}")
    return errors


def _validate_metadata(metadata: Any) -> List[str]:
    """Validate metadata section."""
    errors = []
    if not isinstance(metadata, dict):
        return ["metadata must be an object"]

    required_fields = [
        "run_id", "collection_timestamp", "agent_version",
        "explorers_queried", "total_records"
    ]

    for field in required_fields:
        if field not in metadata:
            errors.append(f"Missing required metadata field: {field}")

    if "run_id" in metadata and not metadata["run_id"]:
        errors.append("metadata.run_id must be non-empty")

    if "collection_timestamp" in metadata:
        errors.extend(_validate_iso8601(
            metadata["collection_timestamp"], "metadata.collection_timestamp"
        ))

    if "explorers_queried" in metadata:
        if not isinstance(metadata["explorers_queried"], list):
            errors.append("metadata.explorers_queried must be an array")
        elif len(metadata["explorers_queried"]) == 0:
            errors.append("metadata.explorers_queried must be non-empty")

    if "total_records" in metadata:
        val = metadata["total_records"]
        if not isinstance(val, int) or val < 0:
            errors.append(
                "metadata.total_records must be a non-negative integer"
            )

    return errors


def _validate_summary(summary: Any) -> List[str]:
    """Validate summary section."""
    errors = []
    if not isinstance(summary, dict):
        return ["summary must be an object"]

    required_fields = ["by_stablecoin", "by_activity_type", "by_chain"]
    for field in required_fields:
        if field not in summary:
            errors.append(f"Missing required summary field: {field}")

    return errors


def _validate_transaction(tx: Any, index: int) -> List[str]:
    """Validate a single transaction record."""
    errors = []
    if not isinstance(tx, dict):
        return [f"transactions[{index}] must be an object"]

    required_fields = [
        "transaction_hash", "block_number", "timestamp",
        "from_address", "to_address", "amount", "stablecoin",
        "chain", "activity_type", "source_explorer"
    ]

    for field in required_fields:
        if field not in tx:
            errors.append(
                f"transactions[{index}]: Missing required field '{field}'"
            )

    if "transaction_hash" in tx and not tx["transaction_hash"]:
        errors.append(
            f"transactions[{index}]: transaction_hash must be non-empty"
        )

    if "block_number" in tx:
        val = tx["block_number"]
        if not isinstance(val, int) or val < 0:
            errors.append(
                f"transactions[{index}]: block_number must be "
                "non-negative integer"
            )

    if "timestamp" in tx:
        ts_errors = _validate_iso8601(tx["timestamp"], "timestamp")
        for e in ts_errors:
            errors.append(
                e.replace("'timestamp'", f"transactions[{index}].timestamp")
            )

    if "amount" in tx:
        amt_errors = _validate_decimal_string(tx["amount"], "amount")
        for e in amt_errors:
            errors.append(
                e.replace("'amount'", f"transactions[{index}].amount")
            )

    if "stablecoin" in tx and tx["stablecoin"] not in SUPPORTED_STABLECOINS:
        errors.append(
            f"transactions[{index}]: Invalid stablecoin '{tx['stablecoin']}', "
            f"must be one of {SUPPORTED_STABLECOINS}"
        )

    if "chain" in tx and tx["chain"] not in SUPPORTED_CHAINS:
        errors.append(
            f"transactions[{index}]: Invalid chain '{tx['chain']}', "
            f"must be one of {SUPPORTED_CHAINS}"
        )

    if "activity_type" in tx and tx["activity_type"] not in ACTIVITY_TYPES:
        errors.append(
            f"transactions[{index}]: Invalid activity_type "
            f"'{tx['activity_type']}', must be one of {ACTIVITY_TYPES}"
        )

    # Optional fields validation
    if "gas_used" in tx and tx["gas_used"] is not None:
        val = tx["gas_used"]
        if not isinstance(val, int) or val < 0:
            errors.append(
                f"transactions[{index}]: gas_used must be non-negative integer"
            )

    if "gas_price" in tx and tx["gas_price"] is not None:
        gp_errors = _validate_decimal_string(tx["gas_price"], "gas_price")
        for e in gp_errors:
            errors.append(
                e.replace("'gas_price'", f"transactions[{index}].gas_price")
            )

    return errors


def _validate_holder(holder: Any, index: int) -> List[str]:
    """Validate a single holder record."""
    errors = []
    if not isinstance(holder, dict):
        return [f"holders[{index}] must be an object"]

    required_fields = [
        "address", "balance", "stablecoin", "chain",
        "first_seen", "last_activity", "is_store_of_value",
        "source_explorer"
    ]

    for field in required_fields:
        if field not in holder:
            errors.append(
                f"holders[{index}]: Missing required field '{field}'"
            )

    if "address" in holder and not holder["address"]:
        errors.append(f"holders[{index}]: address must be non-empty")

    if "balance" in holder:
        bal_errors = _validate_decimal_string(holder["balance"], "balance")
        for e in bal_errors:
            errors.append(
                e.replace("'balance'", f"holders[{index}].balance")
            )

    if "stablecoin" in holder:
        if holder["stablecoin"] not in SUPPORTED_STABLECOINS:
            errors.append(
                f"holders[{index}]: Invalid stablecoin "
                f"'{holder['stablecoin']}', "
                f"must be one of {SUPPORTED_STABLECOINS}"
            )

    if "chain" in holder and holder["chain"] not in SUPPORTED_CHAINS:
        errors.append(
            f"holders[{index}]: Invalid chain '{holder['chain']}', "
            f"must be one of {SUPPORTED_CHAINS}"
        )

    if "first_seen" in holder:
        fs_errors = _validate_iso8601(holder["first_seen"], "first_seen")
        for e in fs_errors:
            errors.append(
                e.replace("'first_seen'", f"holders[{index}].first_seen")
            )

    if "last_activity" in holder:
        la_errors = _validate_iso8601(
            holder["last_activity"], "last_activity"
        )
        for e in la_errors:
            errors.append(
                e.replace("'last_activity'", f"holders[{index}].last_activity")
            )

    # Temporal constraint: last_activity >= first_seen
    if "first_seen" in holder and "last_activity" in holder:
        try:
            first = datetime.fromisoformat(
                holder["first_seen"].replace('Z', '+00:00')
            )
            last = datetime.fromisoformat(
                holder["last_activity"].replace('Z', '+00:00')
            )
            if last < first:
                errors.append(
                    f"holders[{index}]: last_activity must be >= first_seen"
                )
        except (ValueError, AttributeError):
            pass  # Already caught by timestamp validation

    if "is_store_of_value" in holder:
        if not isinstance(holder["is_store_of_value"], bool):
            errors.append(
                f"holders[{index}]: is_store_of_value must be a boolean"
            )

    return errors


def validate_schema(data: dict) -> Tuple[bool, List[str]]:
    """
    Validate JSON structure against the canonical schema.

    Args:
        data: Parsed JSON data dictionary

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    if not isinstance(data, dict):
        return False, ["Data must be a JSON object"]

    # Check required top-level fields
    required_top_level = ["metadata", "summary", "transactions", "holders"]
    for field in required_top_level:
        if field not in data:
            errors.append(f"Missing required top-level field: {field}")

    # Validate metadata
    if "metadata" in data:
        errors.extend(_validate_metadata(data["metadata"]))

    # Validate summary
    if "summary" in data:
        errors.extend(_validate_summary(data["summary"]))

    # Validate transactions array
    if "transactions" in data:
        if not isinstance(data["transactions"], list):
            errors.append("transactions must be an array")
        else:
            for i, tx in enumerate(data["transactions"]):
                errors.extend(_validate_transaction(tx, i))

    # Validate holders array
    if "holders" in data:
        if not isinstance(data["holders"], list):
            errors.append("holders must be an array")
        else:
            for i, holder in enumerate(data["holders"]):
                errors.extend(_validate_holder(holder, i))

    return len(errors) == 0, errors
