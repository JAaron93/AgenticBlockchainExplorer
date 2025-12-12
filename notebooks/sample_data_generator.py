"""
Sample data generation for stablecoin analysis notebook.

This module provides functions to generate synthetic sample data for testing
and demonstration purposes when no real dataset is available.

Requirements: 8.1, 8.2, 8.4
"""

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List, Optional

import pandas as pd

from stablecoin_loader import LoadedData


# Valid enum values (matching validation module)
SUPPORTED_STABLECOINS = ["USDC", "USDT"]
SUPPORTED_CHAINS = ["ethereum", "bsc", "polygon"]
ACTIVITY_TYPES = ["transaction", "store_of_value", "other"]


@dataclass
class SampleDataConfig:
    """Configuration for sample data generation.
    
    Attributes:
        num_transactions: Number of transactions to generate
        num_holders: Number of holders to generate
        stablecoins: List of stablecoins to include
        chains: List of chains to include
        sov_ratio: Ratio of store-of-value holders (0.0 to 1.0)
        date_range_days: Number of days to span for timestamps
        min_amount: Minimum transaction amount
        max_amount: Maximum transaction amount
        min_balance: Minimum holder balance
        max_balance: Maximum holder balance
        seed: Random seed for reproducibility (None for random)
    """
    num_transactions: int = 1000
    num_holders: int = 100
    stablecoins: List[str] = field(
        default_factory=lambda: ["USDC", "USDT"]
    )
    chains: List[str] = field(
        default_factory=lambda: ["ethereum", "bsc", "polygon"]
    )
    sov_ratio: float = 0.3  # 30% store of value
    date_range_days: int = 90
    min_amount: Decimal = Decimal("10")
    max_amount: Decimal = Decimal("100000")
    min_balance: Decimal = Decimal("100")
    max_balance: Decimal = Decimal("1000000")
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.num_transactions < 0:
            raise ValueError("num_transactions must be non-negative")
        if self.num_holders < 0:
            raise ValueError("num_holders must be non-negative")
        if not 0.0 <= self.sov_ratio <= 1.0:
            raise ValueError("sov_ratio must be between 0.0 and 1.0")
        if self.date_range_days < 1:
            raise ValueError("date_range_days must be at least 1")
        if self.min_amount < 0:
            raise ValueError("min_amount must be non-negative")
        if self.max_amount < self.min_amount:
            raise ValueError("max_amount must be >= min_amount")
        if self.min_balance < 0:
            raise ValueError("min_balance must be non-negative")
        if self.max_balance < self.min_balance:
            raise ValueError("max_balance must be >= min_balance")
        
        # Validate stablecoins
        for coin in self.stablecoins:
            if coin not in SUPPORTED_STABLECOINS:
                raise ValueError(
                    f"Invalid stablecoin '{coin}'. "
                    f"Must be one of {SUPPORTED_STABLECOINS}"
                )
        
        # Validate chains
        for chain in self.chains:
            if chain not in SUPPORTED_CHAINS:
                raise ValueError(
                    f"Invalid chain '{chain}'. "
                    f"Must be one of {SUPPORTED_CHAINS}"
                )


def _generate_address() -> str:
    """Generate a random Ethereum-style address."""
    return "0x" + "".join(
        random.choice("0123456789abcdef") for _ in range(40)
    )


def _generate_tx_hash() -> str:
    """Generate a random transaction hash."""
    return "0x" + "".join(
        random.choice("0123456789abcdef") for _ in range(64)
    )


def _generate_amount(
    min_amount: Decimal,
    max_amount: Decimal,
) -> Decimal:
    """Generate a random amount within the specified range."""
    # Generate a random float and convert to Decimal with 6 decimal places
    range_float = float(max_amount - min_amount)
    random_float = random.random() * range_float
    return (min_amount + Decimal(str(random_float))).quantize(
        Decimal("0.000001")
    )


def _generate_timestamp(
    start_date: datetime,
    end_date: datetime,
) -> datetime:
    """Generate a random timestamp within the specified range."""
    delta = end_date - start_date
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start_date + timedelta(seconds=random_seconds)


def _format_timestamp(dt: datetime) -> str:
    """Format datetime as ISO8601 string."""
    return dt.isoformat().replace('+00:00', 'Z')


def _get_explorer_for_chain(chain: str) -> str:
    """Get the explorer name for a chain."""
    explorer_map = {
        "ethereum": "etherscan",
        "bsc": "bscscan",
        "polygon": "polygonscan",
    }
    return explorer_map.get(chain, "etherscan")


def generate_sample_transactions(
    config: SampleDataConfig,
    addresses: List[str],
    start_date: datetime,
    end_date: datetime,
) -> List[dict]:
    """
    Generate sample transaction records.
    
    Args:
        config: Sample data configuration
        addresses: List of addresses to use for from/to
        start_date: Start of date range
        end_date: End of date range
    
    Returns:
        List of transaction dictionaries
    """
    transactions = []
    
    # Activity type distribution:
    # - 60% transaction
    # - 30% store_of_value
    # - 10% other (minting/burning)
    activity_weights = [0.6, 0.3, 0.1]
    
    for _ in range(config.num_transactions):
        chain = random.choice(config.chains)
        stablecoin = random.choice(config.stablecoins)
        activity_type = random.choices(
            ACTIVITY_TYPES,
            weights=activity_weights,
            k=1
        )[0]
        
        timestamp = _generate_timestamp(start_date, end_date)
        
        # Generate gas data (some transactions may have null gas)
        has_gas_data = random.random() > 0.1  # 90% have gas data
        gas_used = random.randint(21000, 200000) if has_gas_data else None
        gas_price = (
            str(Decimal(str(random.randint(1, 100))) * Decimal("1000000000"))
            if has_gas_data else None
        )  # 1-100 Gwei in wei
        
        tx = {
            "transaction_hash": _generate_tx_hash(),
            "block_number": random.randint(10000000, 20000000),
            "timestamp": _format_timestamp(timestamp),
            "from_address": random.choice(addresses),
            "to_address": random.choice(addresses),
            "amount": str(_generate_amount(
                config.min_amount, config.max_amount
            )),
            "stablecoin": stablecoin,
            "chain": chain,
            "activity_type": activity_type,
            "source_explorer": _get_explorer_for_chain(chain),
            "gas_used": gas_used,
            "gas_price": gas_price,
        }
        transactions.append(tx)
    
    return transactions


def generate_sample_holders(
    config: SampleDataConfig,
    addresses: List[str],
    start_date: datetime,
    end_date: datetime,
) -> List[dict]:
    """
    Generate sample holder records.
    
    Args:
        config: Sample data configuration
        addresses: List of addresses to use
        start_date: Start of date range
        end_date: End of date range
    
    Returns:
        List of holder dictionaries
    """
    holders = []
    
    # Determine how many holders should be SoV
    num_sov = int(config.num_holders * config.sov_ratio)
    
    for i in range(config.num_holders):
        # Use provided addresses or generate new ones
        if i < len(addresses):
            address = addresses[i]
        else:
            address = _generate_address()
        
        chain = random.choice(config.chains)
        stablecoin = random.choice(config.stablecoins)
        
        # Determine if this holder is SoV
        is_sov = i < num_sov
        
        # Generate timestamps
        first_seen = _generate_timestamp(start_date, end_date)
        # last_activity must be >= first_seen
        days_after = random.randint(0, 30) if is_sov else random.randint(0, 7)
        last_activity = min(
            first_seen + timedelta(days=days_after),
            end_date
        )
        
        holder = {
            "address": address,
            "balance": str(_generate_amount(
                config.min_balance, config.max_balance
            )),
            "stablecoin": stablecoin,
            "chain": chain,
            "first_seen": _format_timestamp(first_seen),
            "last_activity": _format_timestamp(last_activity),
            "is_store_of_value": is_sov,
            "source_explorer": _get_explorer_for_chain(chain),
        }
        holders.append(holder)
    
    return holders


def compute_summary(
    transactions: List[dict],
    holders: List[dict],
) -> dict:
    """
    Compute summary statistics from generated data.
    
    Args:
        transactions: List of transaction records
        holders: List of holder records
    
    Returns:
        Summary dictionary
    """
    # Initialize counters
    by_stablecoin = {
        coin: {"transaction_count": 0, "total_volume": Decimal("0")}
        for coin in SUPPORTED_STABLECOINS
    }
    by_activity_type = {at: 0 for at in ACTIVITY_TYPES}
    by_chain = {chain: 0 for chain in SUPPORTED_CHAINS}
    
    # Aggregate from transactions
    for tx in transactions:
        coin = tx["stablecoin"]
        activity = tx["activity_type"]
        chain = tx["chain"]
        amount = Decimal(tx["amount"])
        
        if coin in by_stablecoin:
            by_stablecoin[coin]["transaction_count"] += 1
            by_stablecoin[coin]["total_volume"] += amount
        if activity in by_activity_type:
            by_activity_type[activity] += 1
        if chain in by_chain:
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


def _generate_raw_sample_data(config: SampleDataConfig) -> dict:
    """
    Generate raw sample data as dictionaries.
    
    This is the core generation logic shared by generate_sample_data
    and generate_sample_json.
    
    Args:
        config: Configuration specifying sample size and distribution
    
    Returns:
        Dictionary with keys: metadata, summary, transactions, holders
    """
    # Set random seed if provided for reproducibility
    if config.seed is not None:
        random.seed(config.seed)
    
    # Calculate date range
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=config.date_range_days)
    
    # Generate a pool of addresses
    # Ensure we have at least 1 address if we have any transactions
    min_addresses = 1 if config.num_transactions > 0 else 0
    num_addresses = max(
        config.num_holders,
        min(config.num_transactions // 2, 500),
        min_addresses
    )
    addresses = [_generate_address() for _ in range(num_addresses)]
    
    # Generate transactions
    transactions = generate_sample_transactions(
        config, addresses, start_date, end_date
    )
    
    # Generate holders
    holders = generate_sample_holders(
        config, addresses, start_date, end_date
    )
    
    # Compute summary
    summary = compute_summary(transactions, holders)
    
    # Create metadata
    metadata = {
        "run_id": str(uuid.uuid4()),
        "collection_timestamp": _format_timestamp(datetime.now(timezone.utc)),
        "agent_version": "sample-generator-1.0.0",
        "explorers_queried": list(set(
            _get_explorer_for_chain(c) for c in config.chains
        )),
        "total_records": len(transactions) + len(holders),
    }
    
    return {
        "metadata": metadata,
        "summary": summary,
        "transactions": transactions,
        "holders": holders,
    }


def generate_sample_data(config: SampleDataConfig) -> LoadedData:
    """
    Generate synthetic sample data for testing and demonstration.
    
    Creates realistic transaction and holder records following the same
    schema as real exports from the blockchain explorer data collection.
    
    Args:
        config: Configuration specifying sample size and distribution
    
    Returns:
        LoadedData container with generated data, marked as sample data
    
    Requirements: 8.1, 8.2
    """
    # Generate raw data using shared helper
    raw_data = _generate_raw_sample_data(config)
    
    # Convert to DataFrames
    transactions_df = _convert_transactions_to_df(raw_data["transactions"])
    holders_df = _convert_holders_to_df(raw_data["holders"])
    
    return LoadedData(
        metadata=raw_data["metadata"],
        transactions_df=transactions_df,
        holders_df=holders_df,
        summary=raw_data["summary"],
        errors=[],
        is_sample_data=True,
    )


def _convert_transactions_to_df(transactions: List[dict]) -> pd.DataFrame:
    """Convert transactions list to pandas DataFrame with proper types."""
    if not transactions:
        return pd.DataFrame(columns=[
            "transaction_hash", "block_number", "timestamp", "from_address",
            "to_address", "amount", "stablecoin", "chain", "activity_type",
            "source_explorer", "gas_used", "gas_price"
        ])
    
    df = pd.DataFrame(transactions)
    
    # Convert amount to Decimal
    df["amount"] = df["amount"].apply(
        lambda x: Decimal(str(x)) if x is not None else None
    )
    
    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601', utc=True)
    
    # Convert gas_price to Decimal if present
    if "gas_price" in df.columns:
        df["gas_price"] = df["gas_price"].apply(
            lambda x: Decimal(str(x)) if x is not None else None
        )
    
    return df


def _convert_holders_to_df(holders: List[dict]) -> pd.DataFrame:
    """Convert holders list to pandas DataFrame with proper types."""
    if not holders:
        return pd.DataFrame(columns=[
            "address", "balance", "stablecoin", "chain", "first_seen",
            "last_activity", "is_store_of_value", "source_explorer",
            "holding_period_days"
        ])
    
    df = pd.DataFrame(holders)
    
    # Convert balance to Decimal
    df["balance"] = df["balance"].apply(
        lambda x: Decimal(str(x)) if x is not None else None
    )
    
    # Parse timestamps
    df["first_seen"] = pd.to_datetime(df["first_seen"], format='ISO8601', utc=True)
    df["last_activity"] = pd.to_datetime(
        df["last_activity"], format='ISO8601', utc=True
    )
    
    # Calculate holding period in days
    df["holding_period_days"] = (
        df["last_activity"] - df["first_seen"]
    ).dt.days.clip(lower=0)
    
    return df


def generate_sample_json(config: SampleDataConfig) -> dict:
    """
    Generate sample data as a JSON-compatible dictionary.
    
    This is useful for testing schema validation.
    
    Args:
        config: Configuration specifying sample size and distribution
    
    Returns:
        Dictionary matching the JSON export schema
    
    Requirements: 8.2
    """
    # Use shared helper to generate raw data
    return _generate_raw_sample_data(config)
