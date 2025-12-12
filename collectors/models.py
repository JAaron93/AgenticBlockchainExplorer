"""Data models for blockchain explorer collectors."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional


class ActivityType(str, Enum):
    """Type of stablecoin activity."""
    TRANSACTION = "transaction"
    STORE_OF_VALUE = "store_of_value"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass
class Transaction:
    """Represents a stablecoin transaction."""
    
    transaction_hash: str
    block_number: int
    timestamp: datetime
    from_address: str
    to_address: str
    amount: Decimal
    stablecoin: str  # "USDC" or "USDT"
    chain: str  # "ethereum", "bsc", "polygon"
    activity_type: ActivityType
    source_explorer: str
    gas_used: Optional[int] = None
    gas_price: Optional[Decimal] = None
    
    def __post_init__(self) -> None:
        """Validate transaction data after initialization."""
        if not self.transaction_hash:
            raise ValueError("Transaction hash cannot be empty")
        if self.block_number < 0:
            raise ValueError("Block number cannot be negative")
        if self.amount < 0:
            raise ValueError("Amount cannot be negative")
        if not self.from_address:
            raise ValueError("From address cannot be empty")
        if not self.to_address:
            raise ValueError("To address cannot be empty")
        if self.stablecoin not in ("USDC", "USDT"):
            raise ValueError(f"Invalid stablecoin: {self.stablecoin}")
        if self.chain not in ("ethereum", "bsc", "polygon"):
            raise ValueError(f"Invalid chain: {self.chain}")
        if self.gas_used is not None and self.gas_used < 0:
            raise ValueError("Gas used cannot be negative")
        if self.gas_price is not None and self.gas_price < 0:
            raise ValueError("Gas price cannot be negative")


    def to_dict(self) -> dict:
        """Convert transaction to dictionary for JSON serialization."""
        return {
            "transaction_hash": self.transaction_hash,
            "block_number": self.block_number,
            "timestamp": self.timestamp.isoformat(),
            "from_address": self.from_address,
            "to_address": self.to_address,
            "amount": str(self.amount),
            "stablecoin": self.stablecoin,
            "chain": self.chain,
            "activity_type": self.activity_type.value,
            "source_explorer": self.source_explorer,
            "gas_used": self.gas_used,
            "gas_price": str(self.gas_price) if self.gas_price else None,
        }


@dataclass
class Holder:
    """Represents a stablecoin token holder."""

    address: str
    balance: Decimal
    stablecoin: str
    chain: str
    first_seen: datetime
    last_activity: datetime
    is_store_of_value: bool
    source_explorer: str

    def __post_init__(self) -> None:
        """Validate holder data after initialization."""
        if not self.address:
            raise ValueError("Address cannot be empty")
        if self.balance < 0:
            raise ValueError("Balance cannot be negative")
        if self.stablecoin not in ("USDC", "USDT"):
            raise ValueError(f"Invalid stablecoin: {self.stablecoin}")
        if self.chain not in ("ethereum", "bsc", "polygon"):
            raise ValueError(f"Invalid chain: {self.chain}")
        if self.first_seen > self.last_activity:
            raise ValueError("First seen cannot be after last activity")
    
    def to_dict(self) -> dict:
        """Convert holder to dictionary for JSON serialization."""
        return {
            "address": self.address,
            "balance": str(self.balance),
            "stablecoin": self.stablecoin,
            "chain": self.chain,
            "first_seen": self.first_seen.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "is_store_of_value": self.is_store_of_value,
            "source_explorer": self.source_explorer,
        }


@dataclass
class ExplorerData:
    """Data collected from a single explorer."""
    
    explorer_name: str
    chain: str
    transactions: list[Transaction] = field(default_factory=list)
    holders: list[Holder] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    collection_time_seconds: float = 0.0
    
    @property
    def total_records(self) -> int:
        """Get total number of records collected."""
        return len(self.transactions) + len(self.holders)
    
    @property
    def success(self) -> bool:
        """Check if collection was successful.
        
        Returns True if data was collected, regardless of whether some
        non-critical errors occurred (partial success is still success).
        Returns False only if no data was collected.
        """
        return self.total_records > 0
