"""ZenML steps for blockchain data collection.

This module wraps the existing blockchain explorer collectors as ZenML steps
with typed inputs and outputs for pipeline orchestration.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import asyncio
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

import pandas as pd
from zenml import step

from collectors.models import Transaction, Holder, ExplorerData
from collectors.aggregator import DataAggregator, AggregatedData


logger = logging.getLogger(__name__)


# Required columns for transaction DataFrame output
TRANSACTION_REQUIRED_COLUMNS = [
    "transaction_hash",
    "block_number",
    "timestamp",
    "from_address",
    "to_address",
    "amount",
    "stablecoin",
    "chain",
    "activity_type",
    "source_explorer",
    "gas_used",
    "gas_price",
]

# Required columns for holder DataFrame output
HOLDER_REQUIRED_COLUMNS = [
    "address",
    "balance",
    "stablecoin",
    "chain",
    "first_seen",
    "last_activity",
    "is_store_of_value",
    "source_explorer",
]


@dataclass
class CollectorOutput:
    """Output from a collector step.
    
    Contains the collected data as a DataFrame along with metadata
    about the collection run including any errors encountered.
    
    Attributes:
        transactions_df: DataFrame with transaction data
        holders_df: DataFrame with holder data
        explorer_name: Name of the explorer that collected the data
        chain: Blockchain network (ethereum, bsc, polygon)
        success: Whether collection succeeded (at least partial data)
        errors: List of error messages encountered during collection
        collection_time_seconds: Time taken for collection
    """
    transactions_df: pd.DataFrame
    holders_df: pd.DataFrame
    explorer_name: str
    chain: str
    success: bool = True
    errors: list[str] = field(default_factory=list)
    collection_time_seconds: float = 0.0


@dataclass
class AggregatedOutput:
    """Output from the aggregation step.
    
    Contains merged and deduplicated data from all collectors along with
    metadata about data completeness and any errors.
    
    Attributes:
        transactions_df: DataFrame with deduplicated transactions
        holders_df: DataFrame with merged holders
        successful_sources: List of collectors that succeeded
        failed_sources: List of collectors that failed
        completeness_ratio: Ratio of successful collectors (0.0 to 1.0)
        errors: Combined list of all errors from collectors
        run_metadata: Additional metadata about the aggregation run
    """
    transactions_df: pd.DataFrame
    holders_df: pd.DataFrame
    successful_sources: list[str] = field(default_factory=list)
    failed_sources: list[str] = field(default_factory=list)
    completeness_ratio: float = 1.0
    errors: list[str] = field(default_factory=list)
    run_metadata: dict = field(default_factory=dict)


def transactions_to_dataframe(transactions: list[Transaction]) -> pd.DataFrame:
    """Convert a list of Transaction objects to a pandas DataFrame.
    
    Args:
        transactions: List of Transaction objects
        
    Returns:
        DataFrame with transaction data and proper column types
    """
    if not transactions:
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=TRANSACTION_REQUIRED_COLUMNS)
    
    records = []
    for tx in transactions:
        records.append({
            "transaction_hash": tx.transaction_hash,
            "block_number": tx.block_number,
            "timestamp": tx.timestamp,
            "from_address": tx.from_address,
            "to_address": tx.to_address,
            "amount": float(tx.amount),  # Convert Decimal for DataFrame
            "stablecoin": tx.stablecoin,
            "chain": tx.chain,
            "activity_type": tx.activity_type.value,
            "source_explorer": tx.source_explorer,
            "gas_used": tx.gas_used,
            "gas_price": float(tx.gas_price) if tx.gas_price else None,
        })
    
    return pd.DataFrame(records)


def holders_to_dataframe(holders: list[Holder]) -> pd.DataFrame:
    """Convert a list of Holder objects to a pandas DataFrame.
    
    Args:
        holders: List of Holder objects
        
    Returns:
        DataFrame with holder data and proper column types
    """
    if not holders:
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=HOLDER_REQUIRED_COLUMNS)
    
    records = []
    for holder in holders:
        records.append({
            "address": holder.address,
            "balance": float(holder.balance),  # Convert Decimal for DataFrame
            "stablecoin": holder.stablecoin,
            "chain": holder.chain,
            "first_seen": holder.first_seen,
            "last_activity": holder.last_activity,
            "is_store_of_value": holder.is_store_of_value,
            "source_explorer": holder.source_explorer,
        })
    
    return pd.DataFrame(records)


def dataframe_to_transactions(df: pd.DataFrame) -> list[Transaction]:
    """Convert a DataFrame back to a list of Transaction objects.
    
    Args:
        df: DataFrame with transaction data
        
    Returns:
        List of Transaction objects
    """
    from collectors.models import ActivityType
    
    transactions = []
    for _, row in df.iterrows():
        tx = Transaction(
            transaction_hash=row["transaction_hash"],
            block_number=int(row["block_number"]),
            timestamp=pd.to_datetime(row["timestamp"]).to_pydatetime(),
            from_address=row["from_address"],
            to_address=row["to_address"],
            amount=Decimal(str(row["amount"])),
            stablecoin=row["stablecoin"],
            chain=row["chain"],
            activity_type=ActivityType(row["activity_type"]),
            source_explorer=row["source_explorer"],
            gas_used=int(row["gas_used"]) if pd.notna(row.get("gas_used")) else None,
            gas_price=Decimal(str(row["gas_price"])) if pd.notna(row.get("gas_price")) else None,
        )
        transactions.append(tx)
    return transactions


def dataframe_to_holders(df: pd.DataFrame) -> list[Holder]:
    """Convert a DataFrame back to a list of Holder objects.
    
    Args:
        df: DataFrame with holder data
        
    Returns:
        List of Holder objects
    """
    holders = []
    for _, row in df.iterrows():
        holder = Holder(
            address=row["address"],
            balance=Decimal(str(row["balance"])),
            stablecoin=row["stablecoin"],
            chain=row["chain"],
            first_seen=pd.to_datetime(row["first_seen"]).to_pydatetime(),
            last_activity=pd.to_datetime(row["last_activity"]).to_pydatetime(),
            is_store_of_value=bool(row["is_store_of_value"]),
            source_explorer=row["source_explorer"],
        )
        holders.append(holder)
    return holders


async def _run_etherscan_collection(
    stablecoins: list[str],
    date_range_days: int,
    max_records: int,
    run_id: Optional[str] = None,
) -> CollectorOutput:
    """Internal async function to run Etherscan collection.
    
    Args:
        stablecoins: List of stablecoin symbols to collect
        date_range_days: Number of days of data to collect
        max_records: Maximum records per stablecoin
        run_id: Optional run ID for logging
        
    Returns:
        CollectorOutput with collected data
    """
    import os
    from config.models import ExplorerConfig, RetryConfig
    from collectors.etherscan import EtherscanCollector
    
    # Ethereum contract addresses for stablecoins
    ETHEREUM_CONTRACTS = {
        "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
    }
    
    errors: list[str] = []
    transactions: list[Transaction] = []
    holders: list[Holder] = []
    collection_time = 0.0
    
    try:
        api_key = os.environ.get("ETHERSCAN_API_KEY", "")
        if not api_key:
            errors.append("ETHERSCAN_API_KEY environment variable not set")
            return CollectorOutput(
                transactions_df=transactions_to_dataframe([]),
                holders_df=holders_to_dataframe([]),
                explorer_name="etherscan",
                chain="ethereum",
                success=False,
                errors=errors,
            )
        
        config = ExplorerConfig(
            name="etherscan",
            chain="ethereum",
            base_url="https://api.etherscan.io/api",
            api_key=api_key,
            enabled=True,
        )
        retry_config = RetryConfig()
        
        # Build stablecoin contract mapping
        stablecoin_contracts = {
            coin: ETHEREUM_CONTRACTS[coin]
            for coin in stablecoins
            if coin in ETHEREUM_CONTRACTS
        }
        
        async with EtherscanCollector(config, retry_config) as collector:
            result: ExplorerData = await collector.collect_all(
                stablecoins=stablecoin_contracts,
                max_records=max_records,
                run_id=run_id,
            )
            transactions = result.transactions
            holders = result.holders
            errors.extend(result.errors)
            collection_time = result.collection_time_seconds
            
    except Exception as e:
        error_msg = f"Etherscan collection failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        errors.append(error_msg)
    
    success = len(transactions) > 0 or len(holders) > 0
    
    return CollectorOutput(
        transactions_df=transactions_to_dataframe(transactions),
        holders_df=holders_to_dataframe(holders),
        explorer_name="etherscan",
        chain="ethereum",
        success=success,
        errors=errors,
        collection_time_seconds=collection_time,
    )


@step
def etherscan_collector_step(
    stablecoins: list[str] = ["USDC", "USDT"],
    date_range_days: int = 7,
    max_records: int = 1000,
) -> CollectorOutput:
    """ZenML step to collect stablecoin data from Etherscan.
    
    Wraps the EtherscanCollector as a ZenML step with typed outputs.
    Handles API errors gracefully and returns partial results when possible.
    
    Args:
        stablecoins: List of stablecoin symbols to collect (default: USDC, USDT)
        date_range_days: Number of days of historical data to collect
        max_records: Maximum number of records per stablecoin
        
    Returns:
        CollectorOutput containing transactions and holders DataFrames
        
    Requirements: 9.1, 9.2, 9.4
    """
    import uuid
    run_id = str(uuid.uuid4())
    
    logger.info(
        "Starting Etherscan collection step",
        extra={
            "stablecoins": stablecoins,
            "date_range_days": date_range_days,
            "max_records": max_records,
            "run_id": run_id,
        }
    )
    
    # Run async collection in sync context
    result = asyncio.run(_run_etherscan_collection(
        stablecoins=stablecoins,
        date_range_days=date_range_days,
        max_records=max_records,
        run_id=run_id,
    ))
    
    logger.info(
        f"Etherscan collection complete: {len(result.transactions_df)} transactions, "
        f"{len(result.holders_df)} holders",
        extra={
            "transactions": len(result.transactions_df),
            "holders": len(result.holders_df),
            "success": result.success,
            "errors": len(result.errors),
        }
    )
    
    return result


async def _run_bscscan_collection(
    stablecoins: list[str],
    date_range_days: int,
    max_records: int,
    run_id: Optional[str] = None,
) -> CollectorOutput:
    """Internal async function to run BscScan collection.
    
    Args:
        stablecoins: List of stablecoin symbols to collect
        date_range_days: Number of days of data to collect
        max_records: Maximum records per stablecoin
        run_id: Optional run ID for logging
        
    Returns:
        CollectorOutput with collected data
    """
    import os
    from config.models import ExplorerConfig, RetryConfig
    from collectors.bscscan import BscscanCollector
    
    # BSC contract addresses for stablecoins
    BSC_CONTRACTS = {
        "USDC": "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d",
        "USDT": "0x55d398326f99059fF775485246999027B3197955",
    }
    
    errors: list[str] = []
    transactions: list[Transaction] = []
    holders: list[Holder] = []
    collection_time = 0.0
    
    try:
        api_key = os.environ.get("BSCSCAN_API_KEY", "")
        if not api_key:
            errors.append("BSCSCAN_API_KEY environment variable not set")
            return CollectorOutput(
                transactions_df=transactions_to_dataframe([]),
                holders_df=holders_to_dataframe([]),
                explorer_name="bscscan",
                chain="bsc",
                success=False,
                errors=errors,
            )
        
        config = ExplorerConfig(
            name="bscscan",
            chain="bsc",
            base_url="https://api.bscscan.com/api",
            api_key=api_key,
            enabled=True,
        )
        retry_config = RetryConfig()
        
        # Build stablecoin contract mapping
        stablecoin_contracts = {
            coin: BSC_CONTRACTS[coin]
            for coin in stablecoins
            if coin in BSC_CONTRACTS
        }
        
        async with BscscanCollector(config, retry_config) as collector:
            result: ExplorerData = await collector.collect_all(
                stablecoins=stablecoin_contracts,
                max_records=max_records,
                run_id=run_id,
            )
            transactions = result.transactions
            holders = result.holders
            errors.extend(result.errors)
            collection_time = result.collection_time_seconds
            
    except Exception as e:
        error_msg = f"BscScan collection failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        errors.append(error_msg)
    
    success = len(transactions) > 0 or len(holders) > 0
    
    return CollectorOutput(
        transactions_df=transactions_to_dataframe(transactions),
        holders_df=holders_to_dataframe(holders),
        explorer_name="bscscan",
        chain="bsc",
        success=success,
        errors=errors,
        collection_time_seconds=collection_time,
    )


@step
def bscscan_collector_step(
    stablecoins: list[str] = ["USDC", "USDT"],
    date_range_days: int = 7,
    max_records: int = 1000,
) -> CollectorOutput:
    """ZenML step to collect stablecoin data from BscScan.
    
    Wraps the BscscanCollector as a ZenML step with typed outputs.
    Handles API errors gracefully and returns partial results when possible.
    
    Args:
        stablecoins: List of stablecoin symbols to collect (default: USDC, USDT)
        date_range_days: Number of days of historical data to collect
        max_records: Maximum number of records per stablecoin
        
    Returns:
        CollectorOutput containing transactions and holders DataFrames
        
    Requirements: 9.1, 9.2
    """
    import uuid
    run_id = str(uuid.uuid4())
    
    logger.info(
        "Starting BscScan collection step",
        extra={
            "stablecoins": stablecoins,
            "date_range_days": date_range_days,
            "max_records": max_records,
            "run_id": run_id,
        }
    )
    
    # Run async collection in sync context
    result = asyncio.run(_run_bscscan_collection(
        stablecoins=stablecoins,
        date_range_days=date_range_days,
        max_records=max_records,
        run_id=run_id,
    ))
    
    logger.info(
        f"BscScan collection complete: {len(result.transactions_df)} transactions, "
        f"{len(result.holders_df)} holders",
        extra={
            "transactions": len(result.transactions_df),
            "holders": len(result.holders_df),
            "success": result.success,
            "errors": len(result.errors),
        }
    )
    
    return result


async def _run_polygonscan_collection(
    stablecoins: list[str],
    date_range_days: int,
    max_records: int,
    run_id: Optional[str] = None,
) -> CollectorOutput:
    """Internal async function to run Polygonscan collection.
    
    Args:
        stablecoins: List of stablecoin symbols to collect
        date_range_days: Number of days of data to collect
        max_records: Maximum records per stablecoin
        run_id: Optional run ID for logging
        
    Returns:
        CollectorOutput with collected data
    """
    import os
    from config.models import ExplorerConfig, RetryConfig
    from collectors.polygonscan import PolygonscanCollector
    
    # Polygon contract addresses for stablecoins
    POLYGON_CONTRACTS = {
        "USDC": "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",  # Native USDC
        "USDT": "0xc2132D05D31c914a87C6611C10748AEb04B58e8F",
    }
    
    errors: list[str] = []
    transactions: list[Transaction] = []
    holders: list[Holder] = []
    collection_time = 0.0
    
    try:
        api_key = os.environ.get("POLYGONSCAN_API_KEY", "")
        if not api_key:
            errors.append("POLYGONSCAN_API_KEY environment variable not set")
            return CollectorOutput(
                transactions_df=transactions_to_dataframe([]),
                holders_df=holders_to_dataframe([]),
                explorer_name="polygonscan",
                chain="polygon",
                success=False,
                errors=errors,
            )
        
        config = ExplorerConfig(
            name="polygonscan",
            chain="polygon",
            base_url="https://api.polygonscan.com/api",
            api_key=api_key,
            enabled=True,
        )
        retry_config = RetryConfig()
        
        # Build stablecoin contract mapping
        stablecoin_contracts = {
            coin: POLYGON_CONTRACTS[coin]
            for coin in stablecoins
            if coin in POLYGON_CONTRACTS
        }
        
        async with PolygonscanCollector(config, retry_config) as collector:
            result: ExplorerData = await collector.collect_all(
                stablecoins=stablecoin_contracts,
                max_records=max_records,
                run_id=run_id,
            )
            transactions = result.transactions
            holders = result.holders
            errors.extend(result.errors)
            collection_time = result.collection_time_seconds
            
    except Exception as e:
        error_msg = f"Polygonscan collection failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        errors.append(error_msg)
    
    success = len(transactions) > 0 or len(holders) > 0
    
    return CollectorOutput(
        transactions_df=transactions_to_dataframe(transactions),
        holders_df=holders_to_dataframe(holders),
        explorer_name="polygonscan",
        chain="polygon",
        success=success,
        errors=errors,
        collection_time_seconds=collection_time,
    )


@step
def polygonscan_collector_step(
    stablecoins: list[str] = ["USDC", "USDT"],
    date_range_days: int = 7,
    max_records: int = 1000,
) -> CollectorOutput:
    """ZenML step to collect stablecoin data from Polygonscan.
    
    Wraps the PolygonscanCollector as a ZenML step with typed outputs.
    Handles API errors gracefully and returns partial results when possible.
    
    Args:
        stablecoins: List of stablecoin symbols to collect (default: USDC, USDT)
        date_range_days: Number of days of historical data to collect
        max_records: Maximum number of records per stablecoin
        
    Returns:
        CollectorOutput containing transactions and holders DataFrames
        
    Requirements: 9.1, 9.2
    """
    import uuid
    run_id = str(uuid.uuid4())
    
    logger.info(
        "Starting Polygonscan collection step",
        extra={
            "stablecoins": stablecoins,
            "date_range_days": date_range_days,
            "max_records": max_records,
            "run_id": run_id,
        }
    )
    
    # Run async collection in sync context
    result = asyncio.run(_run_polygonscan_collection(
        stablecoins=stablecoins,
        date_range_days=date_range_days,
        max_records=max_records,
        run_id=run_id,
    ))
    
    logger.info(
        f"Polygonscan collection complete: {len(result.transactions_df)} transactions, "
        f"{len(result.holders_df)} holders",
        extra={
            "transactions": len(result.transactions_df),
            "holders": len(result.holders_df),
            "success": result.success,
            "errors": len(result.errors),
        }
    )
    
    return result


@step
def aggregate_data_step(
    etherscan_output: CollectorOutput,
    bscscan_output: CollectorOutput,
    polygonscan_output: CollectorOutput,
    min_successful_collectors: int = 2,
) -> AggregatedOutput:
    """ZenML step to aggregate and deduplicate data from all collectors.
    
    Merges transaction and holder data from all blockchain explorers,
    removes duplicates, and generates summary statistics.
    
    Args:
        etherscan_output: Output from Etherscan collector step
        bscscan_output: Output from BscScan collector step
        polygonscan_output: Output from Polygonscan collector step
        min_successful_collectors: Minimum number of collectors that must
            succeed for aggregation to proceed (default: 2, range: 1-3)
            
    Returns:
        AggregatedOutput with merged transactions and holders DataFrames
        
    Raises:
        ValueError: If fewer than min_successful_collectors succeed
        
    Requirements: 9.3, 9.4, 9.5
    """
    from datetime import datetime, timezone
    
    collector_outputs = [
        ("etherscan", etherscan_output),
        ("bscscan", bscscan_output),
        ("polygonscan", polygonscan_output),
    ]
    
    successful_sources: list[str] = []
    failed_sources: list[str] = []
    all_errors: list[str] = []
    
    # Track which collectors succeeded
    for name, output in collector_outputs:
        if output.success:
            successful_sources.append(name)
        else:
            failed_sources.append(name)
        all_errors.extend(output.errors)
    
    # Calculate completeness ratio
    total_collectors = 3
    completeness_ratio = len(successful_sources) / total_collectors
    
    logger.info(
        f"Aggregation starting: {len(successful_sources)}/{total_collectors} collectors succeeded",
        extra={
            "successful_sources": successful_sources,
            "failed_sources": failed_sources,
            "completeness_ratio": completeness_ratio,
        }
    )
    
    # Check minimum successful collectors requirement
    if len(successful_sources) < min_successful_collectors:
        error_msg = (
            f"Aggregation failed: Only {len(successful_sources)} collectors succeeded, "
            f"but {min_successful_collectors} required. Failed: {failed_sources}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Convert DataFrames back to model objects for aggregation
    all_transactions: list[Transaction] = []
    all_holders: list[Holder] = []
    
    for name, output in collector_outputs:
        if output.success:
            # Convert DataFrames to model objects
            transactions = dataframe_to_transactions(output.transactions_df)
            holders = dataframe_to_holders(output.holders_df)
            all_transactions.extend(transactions)
            all_holders.extend(holders)
            
            logger.debug(
                f"Added {len(transactions)} transactions and {len(holders)} holders from {name}"
            )
    
    # Use existing DataAggregator for deduplication and merging
    aggregator = DataAggregator()
    
    # Create ExplorerData objects for the aggregator
    explorer_data_list = []
    for name, output in collector_outputs:
        if output.success:
            explorer_data = ExplorerData(
                explorer_name=output.explorer_name,
                chain=output.chain,
                transactions=dataframe_to_transactions(output.transactions_df),
                holders=dataframe_to_holders(output.holders_df),
                errors=output.errors,
                collection_time_seconds=output.collection_time_seconds,
            )
            explorer_data_list.append(explorer_data)
    
    # Aggregate the data
    aggregated: AggregatedData = aggregator.aggregate(explorer_data_list)
    
    # Convert back to DataFrames
    transactions_df = transactions_to_dataframe(aggregated.transactions)
    holders_df = holders_to_dataframe(aggregated.holders)
    
    # Build run metadata
    run_metadata = {
        "aggregation_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_transactions": len(transactions_df),
        "total_holders": len(holders_df),
        "by_stablecoin": {
            coin: summary.to_dict()
            for coin, summary in aggregated.by_stablecoin.items()
        },
        "by_activity_type": aggregated.by_activity_type,
        "by_chain": aggregated.by_chain,
    }
    
    logger.info(
        f"Aggregation complete: {len(transactions_df)} transactions, "
        f"{len(holders_df)} holders",
        extra={
            "transactions": len(transactions_df),
            "holders": len(holders_df),
            "successful_sources": successful_sources,
            "failed_sources": failed_sources,
            "completeness_ratio": completeness_ratio,
        }
    )
    
    return AggregatedOutput(
        transactions_df=transactions_df,
        holders_df=holders_df,
        successful_sources=successful_sources,
        failed_sources=failed_sources,
        completeness_ratio=completeness_ratio,
        errors=all_errors,
        run_metadata=run_metadata,
    )
