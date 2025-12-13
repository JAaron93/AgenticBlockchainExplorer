"""ZenML data collection pipeline for stablecoin data.

This module defines the data collection pipeline that chains collector
steps with the aggregation step to collect stablecoin data from multiple
blockchain explorers.

Requirements: 10.3
"""

from typing import List

from zenml import pipeline

from pipelines.steps.collectors import (
    etherscan_collector_step,
    bscscan_collector_step,
    polygonscan_collector_step,
    aggregate_data_step,
    AggregatedOutput,
)


@pipeline(name="stablecoin_collection_pipeline")
def collection_pipeline(
    stablecoins: List[str] = ["USDC", "USDT"],
    date_range_days: int = 7,
    max_records: int = 1000,
    min_successful_collectors: int = 2,
) -> AggregatedOutput:
    """Pipeline to collect stablecoin data from all blockchain explorers.
    
    This pipeline chains the collector steps for Etherscan, BscScan, and
    Polygonscan with the aggregation step to produce a unified dataset.
    
    Args:
        stablecoins: List of stablecoin symbols to collect (default: USDC, USDT)
        date_range_days: Number of days of historical data to collect
        max_records: Maximum number of records per stablecoin per collector
        min_successful_collectors: Minimum collectors that must succeed
            for aggregation to proceed (default: 2, range: 1-3)
    
    Returns:
        AggregatedOutput containing:
        - transactions_df: Deduplicated transactions from all chains
        - holders_df: Merged holders from all chains
        - successful_sources: List of collectors that succeeded
        - failed_sources: List of collectors that failed
        - completeness_ratio: Ratio of successful collectors
        - errors: Combined list of all errors
        - run_metadata: Aggregation metadata
    
    Requirements: 10.3
    """
    # Collect data from each blockchain explorer
    etherscan_output = etherscan_collector_step(
        stablecoins=stablecoins,
        date_range_days=date_range_days,
        max_records=max_records,
    )
    
    bscscan_output = bscscan_collector_step(
        stablecoins=stablecoins,
        date_range_days=date_range_days,
        max_records=max_records,
    )
    
    polygonscan_output = polygonscan_collector_step(
        stablecoins=stablecoins,
        date_range_days=date_range_days,
        max_records=max_records,
    )
    
    # Aggregate and deduplicate data from all collectors
    aggregated_output = aggregate_data_step(
        etherscan_output=etherscan_output,
        bscscan_output=bscscan_output,
        polygonscan_output=polygonscan_output,
        min_successful_collectors=min_successful_collectors,
    )
    
    return aggregated_output


# Convenience function to run the pipeline with default parameters
def run_collection_pipeline(
    stablecoins: List[str] = ["USDC", "USDT"],
    date_range_days: int = 7,
    max_records: int = 1000,
    min_successful_collectors: int = 2,
):
    """Run the collection pipeline with specified parameters.
    
    Args:
        stablecoins: List of stablecoin symbols to collect
        date_range_days: Number of days of historical data
        max_records: Maximum records per stablecoin per collector
        min_successful_collectors: Minimum successful collectors required
        
    Returns:
        Pipeline run result
    """
    return collection_pipeline(
        stablecoins=stablecoins,
        date_range_days=date_range_days,
        max_records=max_records,
        min_successful_collectors=min_successful_collectors,
    )
