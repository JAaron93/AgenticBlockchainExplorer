"""ZenML steps for stablecoin data analysis.

This module wraps the analysis functions from the notebook as ZenML steps
with typed inputs and outputs for pipeline orchestration.

Requirements: 10.1, 10.2
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
from zenml import step

from notebooks.stablecoin_analysis_functions import (
    ActivityBreakdown,
    HolderMetrics,
    ChainMetrics,
    analyze_activity_types,
    analyze_holders,
    analyze_time_series,
    analyze_by_chain,
    get_top_holders,
    TopHolder,
)


logger = logging.getLogger(__name__)


@dataclass
class ActivityAnalysisOutput:
    """Output from activity analysis step.
    
    Attributes:
        counts: Transaction counts by activity type
        percentages: Percentage distribution by activity type
        volumes: Total volumes by activity type (as string for serialization)
        volume_percentages: Volume percentage distribution by activity type
    """
    counts: Dict[str, int]
    percentages: Dict[str, float]
    volumes: Dict[str, str]  # Decimal serialized as string
    volume_percentages: Dict[str, float]
    
    @classmethod
    def from_breakdown(cls, breakdown: ActivityBreakdown) -> "ActivityAnalysisOutput":
        """Create from ActivityBreakdown dataclass."""
        return cls(
            counts=breakdown.counts,
            percentages=breakdown.percentages,
            volumes={k: str(v) for k, v in breakdown.volumes.items()},
            volume_percentages=breakdown.volume_percentages,
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "counts": self.counts,
            "percentages": self.percentages,
            "volumes": self.volumes,
            "volume_percentages": self.volume_percentages,
        }


@dataclass
class HolderAnalysisOutput:
    """Output from holder analysis step.
    
    Attributes:
        total_holders: Total number of holders
        sov_count: Number of store-of-value holders
        sov_percentage: Percentage of SoV holders
        avg_balance_sov: Average balance of SoV holders (as string)
        avg_balance_active: Average balance of active holders (as string)
        avg_holding_period_days: Average holding period for SoV holders
        median_holding_period_days: Median holding period for SoV holders
        top_holders: List of top holders by balance
    """
    total_holders: int
    sov_count: int
    sov_percentage: float
    avg_balance_sov: str  # Decimal serialized as string
    avg_balance_active: str  # Decimal serialized as string
    avg_holding_period_days: float
    median_holding_period_days: float
    top_holders: List[dict] = field(default_factory=list)
    
    @classmethod
    def from_metrics(
        cls,
        metrics: HolderMetrics,
        top_holders: List[TopHolder],
    ) -> "HolderAnalysisOutput":
        """Create from HolderMetrics and top holders list."""
        return cls(
            total_holders=metrics.total_holders,
            sov_count=metrics.sov_count,
            sov_percentage=metrics.sov_percentage,
            avg_balance_sov=str(metrics.avg_balance_sov),
            avg_balance_active=str(metrics.avg_balance_active),
            avg_holding_period_days=metrics.avg_holding_period_days,
            median_holding_period_days=metrics.median_holding_period_days,
            top_holders=[
                {
                    "address": h.address,
                    "balance": str(h.balance),
                    "stablecoin": h.stablecoin,
                    "chain": h.chain,
                    "is_store_of_value": h.is_store_of_value,
                }
                for h in top_holders
            ],
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_holders": self.total_holders,
            "sov_count": self.sov_count,
            "sov_percentage": self.sov_percentage,
            "avg_balance_sov": self.avg_balance_sov,
            "avg_balance_active": self.avg_balance_active,
            "avg_holding_period_days": self.avg_holding_period_days,
            "median_holding_period_days": self.median_holding_period_days,
            "top_holders": self.top_holders,
        }


@dataclass
class ChainAnalysisOutput:
    """Output from chain analysis step.
    
    Attributes:
        chain_metrics: List of metrics for each chain
    """
    chain_metrics: List[dict]
    
    @classmethod
    def from_metrics(cls, metrics: List[ChainMetrics]) -> "ChainAnalysisOutput":
        """Create from list of ChainMetrics."""
        return cls(
            chain_metrics=[m.to_dict() for m in metrics],
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "chain_metrics": self.chain_metrics,
        }


@step
def activity_analysis_step(
    transactions_df: pd.DataFrame,
) -> ActivityAnalysisOutput:
    """ZenML step to calculate activity type distribution.
    
    Wraps the analyze_activity_types function as a ZenML step with
    typed outputs for pipeline orchestration.
    
    Args:
        transactions_df: DataFrame with transaction data including
            'activity_type' and 'amount' columns
            
    Returns:
        ActivityAnalysisOutput with counts, percentages, and volumes
        by activity type
        
    Requirements: 10.1, 10.2
    """
    logger.info(
        f"Starting activity analysis step with {len(transactions_df)} transactions"
    )
    
    try:
        breakdown = analyze_activity_types(transactions_df)
        
        logger.info(
            "Activity analysis complete",
            extra={
                "counts": breakdown.counts,
                "total_transactions": sum(breakdown.counts.values()),
            }
        )
        
        return ActivityAnalysisOutput.from_breakdown(breakdown)
        
    except Exception as e:
        logger.error(f"Activity analysis failed: {e}", exc_info=True)
        raise


@step
def holder_analysis_step(
    holders_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    top_n: int = 10,
) -> HolderAnalysisOutput:
    """ZenML step to analyze holder behavior patterns.
    
    Wraps the analyze_holders and get_top_holders functions as a ZenML
    step with typed outputs for pipeline orchestration.
    
    Args:
        holders_df: DataFrame with holder data including 'is_store_of_value',
            'balance', and optionally 'holding_period_days' columns
        transactions_df: DataFrame with transaction data (for future use)
        top_n: Number of top holders to include (default: 10)
            
    Returns:
        HolderAnalysisOutput with holder metrics and top holders list
        
    Requirements: 10.1, 10.2
    """
    logger.info(
        f"Starting holder analysis step with {len(holders_df)} holders"
    )
    
    try:
        # Analyze holder metrics
        metrics = analyze_holders(holders_df, transactions_df)
        
        # Get top holders
        top_holders = get_top_holders(holders_df, n=top_n)
        
        logger.info(
            "Holder analysis complete",
            extra={
                "total_holders": metrics.total_holders,
                "sov_count": metrics.sov_count,
                "sov_percentage": metrics.sov_percentage,
                "top_holders_count": len(top_holders),
            }
        )
        
        return HolderAnalysisOutput.from_metrics(metrics, top_holders)
        
    except Exception as e:
        logger.error(f"Holder analysis failed: {e}", exc_info=True)
        raise


@step
def time_series_step(
    transactions_df: pd.DataFrame,
    aggregation: str = "daily",
) -> pd.DataFrame:
    """ZenML step to create time-series aggregations.
    
    Wraps the analyze_time_series function as a ZenML step with
    typed outputs for pipeline orchestration.
    
    Args:
        transactions_df: DataFrame with transaction data including
            'timestamp', 'amount', 'activity_type', and 'stablecoin' columns
        aggregation: Aggregation period - "daily", "weekly", or "monthly"
            
    Returns:
        DataFrame with time-aggregated data including columns:
        - period: The time period (date)
        - activity_type: Activity type
        - stablecoin: Stablecoin type
        - transaction_count: Number of transactions in period
        - volume: Total volume in period
        
    Requirements: 10.1, 10.2
    """
    logger.info(
        f"Starting time series analysis with {len(transactions_df)} transactions, "
        f"aggregation={aggregation}"
    )
    
    try:
        result_df = analyze_time_series(transactions_df, aggregation)
        
        logger.info(
            f"Time series analysis complete: {len(result_df)} aggregated periods"
        )
        
        return result_df
        
    except Exception as e:
        logger.error(f"Time series analysis failed: {e}", exc_info=True)
        raise


@step
def chain_analysis_step(
    transactions_df: pd.DataFrame,
    holders_df: Optional[pd.DataFrame] = None,
) -> ChainAnalysisOutput:
    """ZenML step to calculate per-chain metrics.
    
    Wraps the analyze_by_chain function as a ZenML step with
    typed outputs for pipeline orchestration.
    
    Args:
        transactions_df: DataFrame with transaction data including 'chain',
            'amount', 'activity_type', and optionally 'gas_used', 'gas_price'
        holders_df: Optional DataFrame with holder data for SoV ratio
            
    Returns:
        ChainAnalysisOutput with metrics for each chain including
        transaction count, volume, average transaction size, gas costs,
        and activity distribution
        
    Requirements: 10.1, 10.2
    """
    logger.info(
        f"Starting chain analysis with {len(transactions_df)} transactions"
    )
    
    try:
        chain_metrics = analyze_by_chain(transactions_df, holders_df)
        
        logger.info(
            "Chain analysis complete",
            extra={
                "chains_analyzed": len(chain_metrics),
                "chains": [m.chain for m in chain_metrics],
            }
        )
        
        return ChainAnalysisOutput.from_metrics(chain_metrics)
        
    except Exception as e:
        logger.error(f"Chain analysis failed: {e}", exc_info=True)
        raise
