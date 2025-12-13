"""ZenML master pipeline for stablecoin data collection, analysis, and ML inference.

This module defines the master pipeline that orchestrates the entire workflow:
1. Data collection from blockchain explorers
2. Data analysis (activity, holder, time series, chain)
3. ML inference (SoV prediction, wallet classification)

The master pipeline is designed for weekly cron execution to update the live website.

Requirements: 10.1, 10.5
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from zenml import pipeline, get_step_context
from zenml.config.schedule import Schedule

from pipelines.steps.collectors import (
    etherscan_collector_step,
    bscscan_collector_step,
    polygonscan_collector_step,
    aggregate_data_step,
    AggregatedOutput,
)
from pipelines.steps.analysis import (
    activity_analysis_step,
    holder_analysis_step,
    time_series_step,
    chain_analysis_step,
    ActivityAnalysisOutput,
    HolderAnalysisOutput,
    ChainAnalysisOutput,
)
from pipelines.steps.ml import (
    feature_engineering_step,
    train_sov_predictor_step,
    predict_sov_step,
    FeatureEngineeringOutput,
    SoVPredictionOutput,
)
from pipelines.steps.wallet_classifier import (
    train_wallet_classifier_step,
    classify_wallets_step,
    WalletClassificationOutput,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Master Pipeline Output
# =============================================================================

@dataclass
class MasterPipelineOutput:
    """Combined output from the master pipeline.
    
    Contains all results from collection, analysis, and ML inference
    for consumption by the marimo visualization layer.
    
    Attributes:
        aggregated_data: Output from data collection and aggregation
        activity_breakdown: Activity type distribution metrics
        holder_metrics: Holder behavior analysis results
        time_series_df: Time-aggregated transaction data
        chain_metrics: Per-chain analysis metrics
        sov_predictions: Store-of-value predictions for holders
        wallet_classifications: Wallet behavior classifications
        run_metadata: Pipeline run metadata
    """
    aggregated_data: AggregatedOutput
    activity_breakdown: ActivityAnalysisOutput
    holder_metrics: HolderAnalysisOutput
    time_series_df: pd.DataFrame
    chain_metrics: ChainAnalysisOutput
    sov_predictions: Optional[SoVPredictionOutput] = None
    wallet_classifications: Optional[WalletClassificationOutput] = None
    run_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "activity_breakdown": self.activity_breakdown.to_dict(),
            "holder_metrics": self.holder_metrics.to_dict(),
            "time_series": self.time_series_df.to_dict(orient="records"),
            "chain_metrics": self.chain_metrics.to_dict(),
            "sov_predictions": (
                self.sov_predictions.to_dict() 
                if self.sov_predictions else None
            ),
            "wallet_classifications": (
                self.wallet_classifications.to_dict()
                if self.wallet_classifications else None
            ),
            "run_metadata": self.run_metadata,
        }


# =============================================================================
# Weekly Cron Schedule Configuration
# =============================================================================

def get_weekly_schedule(
    name: str = "weekly_stablecoin_analysis",
    cron_expression: str = "0 0 * * 0",  # Every Sunday at midnight
    start_time: Optional[datetime] = None,
) -> Schedule:
    """Create a weekly schedule for the master pipeline.
    
    Args:
        name: Name for the schedule
        cron_expression: Cron expression for scheduling
            Default: "0 0 * * 0" (every Sunday at midnight UTC)
        start_time: Optional start time for the schedule
        
    Returns:
        ZenML Schedule object for weekly execution
        
    Requirements: 10.5
    """
    return Schedule(
        name=name,
        cron_expression=cron_expression,
        start_time=start_time,
    )


# =============================================================================
# Master Pipeline Definition
# =============================================================================

@pipeline(name="stablecoin_master_pipeline")
def master_pipeline(
    stablecoins: List[str] = ["USDC", "USDT"],
    date_range_days: int = 7,
    max_records: int = 1000,
    min_successful_collectors: int = 2,
    aggregation: str = "daily",
    top_n_holders: int = 10,
    run_ml_inference: bool = True,
) -> MasterPipelineOutput:
    """Master pipeline orchestrating collection, analysis, and ML inference.
    
    This is the main pipeline triggered by weekly cron jobs for the live website.
    It chains:
    1. Data collection from Etherscan, BscScan, Polygonscan
    2. Data aggregation and deduplication
    3. Analysis (activity, holder, time series, chain)
    4. ML inference (SoV prediction, wallet classification)
    
    Args:
        stablecoins: List of stablecoin symbols to collect (default: USDC, USDT)
        date_range_days: Number of days of historical data to collect
        max_records: Maximum number of records per stablecoin per collector
        min_successful_collectors: Minimum collectors that must succeed
        aggregation: Time series aggregation period (daily/weekly/monthly)
        top_n_holders: Number of top holders to include in analysis
        run_ml_inference: Whether to run ML inference steps
    
    Returns:
        MasterPipelineOutput containing all results for visualization
        
    Requirements: 10.1, 10.5
    """
    # =========================================================================
    # Phase 1: Data Collection
    # =========================================================================
    
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
    
    # =========================================================================
    # Phase 2: Data Analysis
    # =========================================================================
    
    # Run activity type analysis
    activity_output = activity_analysis_step(
        transactions_df=aggregated_output.transactions_df,
    )
    
    # Run holder behavior analysis
    holder_output = holder_analysis_step(
        holders_df=aggregated_output.holders_df,
        transactions_df=aggregated_output.transactions_df,
        top_n=top_n_holders,
    )
    
    # Run time series analysis
    ts_df = time_series_step(
        transactions_df=aggregated_output.transactions_df,
        aggregation=aggregation,
    )
    
    # Run chain comparison analysis
    chain_output = chain_analysis_step(
        transactions_df=aggregated_output.transactions_df,
        holders_df=aggregated_output.holders_df,
    )
    
    # =========================================================================
    # Phase 3: ML Inference (Optional)
    # =========================================================================
    
    sov_predictions = None
    wallet_classifications = None
    
    if run_ml_inference:
        # Feature engineering
        features_output = feature_engineering_step(
            transactions_df=aggregated_output.transactions_df,
            holders_df=aggregated_output.holders_df,
        )
        
        # Train SoV predictor (or load from registry in production)
        sov_model_output = train_sov_predictor_step(
            features_df=features_output.features_df,
            holders_df=aggregated_output.holders_df,
        )
        
        # Run SoV prediction inference
        sov_predictions = predict_sov_step(
            features_df=features_output.features_df,
            model=sov_model_output.model,
        )
        
        # Train wallet classifier (or load from registry in production)
        wallet_model_output = train_wallet_classifier_step(
            features_df=features_output.features_df,
        )
        
        # Run wallet classification inference
        wallet_classifications = classify_wallets_step(
            features_df=features_output.features_df,
            model=wallet_model_output.model,
        )
    
    # =========================================================================
    # Build Output
    # =========================================================================
    
    run_metadata = {
        "pipeline_name": "stablecoin_master_pipeline",
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "stablecoins": stablecoins,
            "date_range_days": date_range_days,
            "max_records": max_records,
            "min_successful_collectors": min_successful_collectors,
            "aggregation": aggregation,
            "top_n_holders": top_n_holders,
            "run_ml_inference": run_ml_inference,
        },
        "data_completeness": {
            "successful_sources": aggregated_output.successful_sources,
            "failed_sources": aggregated_output.failed_sources,
            "completeness_ratio": aggregated_output.completeness_ratio,
        },
    }
    
    return MasterPipelineOutput(
        aggregated_data=aggregated_output,
        activity_breakdown=activity_output,
        holder_metrics=holder_output,
        time_series_df=ts_df,
        chain_metrics=chain_output,
        sov_predictions=sov_predictions,
        wallet_classifications=wallet_classifications,
        run_metadata=run_metadata,
    )


# =============================================================================
# Convenience Functions
# =============================================================================

def run_master_pipeline(
    stablecoins: List[str] = ["USDC", "USDT"],
    date_range_days: int = 7,
    max_records: int = 1000,
    min_successful_collectors: int = 2,
    aggregation: str = "daily",
    top_n_holders: int = 10,
    run_ml_inference: bool = True,
    schedule: Optional[Schedule] = None,
):
    """Run the master pipeline with specified parameters.
    
    Args:
        stablecoins: List of stablecoin symbols to collect
        date_range_days: Number of days of historical data
        max_records: Maximum records per stablecoin per collector
        min_successful_collectors: Minimum successful collectors required
        aggregation: Time series aggregation period
        top_n_holders: Number of top holders to include
        run_ml_inference: Whether to run ML inference steps
        schedule: Optional schedule for recurring execution
        
    Returns:
        PipelineRunResponse from ZenML pipeline execution
    """
    pipeline_instance = master_pipeline.with_options(
        schedule=schedule,
    ) if schedule else master_pipeline
    
    return pipeline_instance(
        stablecoins=stablecoins,
        date_range_days=date_range_days,
        max_records=max_records,
        min_successful_collectors=min_successful_collectors,
        aggregation=aggregation,
        top_n_holders=top_n_holders,
        run_ml_inference=run_ml_inference,
    )


def run_weekly_master_pipeline(
    stablecoins: List[str] = ["USDC", "USDT"],
    date_range_days: int = 7,
    max_records: int = 1000,
    min_successful_collectors: int = 2,
    aggregation: str = "daily",
    top_n_holders: int = 10,
    run_ml_inference: bool = True,
    cron_expression: str = "0 0 * * 0",
):
    """Run the master pipeline with weekly scheduling.
    
    Convenience function to set up weekly cron execution.
    
    Args:
        stablecoins: List of stablecoin symbols to collect
        date_range_days: Number of days of historical data
        max_records: Maximum records per stablecoin per collector
        min_successful_collectors: Minimum successful collectors required
        aggregation: Time series aggregation period (daily/weekly/monthly)
        top_n_holders: Number of top holders to include in analysis
        run_ml_inference: Whether to run ML inference steps
        cron_expression: Cron expression for scheduling
            Default: "0 0 * * 0" (every Sunday at midnight UTC)
            
    Returns:
        PipelineRunResponse from ZenML pipeline execution
        
    Requirements: 10.5
    """
    schedule = get_weekly_schedule(cron_expression=cron_expression)
    
    return run_master_pipeline(
        stablecoins=stablecoins,
        date_range_days=date_range_days,
        max_records=max_records,
        min_successful_collectors=min_successful_collectors,
        aggregation=aggregation,
        top_n_holders=top_n_holders,
        run_ml_inference=run_ml_inference,
        schedule=schedule,
    )
