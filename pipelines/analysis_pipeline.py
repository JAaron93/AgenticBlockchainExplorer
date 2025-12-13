"""ZenML analysis pipeline for stablecoin data.

This module defines the analysis pipeline that chains all analysis steps
to produce comprehensive analysis results as versioned artifacts.

Requirements: 10.2, 10.4
"""

from dataclasses import dataclass, field
import pandas as pd
from zenml import pipeline

from pipelines.steps.analysis import (
    activity_analysis_step,
    holder_analysis_step,
    time_series_step,
    chain_analysis_step,
    ActivityAnalysisOutput,
    HolderAnalysisOutput,
    ChainAnalysisOutput,
)


@dataclass
class AnalysisPipelineOutput:
    """Combined output from all analysis steps.
    
    This dataclass aggregates all analysis results for easy access
    and serialization.
    
    Attributes:
        activity_breakdown: Activity type distribution metrics
        holder_metrics: Holder behavior analysis results
        time_series_df: Time-aggregated transaction data
        chain_metrics: Per-chain analysis metrics
        run_metadata: Pipeline run metadata
    """
    activity_breakdown: ActivityAnalysisOutput
    holder_metrics: HolderAnalysisOutput
    time_series_df: pd.DataFrame
    chain_metrics: ChainAnalysisOutput
    run_metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "activity_breakdown": self.activity_breakdown.to_dict(),
            "holder_metrics": self.holder_metrics.to_dict(),
            "time_series": self.time_series_df.to_dict(orient="records"),
            "chain_metrics": self.chain_metrics.to_dict(),
            "run_metadata": self.run_metadata,
        }


@pipeline(name="stablecoin_analysis_pipeline")
def analysis_pipeline(
    transactions_df: pd.DataFrame,
    holders_df: pd.DataFrame,
    aggregation: str = "daily",
    top_n_holders: int = 10,
) -> AnalysisPipelineOutput:
    """Pipeline to run all analysis steps on stablecoin data.
    
    This pipeline chains all analysis steps to produce comprehensive
    analysis results as versioned ZenML artifacts.
    
    Args:
        transactions_df: DataFrame with transaction data
        holders_df: DataFrame with holder data
        aggregation: Time series aggregation period (daily/weekly/monthly)
        top_n_holders: Number of top holders to include in analysis
    
    Returns:
        AnalysisPipelineOutput containing all analysis results as
        versioned ZenML artifacts.
    
    Requirements: 10.2, 10.4
    """
    # Run activity type analysis
    activity_output = activity_analysis_step(
        transactions_df=transactions_df,
    )
    
    # Run holder behavior analysis
    holder_output = holder_analysis_step(
        holders_df=holders_df,
        transactions_df=transactions_df,
        top_n=top_n_holders,
    )
    
    # Run time series analysis
    ts_df = time_series_step(
        transactions_df=transactions_df,
        aggregation=aggregation,
    )
    
    # Run chain comparison analysis
    chain_output = chain_analysis_step(
        transactions_df=transactions_df,
        holders_df=holders_df,
    )
    
    return AnalysisPipelineOutput(
        activity_breakdown=activity_output,
        holder_metrics=holder_output,
        time_series_df=ts_df,
        chain_metrics=chain_output,
        run_metadata={
            "aggregation": aggregation,
            "top_n_holders": top_n_holders,
        },
    )


# Convenience function to run the pipeline
def run_analysis_pipeline(
    transactions_df: pd.DataFrame,
    holders_df: pd.DataFrame,
    aggregation: str = "daily",
    top_n_holders: int = 10,
):
    """Run the analysis pipeline with specified parameters.
    
    Args:
        transactions_df: DataFrame with transaction data
        holders_df: DataFrame with holder data
        aggregation: Time series aggregation period
        top_n_holders: Number of top holders to include
        
    Returns:
        PipelineRunResponse from ZenML pipeline execution
    """
    return analysis_pipeline(
        transactions_df=transactions_df,
        holders_df=holders_df,
        aggregation=aggregation,
        top_n_holders=top_n_holders,
    )
