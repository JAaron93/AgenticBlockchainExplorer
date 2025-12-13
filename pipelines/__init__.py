"""ZenML pipelines for stablecoin data collection and analysis."""

from pipelines.collection_pipeline import (
    collection_pipeline,
    run_collection_pipeline,
)

from pipelines.analysis_pipeline import (
    analysis_pipeline,
    run_analysis_pipeline,
    AnalysisPipelineOutput,
)

__all__ = [
    # Collection pipeline
    "collection_pipeline",
    "run_collection_pipeline",
    # Analysis pipeline
    "analysis_pipeline",
    "run_analysis_pipeline",
    "AnalysisPipelineOutput",
]
