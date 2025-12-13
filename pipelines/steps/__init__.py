"""ZenML steps for stablecoin pipelines."""

from pipelines.steps.collectors import (
    etherscan_collector_step,
    bscscan_collector_step,
    polygonscan_collector_step,
    aggregate_data_step,
    CollectorOutput,
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
    SoVModelOutput,
    SoVPredictionOutput,
    REQUIRED_FEATURES,
    compute_holder_features,
    validate_features,
)

__all__ = [
    # Collector steps
    "etherscan_collector_step",
    "bscscan_collector_step",
    "polygonscan_collector_step",
    "aggregate_data_step",
    "CollectorOutput",
    "AggregatedOutput",
    # Analysis steps
    "activity_analysis_step",
    "holder_analysis_step",
    "time_series_step",
    "chain_analysis_step",
    "ActivityAnalysisOutput",
    "HolderAnalysisOutput",
    "ChainAnalysisOutput",
    # ML steps
    "feature_engineering_step",
    "train_sov_predictor_step",
    "predict_sov_step",
    "FeatureEngineeringOutput",
    "SoVModelOutput",
    "SoVPredictionOutput",
    "REQUIRED_FEATURES",
    "compute_holder_features",
    "validate_features",
]
