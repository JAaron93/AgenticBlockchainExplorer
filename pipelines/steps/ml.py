"""ZenML steps for ML feature engineering and prediction.

This module implements ML steps for:
- Feature engineering from transaction and holder data
- Store-of-Value (SoV) prediction training and inference
- Wallet behavior classification

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 12.2
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from zenml import step

logger = logging.getLogger(__name__)


# =============================================================================
# Feature Engineering Constants
# =============================================================================

# Required features for ML models (from Requirements 11.1, 12.2)
REQUIRED_FEATURES = [
    "transaction_count",
    "avg_transaction_size",
    "balance_percentile",
    "holding_period_days",
    "activity_recency_days",
    "transaction_frequency",
    "balance_volatility",
    "cross_chain_flag",
]


# =============================================================================
# Data Classes for ML Outputs
# =============================================================================

@dataclass
class FeatureEngineeringOutput:
    """Output from feature engineering step.
    
    Attributes:
        features_df: DataFrame with extracted features for each holder
        feature_names: List of feature column names
        holder_count: Number of holders processed
        metadata: Additional metadata about feature extraction
    """
    features_df: pd.DataFrame
    feature_names: List[str]
    holder_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "feature_names": self.feature_names,
            "holder_count": self.holder_count,
            "metadata": self.metadata,
        }


@dataclass
class SoVModelOutput:
    """Output from SoV prediction training step.
    
    Attributes:
        model: Trained model object
        metrics: Evaluation metrics (precision, recall, F1, AUC-ROC)
        hyperparameters: Model hyperparameters used
        feature_importances: Feature importance scores
        training_metadata: Metadata about training run
    """
    model: Any  # sklearn or xgboost model
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    feature_importances: Dict[str, float]
    training_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization (excluding model)."""
        return {
            "metrics": self.metrics,
            "hyperparameters": self.hyperparameters,
            "feature_importances": self.feature_importances,
            "training_metadata": self.training_metadata,
        }


@dataclass
class SoVPredictionOutput:
    """Output from SoV prediction inference step.
    
    Attributes:
        predictions_df: DataFrame with predictions for each holder
        prediction_count: Number of predictions made
        metadata: Additional metadata about inference run
    """
    predictions_df: pd.DataFrame
    prediction_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "prediction_count": self.prediction_count,
            "metadata": self.metadata,
        }


# =============================================================================
# Feature Engineering Functions
# =============================================================================

def compute_holder_features(
    holders_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    reference_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Compute ML features for each holder from transaction history.
    
    Extracts the following features per holder:
    - transaction_count: Total number of transactions
    - avg_transaction_size: Mean transaction amount
    - balance_percentile: Holder's balance rank (0-100)
    - holding_period_days: Days since first activity
    - activity_recency_days: Days since last activity
    - transaction_frequency: Transactions per day
    - balance_volatility: Std dev of transaction amounts (proxy for balance changes)
    - cross_chain_flag: Whether holder is active on multiple chains
    
    Args:
        holders_df: DataFrame with holder data
        transactions_df: DataFrame with transaction data
        reference_date: Reference date for recency calculations (default: now)
        
    Returns:
        DataFrame with one row per holder and feature columns
        
    Requirements: 11.1, 12.2
    """
    if reference_date is None:
        reference_date = datetime.now(timezone.utc)
    
    if len(holders_df) == 0:
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=["address"] + REQUIRED_FEATURES)
    
    # Ensure we have the required columns
    holders_df = holders_df.copy()
    
    # Normalize addresses for joining
    holders_df["address_lower"] = holders_df["address"].str.lower()
    
    # Initialize features DataFrame
    features = holders_df[["address", "address_lower"]].copy()
    
    # Calculate balance percentile
    if "balance" in holders_df.columns:
        features["balance_percentile"] = holders_df["balance"].rank(pct=True) * 100
    else:
        features["balance_percentile"] = 50.0  # Default to median
    
    # Calculate holding period and recency from holder data
    if "first_seen" in holders_df.columns:
        first_seen = pd.to_datetime(holders_df["first_seen"])
        features["holding_period_days"] = (
            (reference_date - first_seen).dt.total_seconds() / 86400
        ).fillna(0).clip(lower=0)
    else:
        features["holding_period_days"] = 0.0
    
    if "last_activity" in holders_df.columns:
        last_activity = pd.to_datetime(holders_df["last_activity"])
        features["activity_recency_days"] = (
            (reference_date - last_activity).dt.total_seconds() / 86400
        ).fillna(0).clip(lower=0)
    else:
        features["activity_recency_days"] = 0.0
    
    # Calculate transaction-based features
    if len(transactions_df) > 0:
        transactions_df = transactions_df.copy()
        
        # Normalize addresses for joining
        transactions_df["from_lower"] = transactions_df["from_address"].str.lower()
        transactions_df["to_lower"] = transactions_df["to_address"].str.lower()
        
        # Count transactions per holder (as sender or receiver)
        from_counts = transactions_df.groupby("from_lower").size()
        to_counts = transactions_df.groupby("to_lower").size()
        
        features["tx_from_count"] = features["address_lower"].map(from_counts).fillna(0)
        features["tx_to_count"] = features["address_lower"].map(to_counts).fillna(0)
        features["transaction_count"] = features["tx_from_count"] + features["tx_to_count"]
        
        # Calculate average transaction size per holder
        from_amounts = transactions_df.groupby("from_lower")["amount"].mean()
        to_amounts = transactions_df.groupby("to_lower")["amount"].mean()
        
        features["avg_from_amount"] = features["address_lower"].map(from_amounts).fillna(0)
        features["avg_to_amount"] = features["address_lower"].map(to_amounts).fillna(0)
        
        # Average of send and receive amounts (or just one if only one exists)
        features["avg_transaction_size"] = features.apply(
            lambda row: (
                (row["avg_from_amount"] + row["avg_to_amount"]) / 2
                if row["avg_from_amount"] > 0 and row["avg_to_amount"] > 0
                else max(row["avg_from_amount"], row["avg_to_amount"])
            ),
            axis=1
        )
        
        # Calculate balance volatility (std dev of transaction amounts)
        from_std = transactions_df.groupby("from_lower")["amount"].std()
        to_std = transactions_df.groupby("to_lower")["amount"].std()
        
        features["from_std"] = features["address_lower"].map(from_std).fillna(0)
        features["to_std"] = features["address_lower"].map(to_std).fillna(0)
        features["balance_volatility"] = (features["from_std"] + features["to_std"]) / 2
        features["balance_volatility"] = features["balance_volatility"].fillna(0)
        
        # Calculate cross-chain flag
        from_chains = transactions_df.groupby("from_lower")["chain"].nunique()
        to_chains = transactions_df.groupby("to_lower")["chain"].nunique()
        
        features["from_chains"] = features["address_lower"].map(from_chains).fillna(0)
        features["to_chains"] = features["address_lower"].map(to_chains).fillna(0)
        features["cross_chain_flag"] = (
            (features["from_chains"] > 1) | (features["to_chains"] > 1)
        ).astype(int)
        
        # Clean up temporary columns
        features = features.drop(columns=[
            "tx_from_count", "tx_to_count",
            "avg_from_amount", "avg_to_amount",
            "from_std", "to_std",
            "from_chains", "to_chains",
        ])
    else:
        # No transactions - set defaults
        features["transaction_count"] = 0
        features["avg_transaction_size"] = 0.0
        features["balance_volatility"] = 0.0
        features["cross_chain_flag"] = 0
    
    # Calculate transaction frequency (tx per day)
    features["transaction_frequency"] = features.apply(
        lambda row: (
            row["transaction_count"] / max(row["holding_period_days"], 1)
        ),
        axis=1
    )
    
    # Drop temporary columns and select final features
    features = features.drop(columns=["address_lower"])
    
    # Ensure all required features are present
    for feat in REQUIRED_FEATURES:
        if feat not in features.columns:
            features[feat] = 0.0
    
    # Select only required columns in correct order
    result = features[["address"] + REQUIRED_FEATURES].copy()
    
    # Convert cross_chain_flag to int
    result["cross_chain_flag"] = result["cross_chain_flag"].astype(int)
    
    return result


def validate_features(features_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate that feature DataFrame has all required columns.
    
    Args:
        features_df: DataFrame with extracted features
        
    Returns:
        Tuple of (is_valid, list of missing features)
    """
    missing = []
    for feat in REQUIRED_FEATURES:
        if feat not in features_df.columns:
            missing.append(feat)
    
    return len(missing) == 0, missing


# =============================================================================
# ZenML Steps
# =============================================================================

@step
def feature_engineering_step(
    transactions_df: pd.DataFrame,
    holders_df: pd.DataFrame,
) -> FeatureEngineeringOutput:
    """ZenML step to extract ML features from transaction and holder data.
    
    Extracts features for each holder including:
    - transaction_count: Total transactions for holder
    - avg_transaction_size: Mean transaction amount
    - balance_percentile: Holder's balance rank (0-100)
    - holding_period_days: Days since first activity
    - activity_recency_days: Days since last activity
    - transaction_frequency: Transactions per day
    - balance_volatility: Std dev of balance changes
    - cross_chain_flag: Whether holder active on multiple chains
    
    Args:
        transactions_df: DataFrame with transaction data
        holders_df: DataFrame with holder data
        
    Returns:
        FeatureEngineeringOutput with features DataFrame and metadata
        
    Requirements: 11.1, 12.2
    """
    logger.info(
        f"Starting feature engineering with {len(holders_df)} holders, "
        f"{len(transactions_df)} transactions"
    )
    
    try:
        # Compute features
        features_df = compute_holder_features(holders_df, transactions_df)
        
        # Validate features
        is_valid, missing = validate_features(features_df)
        if not is_valid:
            raise ValueError(f"Feature engineering failed: missing features {missing}")
        
        # Build metadata
        metadata = {
            "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
            "input_holders": len(holders_df),
            "input_transactions": len(transactions_df),
            "output_features": len(features_df),
            "feature_stats": {
                feat: {
                    "mean": float(features_df[feat].mean()),
                    "std": float(features_df[feat].std()),
                    "min": float(features_df[feat].min()),
                    "max": float(features_df[feat].max()),
                }
                for feat in REQUIRED_FEATURES
            },
        }
        
        logger.info(
            f"Feature engineering complete: {len(features_df)} feature vectors",
            extra={
                "holder_count": len(features_df),
                "feature_count": len(REQUIRED_FEATURES),
            }
        )
        
        return FeatureEngineeringOutput(
            features_df=features_df,
            feature_names=REQUIRED_FEATURES,
            holder_count=len(features_df),
            metadata=metadata,
        )
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}", exc_info=True)
        raise



# =============================================================================
# SoV Prediction Training Step
# =============================================================================

@step
def train_sov_predictor_step(
    features_df: pd.DataFrame,
    holders_df: pd.DataFrame,
    algorithm: str = "xgboost",
    n_estimators: int = 100,
    max_depth: int = 10,
    learning_rate: float = 0.1,
    test_size: float = 0.2,
    random_state: int = 42,
) -> SoVModelOutput:
    """ZenML step to train binary classifier for SoV prediction.
    
    Trains a model to predict whether a holder will become a store-of-value
    user based on their transaction history features.
    
    Target: is_store_of_value (boolean)
    Algorithm: XGBoost (default) or RandomForest
    
    Args:
        features_df: DataFrame with extracted features (from feature_engineering_step)
        holders_df: DataFrame with holder data including is_store_of_value labels
        algorithm: Model algorithm - "xgboost" or "random_forest"
        n_estimators: Number of trees in the ensemble
        max_depth: Maximum depth of trees (constrained to <= 10 per Req 11.2)
        learning_rate: Learning rate for XGBoost (ignored for RandomForest)
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        SoVModelOutput with trained model, metrics, and metadata
        
    Requirements: 11.2, 11.3, 11.4
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )
    
    logger.info(
        f"Starting SoV predictor training with {len(features_df)} samples, "
        f"algorithm={algorithm}"
    )
    
    try:
        # Constrain max_depth per Requirements 11.2
        max_depth = min(max_depth, 10)
        
        # Merge features with labels
        features_df = features_df.copy()
        holders_df = holders_df.copy()
        
        # Normalize addresses for joining
        features_df["address_lower"] = features_df["address"].str.lower()
        holders_df["address_lower"] = holders_df["address"].str.lower()
        
        # Get labels
        labels = holders_df[["address_lower", "is_store_of_value"]].drop_duplicates()
        
        # Merge features with labels
        merged = features_df.merge(labels, on="address_lower", how="inner")
        
        if len(merged) == 0:
            raise ValueError("No matching holders found between features and labels")
        
        # Prepare feature matrix and target
        X = merged[REQUIRED_FEATURES].values
        y = merged["is_store_of_value"].astype(int).values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train model
        if algorithm == "xgboost":
            import xgboost as xgb
            
            # Calculate class weights for imbalanced data
            scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
            
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
                use_label_encoder=False,
                eval_metric="logloss",
            )
        else:  # random_forest
            from sklearn.ensemble import RandomForestClassifier
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight="balanced",
                random_state=random_state,
            )
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else 0.5,
        }
        
        # Get feature importances
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            importances = np.zeros(len(REQUIRED_FEATURES))
        
        feature_importances = {
            feat: float(imp)
            for feat, imp in zip(REQUIRED_FEATURES, importances)
        }
        
        # Build hyperparameters dict
        hyperparameters = {
            "algorithm": algorithm,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate if algorithm == "xgboost" else None,
            "test_size": test_size,
            "random_state": random_state,
        }
        
        # Build training metadata
        training_metadata = {
            "training_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_samples": len(merged),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "positive_ratio_train": float(y_train.mean()),
            "positive_ratio_test": float(y_test.mean()),
        }
        
        logger.info(
            f"SoV predictor training complete",
            extra={
                "metrics": metrics,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }
        )
        
        return SoVModelOutput(
            model=model,
            metrics=metrics,
            hyperparameters=hyperparameters,
            feature_importances=feature_importances,
            training_metadata=training_metadata,
        )
        
    except Exception as e:
        logger.error(f"SoV predictor training failed: {e}", exc_info=True)
        raise


# =============================================================================
# SoV Prediction Inference Step
# =============================================================================

@step
def predict_sov_step(
    features_df: pd.DataFrame,
    model: Any,
) -> SoVPredictionOutput:
    """ZenML step to run SoV prediction inference.
    
    Predicts SoV probability for each holder using a trained model.
    
    Args:
        features_df: DataFrame with extracted features (from feature_engineering_step)
        model: Trained SoV prediction model (from train_sov_predictor_step or registry)
        
    Returns:
        SoVPredictionOutput with predictions DataFrame containing:
        - address: Holder address
        - sov_probability: Probability of being SoV (0.0 to 1.0)
        - predicted_class: Predicted class (True/False)
        
    Requirements: 11.5
    """
    logger.info(f"Starting SoV prediction inference with {len(features_df)} holders")
    
    try:
        if len(features_df) == 0:
            # Return empty predictions
            return SoVPredictionOutput(
                predictions_df=pd.DataFrame(columns=[
                    "address", "sov_probability", "predicted_class"
                ]),
                prediction_count=0,
                metadata={"inference_timestamp": datetime.now(timezone.utc).isoformat()},
            )
        
        # Prepare feature matrix
        X = features_df[REQUIRED_FEATURES].values
        
        # Get predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        # Ensure probabilities are in valid range [0.0, 1.0]
        y_proba = np.clip(y_proba, 0.0, 1.0)
        
        # Build predictions DataFrame
        predictions_df = pd.DataFrame({
            "address": features_df["address"].values,
            "sov_probability": y_proba,
            "predicted_class": y_pred.astype(bool),
        })
        
        # Build metadata
        metadata = {
            "inference_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_predictions": len(predictions_df),
            "predicted_sov_count": int(y_pred.sum()),
            "predicted_sov_ratio": float(y_pred.mean()),
            "avg_probability": float(y_proba.mean()),
        }
        
        logger.info(
            f"SoV prediction inference complete: {len(predictions_df)} predictions",
            extra={
                "prediction_count": len(predictions_df),
                "predicted_sov_ratio": float(y_pred.mean()),
            }
        )
        
        return SoVPredictionOutput(
            predictions_df=predictions_df,
            prediction_count=len(predictions_df),
            metadata=metadata,
        )
        
    except Exception as e:
        logger.error(f"SoV prediction inference failed: {e}", exc_info=True)
        raise
