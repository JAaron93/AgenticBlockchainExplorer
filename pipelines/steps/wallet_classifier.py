"""ZenML steps for wallet behavior classification.

This module implements wallet behavior classification for categorizing
wallets into behavioral patterns: trader, holder, whale, retail.

Requirements: 12.1, 12.3, 12.4, 12.5
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from zenml import step

logger = logging.getLogger(__name__)


# =============================================================================
# Wallet Behavior Classes (Taxonomy v1.0)
# =============================================================================

class WalletBehaviorClass(str, Enum):
    """Wallet behavior classification categories.
    
    Taxonomy v1.0 - Mutually exclusive classes:
    - TRADER: High frequency, low holding period
    - HOLDER: Low frequency, high holding period
    - WHALE: High balance (top 1%), regardless of activity pattern
    - RETAIL: All wallets not matching above criteria
    
    Requirements: 12.1
    """
    TRADER = "trader"
    HOLDER = "holder"
    WHALE = "whale"
    RETAIL = "retail"
    
    @classmethod
    def all_classes(cls) -> List[str]:
        """Return all class values as strings."""
        return [c.value for c in cls]


# =============================================================================
# Classification Thresholds (Taxonomy v1.0)
# =============================================================================

@dataclass
class ClassificationThresholds:
    """Thresholds for wallet behavior classification.
    
    Based on Taxonomy v1.0 from Requirements 12.1:
    - trader: transaction_frequency > 1.0 tx/day AND holding_period_days < 7
    - holder: transaction_frequency < 0.1 tx/day AND holding_period_days > 30
    - whale: balance_percentile >= 99 (top 1% by balance)
    - retail: All wallets not matching above criteria
    
    Attributes:
        trader_frequency_threshold: Min tx/day to be trader (default: 1.0)
        trader_holding_period_max: Max holding days for trader (default: 7)
        holder_frequency_threshold: Max tx/day to be holder (default: 0.1)
        holder_holding_period_min: Min holding days for holder (default: 30)
        whale_percentile_threshold: Min balance percentile for whale (default: 99)
        taxonomy_version: Version of the classification taxonomy
    """
    trader_frequency_threshold: float = 1.0
    trader_holding_period_max: int = 7
    holder_frequency_threshold: float = 0.1
    holder_holding_period_min: int = 30
    whale_percentile_threshold: float = 99.0
    taxonomy_version: str = "1.0"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "trader_frequency_threshold": self.trader_frequency_threshold,
            "trader_holding_period_max": self.trader_holding_period_max,
            "holder_frequency_threshold": self.holder_frequency_threshold,
            "holder_holding_period_min": self.holder_holding_period_min,
            "whale_percentile_threshold": self.whale_percentile_threshold,
            "taxonomy_version": self.taxonomy_version,
        }


# Default thresholds based on Requirements 12.1
DEFAULT_THRESHOLDS = ClassificationThresholds()


# =============================================================================
# Data Classes for Classifier Outputs
# =============================================================================

@dataclass
class WalletClassifierOutput:
    """Output from wallet classifier training step.
    
    Attributes:
        model: Trained classifier model
        metrics: Evaluation metrics (precision, recall, F1 per class, overall accuracy)
        hyperparameters: Model hyperparameters used
        feature_importances: Feature importance scores
        class_distribution: Distribution of classes in training data
        thresholds: Classification thresholds used for labeling
        training_metadata: Metadata about training run
    """
    model: Any
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    feature_importances: Dict[str, float]
    class_distribution: Dict[str, int]
    thresholds: ClassificationThresholds
    training_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization (excluding model)."""
        return {
            "metrics": self.metrics,
            "hyperparameters": self.hyperparameters,
            "feature_importances": self.feature_importances,
            "class_distribution": self.class_distribution,
            "thresholds": self.thresholds.to_dict(),
            "training_metadata": self.training_metadata,
        }


@dataclass
class WalletClassificationOutput:
    """Output from wallet classification inference step.
    
    Attributes:
        classifications_df: DataFrame with classifications for each wallet
        classification_count: Number of wallets classified
        class_distribution: Distribution of predicted classes
        low_confidence_count: Number of predictions with confidence < 0.6
        metadata: Additional metadata about inference run
    """
    classifications_df: pd.DataFrame
    classification_count: int
    class_distribution: Dict[str, int]
    low_confidence_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "classification_count": self.classification_count,
            "class_distribution": self.class_distribution,
            "low_confidence_count": self.low_confidence_count,
            "metadata": self.metadata,
        }


# =============================================================================
# Labeling Functions
# =============================================================================

def label_wallet_behavior(
    features_df: pd.DataFrame,
    thresholds: Optional[ClassificationThresholds] = None,
) -> pd.Series:
    """Assign wallet behavior labels based on feature thresholds.
    
    Classification rules (Taxonomy v1.0, Requirements 12.1):
    1. WHALE: balance_percentile >= 99 (checked first, overrides other patterns)
    2. TRADER: transaction_frequency > 1.0 AND holding_period_days < 7
    3. HOLDER: transaction_frequency < 0.1 AND holding_period_days > 30
    4. RETAIL: All wallets not matching above criteria
    
    Args:
        features_df: DataFrame with extracted features including:
            - transaction_frequency
            - holding_period_days
            - balance_percentile
        thresholds: Classification thresholds (default: DEFAULT_THRESHOLDS)
        
    Returns:
        Series with wallet behavior class labels
        
    Requirements: 12.1
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    
    n = len(features_df)
    labels = pd.Series([WalletBehaviorClass.RETAIL.value] * n, index=features_df.index)
    
    if n == 0:
        return labels
    
    # Get required features with defaults
    tx_freq = features_df.get("transaction_frequency", pd.Series([0.0] * n))
    hold_days = features_df.get("holding_period_days", pd.Series([0.0] * n))
    balance_pct = features_df.get("balance_percentile", pd.Series([50.0] * n))
    
    # Apply classification rules in order of priority
    # 1. WHALE: top 1% by balance (highest priority)
    whale_mask = balance_pct >= thresholds.whale_percentile_threshold
    labels[whale_mask] = WalletBehaviorClass.WHALE.value
    
    # 2. TRADER: high frequency, low holding period (only if not whale)
    trader_mask = (
        (tx_freq > thresholds.trader_frequency_threshold) &
        (hold_days < thresholds.trader_holding_period_max) &
        ~whale_mask
    )
    labels[trader_mask] = WalletBehaviorClass.TRADER.value
    
    # 3. HOLDER: low frequency, high holding period (only if not whale or trader)
    holder_mask = (
        (tx_freq < thresholds.holder_frequency_threshold) &
        (hold_days > thresholds.holder_holding_period_min) &
        ~whale_mask &
        ~trader_mask
    )
    labels[holder_mask] = WalletBehaviorClass.HOLDER.value
    
    # 4. RETAIL: everything else (already set as default)
    
    return labels


def get_class_weights(labels: pd.Series) -> Dict[int, float]:
    """Calculate class weights inversely proportional to class frequency.
    
    Args:
        labels: Series of class labels (encoded as integers)
        
    Returns:
        Dictionary mapping class index to weight
        
    Requirements: 12.3
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    weights = {}
    for cls, count in zip(unique, counts):
        # Weight inversely proportional to frequency
        weights[int(cls)] = total / (len(unique) * count)
    
    return weights


# =============================================================================
# ZenML Steps
# =============================================================================

@step
def train_wallet_classifier_step(
    features_df: pd.DataFrame,
    thresholds: Optional[Dict[str, Any]] = None,
    algorithm: str = "xgboost",
    n_estimators: int = 100,
    max_depth: int = 10,
    learning_rate: float = 0.1,
    test_size: float = 0.2,
    random_state: int = 42,
) -> WalletClassifierOutput:
    """ZenML step to train multi-class classifier for wallet behavior.
    
    Trains a model to classify wallets into behavior categories:
    - trader: High frequency, low holding period
    - holder: Low frequency, high holding period
    - whale: High balance (top 1%)
    - retail: All others
    
    Args:
        features_df: DataFrame with extracted features
        thresholds: Classification thresholds dict (optional)
        algorithm: Model algorithm - "xgboost" or "random_forest"
        n_estimators: Number of trees in the ensemble
        max_depth: Maximum depth of trees (constrained to <= 10)
        learning_rate: Learning rate for XGBoost
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        WalletClassifierOutput with trained model, metrics, and metadata
        
    Requirements: 12.3
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        accuracy_score,
    )
    from sklearn.preprocessing import LabelEncoder
    
    logger.info(
        f"Starting wallet classifier training with {len(features_df)} samples, "
        f"algorithm={algorithm}"
    )
    
    try:
        # Parse thresholds
        if thresholds is not None:
            thresh = ClassificationThresholds(**thresholds)
        else:
            thresh = DEFAULT_THRESHOLDS
        
        # Constrain max_depth per Requirements 11.2 (same constraints apply)
        max_depth = min(max_depth, 10)
        
        # Required features for classification
        required_features = [
            "transaction_count",
            "avg_transaction_size",
            "balance_percentile",
            "holding_period_days",
            "activity_recency_days",
            "transaction_frequency",
            "balance_volatility",
            "cross_chain_flag",
        ]
        
        # Verify required features exist
        missing = [f for f in required_features if f not in features_df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        # Generate labels using threshold-based labeling
        labels = label_wallet_behavior(features_df, thresh)
        
        # Encode labels to integers
        label_encoder = LabelEncoder()
        label_encoder.fit(WalletBehaviorClass.all_classes())
        y = label_encoder.transform(labels)
        
        # Prepare feature matrix
        X = features_df[required_features].values
        
        # Calculate class distribution
        class_distribution = {
            cls: int((labels == cls).sum())
            for cls in WalletBehaviorClass.all_classes()
        }
        
        logger.info(f"Class distribution: {class_distribution}")
        
        # Check if we have enough samples per class for stratified split
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class_count = class_counts.min() if len(class_counts) > 0 else 0
        min_samples_for_stratify = max(2, int(np.ceil(1 / test_size)))
        
        use_stratify = (
            len(unique_classes) > 1 and
            min_class_count >= min_samples_for_stratify
        )
        
        if use_stratify:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
                logger.debug("Using stratified train/test split")
            except ValueError as e:
                logger.warning(f"Stratified split failed ({e}), using random split")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
        else:
            logger.warning(
                f"Cannot use stratified split: {len(unique_classes)} classes, "
                f"min class count={min_class_count}. Using random split."
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        # Calculate class weights for imbalanced data
        class_weights = get_class_weights(pd.Series(y_train))
        
        # Train model
        if algorithm == "xgboost":
            import xgboost as xgb
            
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                use_label_encoder=False,
                eval_metric="mlogloss",
                objective="multi:softprob",
                num_class=len(WalletBehaviorClass.all_classes()),
            )
            # XGBoost uses sample_weight instead of class_weight
            sample_weights = np.array([class_weights.get(yi, 1.0) for yi in y_train])
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:  # random_forest
            from sklearn.ensemble import RandomForestClassifier
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight="balanced",
                random_state=random_state,
            )
            model.fit(X_train, y_train)
        
        # Store label encoder with model for inference
        model._label_encoder = label_encoder
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_macro": float(precision_score(
                y_test, y_pred, average="macro", zero_division=0
            )),
            "recall_macro": float(recall_score(
                y_test, y_pred, average="macro", zero_division=0
            )),
            "f1_macro": float(f1_score(
                y_test, y_pred, average="macro", zero_division=0
            )),
        }
        
        # Per-class metrics
        for i, cls in enumerate(label_encoder.classes_):
            y_test_binary = (y_test == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            
            metrics[f"precision_{cls}"] = float(precision_score(
                y_test_binary, y_pred_binary, zero_division=0
            ))
            metrics[f"recall_{cls}"] = float(recall_score(
                y_test_binary, y_pred_binary, zero_division=0
            ))
            metrics[f"f1_{cls}"] = float(f1_score(
                y_test_binary, y_pred_binary, zero_division=0
            ))
        
        # Get feature importances
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            importances = np.zeros(len(required_features))
        
        feature_importances = {
            feat: float(imp)
            for feat, imp in zip(required_features, importances)
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
            "total_samples": len(features_df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "num_classes": len(unique_classes),
            "taxonomy_version": thresh.taxonomy_version,
        }
        
        logger.info(
            f"Wallet classifier training complete",
            extra={
                "metrics": metrics,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }
        )
        
        return WalletClassifierOutput(
            model=model,
            metrics=metrics,
            hyperparameters=hyperparameters,
            feature_importances=feature_importances,
            class_distribution=class_distribution,
            thresholds=thresh,
            training_metadata=training_metadata,
        )
        
    except Exception as e:
        logger.error(f"Wallet classifier training failed: {e}", exc_info=True)
        raise


@step
def classify_wallets_step(
    features_df: pd.DataFrame,
    model: Any,
    low_confidence_threshold: float = 0.6,
) -> WalletClassificationOutput:
    """ZenML step to run wallet behavior classification inference.
    
    Classifies each wallet into exactly one behavior class with confidence.
    
    Args:
        features_df: DataFrame with extracted features
        model: Trained wallet classifier model
        low_confidence_threshold: Threshold below which predictions are flagged
        
    Returns:
        WalletClassificationOutput with classifications DataFrame containing:
        - address: Wallet address
        - behavior_class: Predicted class (trader/holder/whale/retail)
        - confidence: Prediction confidence (max probability)
        - class_probabilities: Dict of probabilities per class
        
    Requirements: 12.4
    """
    logger.info(f"Starting wallet classification with {len(features_df)} wallets")
    
    try:
        if len(features_df) == 0:
            return WalletClassificationOutput(
                classifications_df=pd.DataFrame(columns=[
                    "address", "behavior_class", "confidence"
                ]),
                classification_count=0,
                class_distribution={cls: 0 for cls in WalletBehaviorClass.all_classes()},
                low_confidence_count=0,
                metadata={"inference_timestamp": datetime.now(timezone.utc).isoformat()},
            )
        
        # Required features
        required_features = [
            "transaction_count",
            "avg_transaction_size",
            "balance_percentile",
            "holding_period_days",
            "activity_recency_days",
            "transaction_frequency",
            "balance_volatility",
            "cross_chain_flag",
        ]
        
        # Prepare feature matrix
        X = features_df[required_features].values
        
        # Get predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        # Get label encoder from model
        label_encoder = getattr(model, "_label_encoder", None)
        if label_encoder is None:
            # Fallback: create encoder with default classes
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            label_encoder.fit(WalletBehaviorClass.all_classes())
        
        # Decode predictions to class names
        predicted_classes = label_encoder.inverse_transform(y_pred)
        
        # Calculate confidence (max probability)
        confidence = np.max(y_proba, axis=1)
        
        # Build classifications DataFrame
        classifications_df = pd.DataFrame({
            "address": features_df["address"].values,
            "behavior_class": predicted_classes,
            "confidence": confidence,
        })
        
        # Get the classes the model was trained on
        # model.classes_ contains the actual class indices the model knows
        model_classes = getattr(model, "classes_", None)
        if model_classes is None:
            model_classes = np.arange(len(label_encoder.classes_))
        
        # Add class probabilities as separate columns
        # Handle case where model was trained on fewer classes than expected
        all_classes = WalletBehaviorClass.all_classes()
        for cls in all_classes:
            # Find if this class was in training data
            try:
                cls_idx = list(label_encoder.classes_).index(cls)
                # Check if this class index is in model's known classes
                if cls_idx < y_proba.shape[1]:
                    classifications_df[f"prob_{cls}"] = y_proba[:, cls_idx]
                else:
                    # Class not in model's training data
                    classifications_df[f"prob_{cls}"] = 0.0
            except (ValueError, IndexError):
                # Class not in label encoder
                classifications_df[f"prob_{cls}"] = 0.0
        
        # Calculate class distribution
        class_distribution = {
            cls: int((predicted_classes == cls).sum())
            for cls in WalletBehaviorClass.all_classes()
        }
        
        # Count low confidence predictions
        low_confidence_count = int((confidence < low_confidence_threshold).sum())
        
        # Build metadata
        metadata = {
            "inference_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_classifications": len(classifications_df),
            "low_confidence_threshold": low_confidence_threshold,
            "avg_confidence": float(confidence.mean()),
            "min_confidence": float(confidence.min()),
            "max_confidence": float(confidence.max()),
        }
        
        logger.info(
            f"Wallet classification complete: {len(classifications_df)} wallets",
            extra={
                "classification_count": len(classifications_df),
                "class_distribution": class_distribution,
                "low_confidence_count": low_confidence_count,
            }
        )
        
        return WalletClassificationOutput(
            classifications_df=classifications_df,
            classification_count=len(classifications_df),
            class_distribution=class_distribution,
            low_confidence_count=low_confidence_count,
            metadata=metadata,
        )
        
    except Exception as e:
        logger.error(f"Wallet classification failed: {e}", exc_info=True)
        raise
