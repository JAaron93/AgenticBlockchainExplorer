"""Centralized service for stablecoin data analysis and classification.

Unifies heuristic-based and ML-based classification to ensure consistent
metrics across the entire application (API, CLI, and ZenML).
"""

import logging
from typing import List, Optional
import pandas as pd

from collectors.models import Transaction, Holder, ActivityType
from collectors.classifier import ActivityClassifier

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for orchestrating data analysis and classification."""

    def __init__(self, classifier: Optional[ActivityClassifier] = None):
        """Initialize the analysis service.

        Args:
            classifier: Optional ActivityClassifier instance to use.
        """
        self._classifier = classifier or ActivityClassifier()

    def classify_transactions(self, transactions: List[Transaction]) -> List[Transaction]:
        """Classify a list of transactions.

        Args:
            transactions: List of Transaction objects to classify.

        Returns:
            List of classified Transaction objects.
        """
        for tx in transactions:
            tx.activity_type = self._classifier.classify_transaction(tx)
        return transactions

    def classify_holders(
        self, 
        holders: List[Holder], 
        transactions: List[Transaction],
        use_ml_results: bool = False,
        sov_predictions: Optional[pd.DataFrame] = None
    ) -> List[Holder]:
        """Classify holders, including Store of Value (SoV) behavior.

        Args:
            holders: List of Holder objects to classify.
            transactions: List of transactions for behavior analysis.
            use_ml_results: Whether to prioritize ML results over heuristics.
            sov_predictions: DataFrame with ML-based SoV predictions.

        Returns:
            List of classified Holder objects.
        """
        # First apply based heuristic
        for holder in holders:
            # We track heuristic separately in the model if we wanted to refactor Holder,
            # but for now we'll just set the main flag and log the source.
            self._classifier.classify_holder(holder, transactions)
            holder.is_sov_heuristic = holder.is_store_of_value # New field if we add it

        # If ML results are available and requested, override heuristic
        if use_ml_results and sov_predictions is not None and not sov_predictions.empty:
            logger.info("Applying ML-based SoV predictions to holder data")
            # Map predictions to holders
            pred_map = sov_predictions.set_index("address")["prediction"].to_dict()
            
            for holder in holders:
                if holder.address in pred_map:
                    # Update the definitive flag
                    holder.is_store_of_value = bool(pred_map[holder.address])
                    
        return holders

    def aggregate_metrics(self, transactions: List[Transaction], holders: List[Holder]) -> dict:
        """Generate high-level metrics for the collected data.

        Args:
            transactions: List of classified transactions.
            holders: List of classified holders.

        Returns:
            Dictionary with aggregated metrics.
        """
        # This can eventually replace logic in DataAggregator or work in tandem
        return {
            "total_transactions": len(transactions),
            "total_holders": len(holders),
            "sov_count": sum(1 for h in holders if h.is_store_of_value),
            "activity_counts": self._get_activity_counts(transactions),
        }

    def _get_activity_counts(self, transactions: List[Transaction]) -> dict:
        """Count occurrences of each activity type."""
        counts = {activity.value: 0 for activity in ActivityType}
        for tx in transactions:
            counts[tx.activity_type.value] += 1
        return counts
