import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from core.exceptions import ModelPersistenceError
from pipelines.steps.ml import train_sov_predictor_step

def test_train_sov_predictor_step_raises_persistence_error_on_value_error():
    # Setup mock data
    features_df = pd.DataFrame({
        "address": ["0x1"],
        "transaction_count": [1],
        "avg_transaction_size": [1.0],
        "balance_percentile": [50.0],
        "holding_period_days": [10.0],
        "activity_recency_days": [5.0],
        "transaction_frequency": [0.1],
        "balance_volatility": [0.0],
        "cross_chain_flag": [0]
    })
    holders_df = pd.DataFrame({
        "address": ["0x1"],
        "is_store_of_value": [True]
    })
    
    # Mock ModelRegistry to raise ValueError in save_model
    with patch("pipelines.steps.ml.ModelRegistry") as MockRegistry:
        mock_instance = MockRegistry.return_value
        mock_instance.save_model.side_effect = ValueError("Invalid identifier")
        
        # The step should catch ValueError and raise ModelPersistenceError
        # Note: We need to mock sklearn/xgboost if they are not installed or if we want to skip training
        import numpy as np
        with patch("sklearn.model_selection.train_test_split") as mock_split:
            # X_train, X_test, y_train, y_test
            mock_split.return_value = (
                np.array([[0]]), np.array([[0]]), 
                np.array([1, 0]), np.array([1])
            )
            with patch("xgboost.XGBClassifier") as MockXGB:
                mock_model = MockXGB.return_value
                mock_model.predict.return_value = np.array([1])
                mock_model.predict_proba.return_value = np.array([[0, 1]])
                
                with pytest.raises(ModelPersistenceError, match="Failed to persist model 'sov_predictor'"):
                    train_sov_predictor_step.entrypoint(
                        features_df=features_df,
                        holders_df=holders_df,
                        algorithm="xgboost"
                    )

if __name__ == "__main__":
    pytest.main([__file__])
