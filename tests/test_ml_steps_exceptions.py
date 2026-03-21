import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from core.exceptions import ModelPersistenceError
from pipelines.steps.ml import train_sov_predictor_step

@pytest.mark.parametrize("side_effect, expected_match", [
    (ValueError("Invalid identifier"), "Failed to persist model 'sov_predictor'"),
    (IOError("Disk full"), "Failed to persist model 'sov_predictor'"),
    (Exception("Unexpected backup failure"), "Failed to persist model 'sov_predictor'"),
])
def test_train_sov_predictor_step_raises_persistence_error_on_various_errors(side_effect, expected_match):
    # Setup mock data
    import numpy as np
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
    
    # Mock ModelRegistry and training process
    with patch("pipelines.steps.ml.ModelRegistry") as MockRegistry, \
         patch("sklearn.model_selection.train_test_split") as mock_split, \
         patch("xgboost.XGBClassifier") as MockXGB:
        
        mock_instance = MockRegistry.return_value
        mock_instance.save_model.side_effect = side_effect
        
        # X_train, X_test, y_train, y_test
        mock_split.return_value = (
            np.array([[0]]), np.array([[0]]),
            np.array([1]), np.array([1])
        )
        
        mock_model = MockXGB.return_value
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0, 1]])
        
        with pytest.raises(ModelPersistenceError, match=expected_match):
            train_sov_predictor_step.entrypoint(
                features_df=features_df,
                holders_df=holders_df,
                algorithm="xgboost"
            )

if __name__ == "__main__":
    pytest.main([__file__])
