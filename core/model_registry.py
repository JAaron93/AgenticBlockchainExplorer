"""Service for managing ML model persistence and retrieval.

Provides a simple registry for saving and loading trained models
to avoid redundant training cycles in production pipelines.
"""

import os
import joblib
import logging
from typing import Any, Optional, Dict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for persisting and retrieving ML models."""

    def __init__(self, registry_path: str = "models/registry"):
        """Initialize the model registry.

        Args:
            registry_path: Local path to store serialized models.
        """
        self.registry_path = registry_path
        os.makedirs(self.registry_path, exist_ok=True)

    def save_model(self, model_name: str, model: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Serialize and save a model to the registry.

        Args:
            model_name: Unique name for the model (e.g., 'sov_predictor').
            model: The trained model object (sklearn, xgboost, etc.).
            metadata: Optional dictionary of model metadata.

        Returns:
            Path to the saved model file.
        """
        import joblib
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.joblib"
        target_path = os.path.join(self.registry_path, filename)
        
        # Also save as 'latest' for easy retrieval
        latest_path = os.path.join(self.registry_path, f"{model_name}_latest.joblib")
        
        try:
            joblib.dump(model, target_path)
            joblib.dump(model, latest_path)
            
            # Save metadata if provided
            if metadata:
                meta_path = f"{target_path}.metadata.json"
                import json
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved model '{model_name}' to {target_path}")
            return target_path
        except Exception as e:
            logger.error(f"Failed to save model '{model_name}': {e}")
            raise

    def load_model(self, model_name: str, version: str = "latest") -> Optional[Any]:
        """Retrieve a model from the registry.

        Args:
            model_name: Name of the model to load.
            version: Specific version string or 'latest'.

        Returns:
            The loaded model object or None if not found.
        """
        import joblib
        
        if version == "latest":
            path = os.path.join(self.registry_path, f"{model_name}_latest.joblib")
        else:
            # Simple versioning based on filename matching if needed
            path = os.path.join(self.registry_path, f"{model_name}_{version}.joblib")

        if not os.path.exists(path):
            logger.debug(f"Model '{model_name}' (version: {version}) not found at {path}")
            return None

        try:
            model = joblib.load(path)
            logger.info(f"Loaded model '{model_name}' from {path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            return None
