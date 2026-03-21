"""Service for managing ML model persistence and retrieval.

Provides a simple registry for saving and loading trained models
to avoid redundant training cycles in production pipelines.
"""

import os
import joblib
import logging
import shutil
from typing import Any, Optional, Dict
from datetime import datetime, timezone

from .exceptions import ModelLoadingError

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

    def _sanitize_identifier(self, identifier: str) -> str:
        """Sanitize an identifier to prevent path traversal.

        Args:
            identifier: The identifier to sanitize (e.g., model name, version).

        Returns:
            The sanitized identifier.

        Raises:
            ValueError: If the identifier is empty or contains path separators.
        """
        if not identifier.strip():
            raise ValueError(
                f"Invalid identifier: '{identifier}' is empty or whitespace-only"
            )
        safe = os.path.basename(identifier)
        if not safe or safe != identifier:
            raise ValueError(
                f"Invalid identifier: '{identifier}' contains "
                "path separators or is empty"
            )
        return safe

    def save_model(
        self,
        model_name: str,
        model: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Serialize and save a model to the registry.

        Args:
            model_name: Unique name for the model (e.g., 'sov_predictor').
            model: The trained model object (sklearn, xgboost, etc.).
            metadata: Optional dictionary of model metadata.

        Returns:
            Path to the saved model file.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        # Sanitize model_name to prevent path traversal
        safe_name = self._sanitize_identifier(model_name)
        filename = f"{safe_name}_{timestamp}.joblib"
        target_path = os.path.join(self.registry_path, filename)
        
        # Also save as 'latest' for easy retrieval
        latest_path = os.path.join(
            self.registry_path, f"{safe_name}_latest.joblib"
        )
        
        try:
            # Serialize model once to target path
            joblib.dump(model, target_path)
            # Copy to latest path for easy retrieval
            shutil.copy2(target_path, latest_path)
            
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

    def load_model(
        self, model_name: str, version: str = "latest"
    ) -> Optional[Any]:
        """Retrieve a model from the registry.

        Args:
            model_name: Name of the model to load.
            version: Specific version string or 'latest'.

        Returns:
            The loaded model object or None if the model is not found in the registry.

        Raises:
            ModelLoadingError: If the model file is found but fails to load correctly.
            ValueError: If the model name or version identifier is invalid.
        """
        # Sanitize inputs
        safe_model_name = self._sanitize_identifier(model_name)
        if version != "latest":
            safe_version = self._sanitize_identifier(version)
        else:
            safe_version = version

        if version == "latest":
            path = os.path.join(
                self.registry_path, f"{safe_model_name}_latest.joblib"
            )
        else:
            # Simple versioning based on filename matching if needed
            path = os.path.join(
                self.registry_path, f"{safe_model_name}_{safe_version}.joblib"
            )

        if not os.path.exists(path):
            logger.debug(
                f"Model '{model_name}' (version: {version}) "
                f"not found at {path}"
            )
            return None

        try:
            model = joblib.load(path)
            logger.info(f"Loaded model '{model_name}' from {path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise ModelLoadingError(model_name, e) from e
