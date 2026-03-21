"""Configuration helper for the AgenticBlockchainExplorer.

Provides a singleton configuration instance for use across the application.
"""

import logging
import threading
from typing import Optional
from config.loader import ConfigurationManager
from config.models import Config

logger = logging.getLogger(__name__)

_config: Optional[Config] = None
_config_lock = threading.Lock()


def get_config(
    config_path: Optional[str] = None,
    env_file: Optional[str] = None,
) -> Config:
    """Get the application configuration (singleton).

    Args:
        config_path: Path to JSON configuration file.
        env_file: Path to .env file.

    Returns:
        The loaded Config object.
    """
    global _config
    if _config is None:
        with _config_lock:
            if _config is None:
                manager = ConfigurationManager(
                    config_path=config_path, env_file=env_file
                )
                _config = manager.load_config()
                logger.info("Loaded application configuration")
    else:
        # Warn if cached config exists but explicit paths were provided
        if config_path is not None or env_file is not None:
            logger.warning(
                "Returning cached configuration. Ignoring config_path=%s and env_file=%s",
                config_path,
                env_file,
            )
    return _config


def reset_config() -> None:
    """Reset the singleton configuration (for testing)."""
    global _config
    with _config_lock:
        _config = None
