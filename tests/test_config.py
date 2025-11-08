"""Tests for configuration management."""

import json
import pytest

from config import ConfigurationManager, Config


@pytest.fixture
def base_config_data():
    """Base configuration data for tests."""
    return {
        "explorers": [
            {
                "name": "etherscan",
                "base_url": "https://api.etherscan.io/api",
                "api_key": "test_key",
                "type": "api",
                "chain": "ethereum"
            },
            {
                "name": "bscscan",
                "base_url": "https://api.bscscan.com/api",
                "api_key": "test_key",
                "type": "api",
                "chain": "bsc"
            },
            {
                "name": "polygonscan",
                "base_url": "https://api.polygonscan.com/api",
                "api_key": "test_key",
                "type": "api",
                "chain": "polygon"
            }
        ],
        "stablecoins": {
            "USDC": {
                "ethereum": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                "bsc": "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d",
                "polygon": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
            },
            "USDT": {
                "ethereum": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
                "bsc": "0x55d398326f99059fF775485246999027B3197955",
                "polygon": "0xc2132D05D31c914a87C6611C10748AEb04B58e8F"
            }
        },
        "auth0": {
            "domain": "test.auth0.com",
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "audience": "https://test-api",
            "callback_url": "http://localhost:8000/callback",
            "logout_url": "http://localhost:8000"
        },
        "database": {
            "url": "postgresql://user:pass@localhost:5432/test",
            "pool_size": 10,
            "max_overflow": 20
        },
        "app": {
            "env": "development",
            "host": "0.0.0.0",
            "port": 8000,
            "debug": True,
            "secret_key": "test_secret_key_at_least_32_chars_long"
        }
    }


@pytest.fixture
def config_file(tmp_path, base_config_data):
    """Create a temporary config file with base configuration data."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(base_config_data))
    return str(config_path)


def test_load_config_from_json(config_file):
    """Test loading configuration from JSON file."""
    # Load configuration
    manager = ConfigurationManager(config_path=config_file)
    config = manager.load_config()

    # Verify configuration
    assert isinstance(config, Config)
    assert len(config.explorers) == 3
    assert config.explorers[0].name == "etherscan"
    assert config.explorers[0].chain == "ethereum"
    assert "USDC" in config.stablecoins
    assert "USDT" in config.stablecoins
    assert config.auth0.domain == "test.auth0.com"
    assert config.database.url == "postgresql://user:pass@localhost:5432/test"
    assert config.app.env == "development"


def test_config_validation(config_file):
    """Test configuration validation."""
    manager = ConfigurationManager(config_path=config_file)
    config = manager.load_config()

    # Validate configuration
    assert manager.validate_config(config) is True


def test_get_explorer_by_name(config_file):
    """Test getting explorer configuration by name."""
    manager = ConfigurationManager(config_path=config_file)
    manager.load_config()

    # Get explorer by name
    etherscan = manager.get_explorer_by_name("etherscan")
    assert etherscan is not None
    assert etherscan.name == "etherscan"
    assert etherscan.chain == "ethereum"

    bscscan = manager.get_explorer_by_name("bscscan")
    assert bscscan is not None
    assert bscscan.name == "bscscan"
    assert bscscan.chain == "bsc"

    # Non-existent explorer
    unknown = manager.get_explorer_by_name("unknown")
    assert unknown is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
