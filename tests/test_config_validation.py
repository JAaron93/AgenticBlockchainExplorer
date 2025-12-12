"""Tests for configuration validation logic."""

import json
import os
import tempfile
from typing import List, Optional

import pytest

from config import ConfigurationManager


# Explorer factory for creating per-chain explorer dicts
def make_explorer(chain: str) -> dict:
    """Create an explorer config dict for the given chain."""
    explorer_map = {
        "ethereum": {
            "name": "etherscan",
            "base_url": "https://api.etherscan.io/api",
            "api_key": "test_key",
            "type": "api",
            "chain": "ethereum"
        },
        "bsc": {
            "name": "bscscan",
            "base_url": "https://api.bscscan.com/api",
            "api_key": "test_key",
            "type": "api",
            "chain": "bsc"
        },
        "polygon": {
            "name": "polygonscan",
            "base_url": "https://api.polygonscan.com/api",
            "api_key": "test_key",
            "type": "api",
            "chain": "polygon"
        }
    }
    return explorer_map[chain]


@pytest.fixture
def base_config() -> dict:
    """Base configuration with common settings (no explorers)."""
    return {
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
            "url": "postgresql://user:pass@localhost:5432/test"
        },
        "app": {
            "secret_key": "test_secret_key_at_least_32_chars_long"
        }
    }


@pytest.fixture
def temp_config_file(base_config):
    """Fixture that writes config to temp JSON file and cleans up."""
    config_path = None

    def _create_config(explorers: List[dict]) -> str:
        nonlocal config_path
        config_data = {**base_config, "explorers": explorers}
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(config_data, f)
            config_path = f.name
        return config_path

    yield _create_config

    # Cleanup
    if config_path and os.path.exists(config_path):
        os.unlink(config_path)


@pytest.mark.parametrize(
    "chains,missing_chains,should_pass",
    [
        # All explorers present - should pass
        (
            ["ethereum", "bsc", "polygon"],
            [],
            True
        ),
        # Only ethereum - missing bsc and polygon
        (
            ["ethereum"],
            ["bsc", "polygon"],
            False
        ),
        # Missing only polygon
        (
            ["ethereum", "bsc"],
            ["polygon"],
            False
        ),
    ],
    ids=[
        "all_explorers_present",
        "missing_bsc_and_polygon",
        "missing_only_polygon",
    ]
)
def test_explorer_chain_validation(
    temp_config_file,
    chains: List[str],
    missing_chains: List[str],
    should_pass: bool
):
    """Test validation of explorer configurations for stablecoin chains."""
    explorers = [make_explorer(chain) for chain in chains]
    config_path = temp_config_file(explorers)

    manager = ConfigurationManager(config_path=config_path)
    config = manager.load_config()

    if should_pass:
        assert manager.validate_config(config) is True
    else:
        with pytest.raises(ValueError) as exc_info:
            manager.validate_config(config)

        error_message = str(exc_info.value)
        assert "Missing explorer configurations" in error_message

        # Check all missing chains are mentioned
        for chain in missing_chains:
            assert chain in error_message

        # Check present chains are not in the missing section
        missing_idx = error_message.find("Missing")
        stablecoins_idx = error_message.find("Stablecoins")
        if missing_idx != -1 and stablecoins_idx != -1:
            missing_section = error_message[missing_idx:stablecoins_idx]
            for chain in chains:
                assert chain not in missing_section


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
