"""Tests for configuration validation logic."""

import json
import os
import tempfile
import pytest

from config import ConfigurationManager


def test_missing_explorer_for_stablecoin_chain():
    """Test that validation fails when a stablecoin chain has no explorer."""
    config_data = {
        "explorers": [
            {
                "name": "etherscan",
                "base_url": "https://api.etherscan.io/api",
                "api_key": "test_key",
                "type": "api",
                "chain": "ethereum"
            }
            # Missing BSC and Polygon explorers
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
            "url": "postgresql://user:pass@localhost:5432/test"
        },
        "app": {
            "secret_key": "test_secret_key_at_least_32_chars_long"
        }
    }
    
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False
    ) as f:
        json.dump(config_data, f)
        config_path = f.name
    
    try:
        manager = ConfigurationManager(config_path=config_path)
        config = manager.load_config()
        
        # Validation should fail with clear error message
        with pytest.raises(ValueError) as exc_info:
            manager.validate_config(config)
        
        error_message = str(exc_info.value)
        assert "Missing explorer configurations" in error_message
        assert "bsc" in error_message
        assert "polygon" in error_message
        
    finally:
        os.unlink(config_path)


def test_all_explorers_present():
    """Test that validation passes when all required explorers are present."""
    config_data = {
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
            "url": "postgresql://user:pass@localhost:5432/test"
        },
        "app": {
            "secret_key": "test_secret_key_at_least_32_chars_long"
        }
    }
    
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False
    ) as f:
        json.dump(config_data, f)
        config_path = f.name
    
    try:
        manager = ConfigurationManager(config_path=config_path)
        config = manager.load_config()
        
        # Validation should pass
        assert manager.validate_config(config) is True
        
    finally:
        os.unlink(config_path)


def test_partial_missing_explorers():
    """Test validation with only one missing explorer."""
    config_data = {
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
            }
            # Missing Polygon explorer
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
            "url": "postgresql://user:pass@localhost:5432/test"
        },
        "app": {
            "secret_key": "test_secret_key_at_least_32_chars_long"
        }
    }
    
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False
    ) as f:
        json.dump(config_data, f)
        config_path = f.name
    
    try:
        manager = ConfigurationManager(config_path=config_path)
        config = manager.load_config()
        
        # Validation should fail
        with pytest.raises(ValueError) as exc_info:
            manager.validate_config(config)
        
        error_message = str(exc_info.value)
        assert "Missing explorer configurations" in error_message
        assert "polygon" in error_message
        # Should not mention ethereum or bsc since they have explorers
        assert "ethereum" not in error_message.split("Missing")[1].split("Stablecoins")[0]
        
    finally:
        os.unlink(config_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
