"""Tests for configuration management."""

import json
import pytest

from pydantic import ValidationError
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


# ============================================================================
# Security Configuration Tests (Task 13.3)
# Requirements: 1.9, 2.3, 6.5
# ============================================================================


class TestSecurityConfigDefaults:
    """Test default values for security configuration."""

    def test_security_config_has_defaults(self, config_file):
        """Test that security config has sensible defaults when not specified."""
        manager = ConfigurationManager(config_path=config_file)
        config = manager.load_config()

        # Verify security config exists with defaults
        assert config.security is not None

        # Credential sanitizer defaults
        sanitizer = config.security.credential_sanitizer
        assert "apikey" in sanitizer.sensitive_param_names
        assert "Authorization" in sanitizer.sensitive_header_names
        assert sanitizer.redaction_placeholder == "[REDACTED]"

        # SSRF protection defaults
        ssrf = config.security.ssrf_protection
        assert "api.etherscan.io" in ssrf.allowed_domains
        assert ssrf.require_https is True
        assert ssrf.block_private_ips is True

        # Resource limits defaults
        limits = config.security.resource_limits
        assert limits.max_response_size_bytes == 10 * 1024 * 1024  # 10MB
        assert limits.max_output_file_size_bytes == 100 * 1024 * 1024  # 100MB
        assert limits.max_memory_usage_mb == 512

        # Timeout defaults
        timeouts = config.security.timeouts
        assert timeouts.overall_run_timeout_seconds == 1800  # 30 min
        assert timeouts.per_collection_timeout_seconds == 180  # 3 min
        assert timeouts.shutdown_timeout_seconds == 30


class TestSecurityConfigEnvOverrides:
    """Test environment variable overrides for security configuration."""

    def test_allowed_domains_env_override(self, config_file, monkeypatch):
        """Test ALLOWED_EXPLORER_DOMAINS environment variable override."""
        monkeypatch.setenv(
            "ALLOWED_EXPLORER_DOMAINS",
            "api.etherscan.io,api.bscscan.com,custom.domain.com"
        )

        manager = ConfigurationManager(config_path=config_file)
        config = manager.load_config()

        assert config.security.ssrf_protection.allowed_domains == [
            "api.etherscan.io",
            "api.bscscan.com",
            "custom.domain.com"
        ]

    def test_max_response_size_env_override(self, config_file, monkeypatch):
        """Test MAX_RESPONSE_SIZE_BYTES environment variable override."""
        monkeypatch.setenv("MAX_RESPONSE_SIZE_BYTES", "5242880")  # 5MB

        manager = ConfigurationManager(config_path=config_file)
        config = manager.load_config()

        assert config.security.resource_limits.max_response_size_bytes == 5242880

    def test_overall_timeout_env_override(self, config_file, monkeypatch):
        """Test OVERALL_RUN_TIMEOUT_SECONDS environment variable override."""
        monkeypatch.setenv("OVERALL_RUN_TIMEOUT_SECONDS", "3600")  # 1 hour

        manager = ConfigurationManager(config_path=config_file)
        config = manager.load_config()

        assert config.security.timeouts.overall_run_timeout_seconds == 3600

    def test_per_collection_timeout_env_override(self, config_file, monkeypatch):
        """Test PER_COLLECTION_TIMEOUT_SECONDS environment variable override."""
        monkeypatch.setenv("PER_COLLECTION_TIMEOUT_SECONDS", "300")  # 5 min

        manager = ConfigurationManager(config_path=config_file)
        config = manager.load_config()

        assert config.security.timeouts.per_collection_timeout_seconds == 300

    def test_shutdown_timeout_env_override(self, config_file, monkeypatch):
        """Test SHUTDOWN_TIMEOUT_SECONDS environment variable override."""
        monkeypatch.setenv("SHUTDOWN_TIMEOUT_SECONDS", "60")

        manager = ConfigurationManager(config_path=config_file)
        config = manager.load_config()

        assert config.security.timeouts.shutdown_timeout_seconds == 60

    def test_ssrf_require_https_env_override(self, config_file, monkeypatch):
        """Test SSRF_REQUIRE_HTTPS environment variable override."""
        monkeypatch.setenv("SSRF_REQUIRE_HTTPS", "false")

        manager = ConfigurationManager(config_path=config_file)
        config = manager.load_config()

        assert config.security.ssrf_protection.require_https is False

    def test_ssrf_block_private_ips_env_override(self, config_file, monkeypatch):
        """Test SSRF_BLOCK_PRIVATE_IPS environment variable override."""
        monkeypatch.setenv("SSRF_BLOCK_PRIVATE_IPS", "false")

        manager = ConfigurationManager(config_path=config_file)
        config = manager.load_config()

        assert config.security.ssrf_protection.block_private_ips is False

    def test_max_memory_env_override(self, config_file, monkeypatch):
        """Test MAX_MEMORY_USAGE_MB environment variable override."""
        monkeypatch.setenv("MAX_MEMORY_USAGE_MB", "1024")

        manager = ConfigurationManager(config_path=config_file)
        config = manager.load_config()

        assert config.security.resource_limits.max_memory_usage_mb == 1024

    def test_redaction_placeholder_env_override(self, config_file, monkeypatch):
        """Test CREDENTIAL_REDACTION_PLACEHOLDER environment variable override."""
        monkeypatch.setenv("CREDENTIAL_REDACTION_PLACEHOLDER", "***HIDDEN***")

        manager = ConfigurationManager(config_path=config_file)
        config = manager.load_config()

        assert (
            config.security.credential_sanitizer.redaction_placeholder
            == "***HIDDEN***"
        )


class TestSecurityConfigValidationErrors:
    """Test validation errors for security configuration."""

    def test_empty_allowed_domains_env_raises_error(
        self, config_file, monkeypatch
    ):
        """Test that empty ALLOWED_EXPLORER_DOMAINS raises ValueError."""
        monkeypatch.setenv("ALLOWED_EXPLORER_DOMAINS", "   ,  ,  ")

        manager = ConfigurationManager(config_path=config_file)
        with pytest.raises(ValueError) as exc_info:
            manager.load_config()

        assert "ALLOWED_EXPLORER_DOMAINS cannot be empty" in str(exc_info.value)

    def test_invalid_max_response_size_raises_error(
        self, config_file, monkeypatch
    ):
        """Test that non-integer MAX_RESPONSE_SIZE_BYTES raises ValueError."""
        monkeypatch.setenv("MAX_RESPONSE_SIZE_BYTES", "not_a_number")

        manager = ConfigurationManager(config_path=config_file)
        with pytest.raises(ValueError) as exc_info:
            manager.load_config()

        assert "Invalid MAX_RESPONSE_SIZE_BYTES" in str(exc_info.value)

    def test_invalid_overall_timeout_raises_error(
        self, config_file, monkeypatch
    ):
        """Test that non-integer OVERALL_RUN_TIMEOUT_SECONDS raises ValueError."""
        monkeypatch.setenv("OVERALL_RUN_TIMEOUT_SECONDS", "thirty_minutes")

        manager = ConfigurationManager(config_path=config_file)
        with pytest.raises(ValueError) as exc_info:
            manager.load_config()

        assert "Invalid OVERALL_RUN_TIMEOUT_SECONDS" in str(exc_info.value)

    def test_invalid_per_collection_timeout_raises_error(
        self, config_file, monkeypatch
    ):
        """Test non-integer PER_COLLECTION_TIMEOUT_SECONDS raises ValueError."""
        monkeypatch.setenv("PER_COLLECTION_TIMEOUT_SECONDS", "abc")

        manager = ConfigurationManager(config_path=config_file)
        with pytest.raises(ValueError) as exc_info:
            manager.load_config()

        assert "Invalid PER_COLLECTION_TIMEOUT_SECONDS" in str(exc_info.value)

    def test_invalid_shutdown_timeout_raises_error(
        self, config_file, monkeypatch
    ):
        """Test that non-integer SHUTDOWN_TIMEOUT_SECONDS raises ValueError."""
        monkeypatch.setenv("SHUTDOWN_TIMEOUT_SECONDS", "xyz")

        manager = ConfigurationManager(config_path=config_file)
        with pytest.raises(ValueError) as exc_info:
            manager.load_config()

        assert "Invalid SHUTDOWN_TIMEOUT_SECONDS" in str(exc_info.value)

    def test_invalid_max_memory_raises_error(self, config_file, monkeypatch):
        """Test that non-integer MAX_MEMORY_USAGE_MB raises ValueError."""
        monkeypatch.setenv("MAX_MEMORY_USAGE_MB", "lots")

        manager = ConfigurationManager(config_path=config_file)
        with pytest.raises(ValueError) as exc_info:
            manager.load_config()

        assert "Invalid MAX_MEMORY_USAGE_MB" in str(exc_info.value)

    def test_timeout_relationship_validation(self, tmp_path, base_config_data):
        """Test that invalid timeout relationships raise ValidationError."""
        # Set per_collection > overall (invalid)
        base_config_data["security"] = {
            "timeouts": {
                "overall_run_timeout_seconds": 100,
                "per_collection_timeout_seconds": 200,  # > overall
                "shutdown_timeout_seconds": 10
            }
        }

        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(base_config_data))

        manager = ConfigurationManager(config_path=str(config_path))
        with pytest.raises(ValidationError):
            manager.load_config()

    def test_invalid_wildcard_domain_pattern(self, tmp_path, base_config_data):
        """Test that invalid wildcard domain patterns raise ValidationError."""
        base_config_data["security"] = {
            "ssrf_protection": {
                "allowed_domains": ["invalid*pattern.com"]  # Invalid wildcard
            }
        }

        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(base_config_data))

        manager = ConfigurationManager(config_path=str(config_path))
        with pytest.raises(ValidationError):
            manager.load_config()

    def test_empty_allowed_domains_in_json_raises_error(
        self, tmp_path, base_config_data
    ):
        """Test that empty allowed_domains in JSON raises ValidationError."""
        base_config_data["security"] = {
            "ssrf_protection": {
                "allowed_domains": []  # Empty list
            }
        }

        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(base_config_data))

        manager = ConfigurationManager(config_path=str(config_path))
        with pytest.raises(ValidationError):
            manager.load_config()


class TestSecurityConfigFromJson:
    """Test loading security configuration from JSON file."""

    def test_security_config_from_json(self, tmp_path, base_config_data):
        """Test loading complete security config from JSON."""
        base_config_data["security"] = {
            "credential_sanitizer": {
                "sensitive_param_names": ["custom_key", "secret_token"],
                "redaction_placeholder": "***"
            },
            "ssrf_protection": {
                "allowed_domains": ["api.custom.com", "*.custom.io"],
                "require_https": True,
                "block_private_ips": True
            },
            "resource_limits": {
                "max_response_size_bytes": 5000000,
                "max_memory_usage_mb": 256
            },
            "timeouts": {
                "overall_run_timeout_seconds": 900,
                "per_collection_timeout_seconds": 120,
                "shutdown_timeout_seconds": 15
            }
        }

        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(base_config_data))

        manager = ConfigurationManager(config_path=str(config_path))
        config = manager.load_config()

        # Verify custom values
        assert config.security.credential_sanitizer.sensitive_param_names == [
            "custom_key", "secret_token"
        ]
        assert config.security.credential_sanitizer.redaction_placeholder == "***"
        assert config.security.ssrf_protection.allowed_domains == [
            "api.custom.com", "*.custom.io"
        ]
        assert config.security.resource_limits.max_response_size_bytes == 5000000
        assert config.security.resource_limits.max_memory_usage_mb == 256
        assert config.security.timeouts.overall_run_timeout_seconds == 900
        assert config.security.timeouts.per_collection_timeout_seconds == 120
        assert config.security.timeouts.shutdown_timeout_seconds == 15

    def test_env_overrides_json_security_config(
        self, tmp_path, base_config_data, monkeypatch
    ):
        """Test that env vars override JSON security config values."""
        # Set values in JSON
        base_config_data["security"] = {
            "resource_limits": {
                "max_response_size_bytes": 5000000
            },
            "timeouts": {
                "overall_run_timeout_seconds": 900,
                "per_collection_timeout_seconds": 120,
                "shutdown_timeout_seconds": 15
            }
        }

        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(base_config_data))

        # Override with env vars
        monkeypatch.setenv("MAX_RESPONSE_SIZE_BYTES", "8000000")
        monkeypatch.setenv("OVERALL_RUN_TIMEOUT_SECONDS", "1200")

        manager = ConfigurationManager(config_path=str(config_path))
        config = manager.load_config()

        # Env vars should override JSON
        assert config.security.resource_limits.max_response_size_bytes == 8000000
        assert config.security.timeouts.overall_run_timeout_seconds == 1200
        # Non-overridden values should remain from JSON
        assert config.security.timeouts.per_collection_timeout_seconds == 120

        assert config.security.timeouts.shutdown_timeout_seconds == 15
