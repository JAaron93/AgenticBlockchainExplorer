"""Tests for security functionality."""

import pytest
from api.security import (
    CORSConfig,
    CSRFProtection,
    init_cors_config,
    init_csrf_protection,
    get_cors_config,
    get_csrf_protection,
    reset_cors_config,
    reset_csrf_protection,
)


@pytest.fixture(autouse=True)
def reset_security_globals():
    """Reset global security state before and after each test.

    This prevents cross-test pollution from tests that modify
    the global CORS and CSRF configuration.
    """
    # Reset before test
    reset_cors_config()
    reset_csrf_protection()

    yield

    # Reset after test
    reset_cors_config()
    reset_csrf_protection()


class TestCORSConfig:
    """Tests for CORS configuration."""

    def test_cors_config_allows_configured_origins(self):
        """Test that configured origins are allowed."""
        config = CORSConfig(
            allowed_origins=["http://localhost:3000", "https://example.com"]
        )

        assert config.is_origin_allowed("http://localhost:3000") is True
        assert config.is_origin_allowed("https://example.com") is True

    def test_cors_config_blocks_unconfigured_origins(self):
        """Test that unconfigured origins are blocked."""
        config = CORSConfig(allowed_origins=["http://localhost:3000"])

        assert config.is_origin_allowed("https://evil.com") is False
        assert config.is_origin_allowed("http://localhost:8000") is False

    def test_cors_config_handles_none_origin(self):
        """Test that None origin is blocked."""
        config = CORSConfig(allowed_origins=["http://localhost:3000"])

        assert config.is_origin_allowed(None) is False

    def test_cors_config_default_methods(self):
        """Test default allowed methods."""
        config = CORSConfig(allowed_origins=["http://localhost:3000"])

        assert "GET" in config.allowed_methods
        assert "POST" in config.allowed_methods
        assert "PUT" in config.allowed_methods
        assert "DELETE" in config.allowed_methods
        assert "OPTIONS" in config.allowed_methods

    def test_cors_config_default_headers(self):
        """Test default allowed headers."""
        config = CORSConfig(allowed_origins=["http://localhost:3000"])

        assert "Authorization" in config.allowed_headers
        assert "Content-Type" in config.allowed_headers
        assert "X-CSRF-Token" in config.allowed_headers


class TestCSRFProtection:
    """Tests for CSRF protection."""

    def test_csrf_token_generation(self):
        """Test that CSRF tokens are generated."""
        csrf = CSRFProtection(secret_key="test-secret-key-32-chars-long!!")

        token = csrf.generate_token()
        assert token is not None
        assert len(token) > 0
        assert ":" in token  # Token format: random:timestamp:signature

    def test_csrf_token_validation_valid(self):
        """Test that valid tokens are accepted."""
        csrf = CSRFProtection(secret_key="test-secret-key-32-chars-long!!")

        token = csrf.generate_token()
        assert csrf.validate_token(token) is True

    def test_csrf_token_validation_invalid(self):
        """Test that invalid tokens are rejected."""
        csrf = CSRFProtection(secret_key="test-secret-key-32-chars-long!!")

        assert csrf.validate_token("invalid-token") is False
        assert csrf.validate_token("") is False
        assert csrf.validate_token("a:b") is False  # Wrong format

    def test_csrf_token_validation_wrong_signature(self):
        """Test that tokens with wrong signature are rejected."""
        csrf = CSRFProtection(secret_key="test-secret-key-32-chars-long!!")

        token = csrf.generate_token()
        # Tamper with the signature
        parts = token.split(":")
        parts[2] = "wrong_signature"
        tampered_token = ":".join(parts)

        assert csrf.validate_token(tampered_token) is False

    def test_csrf_tokens_are_unique(self):
        """Test that generated tokens are unique."""
        csrf = CSRFProtection(secret_key="test-secret-key-32-chars-long!!")

        tokens = [csrf.generate_token() for _ in range(10)]
        assert len(set(tokens)) == 10  # All unique


class TestGlobalInitialization:
    """Tests for global security initialization."""

    def test_init_cors_config(self):
        """Test CORS config initialization."""
        config = init_cors_config(
            allowed_origins=["http://test.com"],
            allow_credentials=True,
        )

        assert config is not None
        assert config.is_origin_allowed("http://test.com") is True

        # Global getter should return same config
        global_config = get_cors_config()
        assert global_config is config

    def test_init_csrf_protection(self):
        """Test CSRF protection initialization."""
        csrf = init_csrf_protection(secret_key="test-key-32-characters-long!!")

        assert csrf is not None

        # Global getter should return same instance
        global_csrf = get_csrf_protection()
        assert global_csrf is csrf

        # Should be able to generate and validate tokens
        token = csrf.generate_token()
        assert csrf.validate_token(token) is True
