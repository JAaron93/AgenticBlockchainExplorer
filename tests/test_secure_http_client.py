"""
Integration tests for SecureHTTPClient.

These tests verify that the SecureHTTPClient correctly integrates
all security components: SSRF protection, response size limiting,
and credential sanitization.

Requirements: 1.1, 2.4, 2.5, 3.1
"""

import logging
from unittest.mock import AsyncMock

import pytest

from config.models import (
    CredentialSanitizerConfig,
    ResourceLimitConfig,
    SSRFProtectionConfig,
)
from core.security.credential_sanitizer import CredentialSanitizer
from core.security.resource_limiter import (
    ResourceLimiter,
    ResponseTooLargeError,
)
from core.security.secure_http_client import (
    InvalidParameterError,
    SecureHTTPClient,
)
from core.security.secure_logger import SecureLogger
from core.security.ssrf_protector import (
    DomainNotAllowedError,
    ProtocolNotAllowedError,
    SSRFProtector,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ssrf_config():
    """Create SSRF protection config with test domains."""
    return SSRFProtectionConfig(
        allowed_domains=[
            "api.etherscan.io",
            "*.etherscan.io",
            "api.bscscan.com",
            "api.polygonscan.com",
        ],
        require_https=True,
        block_private_ips=True,
    )


@pytest.fixture
def ssrf_protector(ssrf_config):
    """Create SSRFProtector instance."""
    return SSRFProtector.from_config(ssrf_config)


@pytest.fixture
def resource_config():
    """Create resource limit config."""
    return ResourceLimitConfig(
        max_response_size_bytes=1024 * 1024,  # 1MB for tests
        max_output_file_size_bytes=10 * 1024 * 1024,
        max_memory_usage_mb=512,
    )


@pytest.fixture
def resource_limiter(resource_config):
    """Create ResourceLimiter instance."""
    return ResourceLimiter(resource_config)


@pytest.fixture
def sanitizer_config():
    """Create credential sanitizer config."""
    return CredentialSanitizerConfig()


@pytest.fixture
def sanitizer(sanitizer_config):
    """Create CredentialSanitizer instance."""
    return CredentialSanitizer(sanitizer_config)


@pytest.fixture
def secure_logger(sanitizer):
    """Create SecureLogger instance."""
    logger = logging.getLogger("test_secure_http_client")
    return SecureLogger(logger, sanitizer)


@pytest.fixture
def secure_client(ssrf_protector, resource_limiter, sanitizer, secure_logger):
    """Create SecureHTTPClient instance."""
    return SecureHTTPClient(
        ssrf_protector=ssrf_protector,
        resource_limiter=resource_limiter,
        sanitizer=sanitizer,
        secure_logger=secure_logger,
        timeout=30.0,
    )


# =============================================================================
# Parameter Validation Tests
# =============================================================================


class TestParameterValidation:
    """Tests for parameter validation and sanitization.

    Requirements: 2.4, 2.5
    """

    def test_valid_params_accepted(self, secure_client):
        """Valid explorer API parameters should be accepted."""
        params = {
            "module": "account",
            "action": "tokentx",
            "contractaddress": "0x123",
            "address": "0x456",
            "page": 1,
            "offset": 100,
            "sort": "asc",
            "apikey": "test_key",
        }

        validated = secure_client._validate_params(params)

        assert validated["module"] == "account"
        assert validated["action"] == "tokentx"
        assert validated["page"] == "1"
        assert validated["offset"] == "100"

    def test_invalid_param_key_rejected(self, secure_client):
        """Parameters with keys not in allowlist should be rejected."""
        params = {
            "module": "account",
            "invalid_key": "value",
        }

        with pytest.raises(InvalidParameterError) as exc_info:
            secure_client._validate_params(params)

        assert "invalid_key" in str(exc_info.value)
        assert "not in the allowed list" in str(exc_info.value)

    def test_invalid_param_type_rejected(self, secure_client):
        """Parameters with invalid types should be rejected."""
        params = {
            "module": "account",
            "page": ["list", "not", "allowed"],
        }

        with pytest.raises(InvalidParameterError) as exc_info:
            secure_client._validate_params(params)

        assert "invalid type" in str(exc_info.value)

    def test_url_in_param_value_rejected(self, secure_client):
        """Parameter values that look like URLs should be rejected."""
        params = {
            "module": "account",
            "address": "https://evil.com/attack",
        }

        with pytest.raises(InvalidParameterError) as exc_info:
            secure_client._validate_params(params)

        assert "appears to be a URL" in str(exc_info.value)

    def test_bool_params_converted_to_lowercase(self, secure_client):
        """Boolean parameters should be converted to lowercase strings."""
        params = {
            "module": "account",
            "sort": True,
        }

        validated = secure_client._validate_params(params)

        assert validated["sort"] == "true"

    def test_empty_params_returns_empty_dict(self, secure_client):
        """Empty or None params should return empty dict."""
        assert secure_client._validate_params(None) == {}
        assert secure_client._validate_params({}) == {}

    def test_case_insensitive_param_keys(self, secure_client):
        """Parameter key validation should be case-insensitive."""
        params = {
            "MODULE": "account",
            "Action": "tokentx",
        }

        validated = secure_client._validate_params(params)

        assert "MODULE" in validated
        assert "Action" in validated


class TestParamSanitizationForLogging:
    """Tests for credential sanitization in logging.

    Requirements: 1.1, 1.2
    """

    def test_apikey_redacted_in_logging(self, secure_client):
        """API keys should be redacted when sanitizing for logging."""
        params = {
            "module": "account",
            "apikey": "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456",
        }

        sanitized = secure_client._sanitize_params_for_logging(params)

        assert sanitized["apikey"] == "[REDACTED]"
        assert sanitized["module"] == "account"

    def test_token_redacted_in_logging(self, secure_client):
        """Token values should be redacted when sanitizing for logging."""
        params = {
            "module": "account",
            "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.sig",
        }

        sanitized = secure_client._sanitize_params_for_logging(params)

        assert sanitized["token"] == "[REDACTED]"

    def test_non_sensitive_params_preserved(self, secure_client):
        """Non-sensitive parameters should be preserved in logging."""
        params = {
            "module": "account",
            "action": "tokentx",
            "page": "1",
        }

        sanitized = secure_client._sanitize_params_for_logging(params)

        assert sanitized["module"] == "account"
        assert sanitized["action"] == "tokentx"
        assert sanitized["page"] == "1"


# =============================================================================
# SSRF Protection Integration Tests
# =============================================================================


class TestSSRFProtectionIntegration:
    """Tests for SSRF protection integration.

    Requirements: 2.4, 2.5
    """

    @pytest.mark.asyncio
    async def test_allowed_domain_passes_validation(self, secure_client):
        """Requests to allowed domains should pass SSRF validation."""
        url = "https://api.etherscan.io/api"

        # Should not raise
        await secure_client._ssrf_protector.validate_request(url)

    @pytest.mark.asyncio
    async def test_disallowed_domain_rejected(self, secure_client):
        """Requests to disallowed domains should be rejected."""
        url = "https://evil.com/api"

        with pytest.raises(DomainNotAllowedError):
            await secure_client._ssrf_protector.validate_request(url)

    @pytest.mark.asyncio
    async def test_http_protocol_rejected(self, secure_client):
        """HTTP (non-HTTPS) requests should be rejected."""
        url = "http://api.etherscan.io/api"

        with pytest.raises(ProtocolNotAllowedError):
            await secure_client._ssrf_protector.validate_request(url)

    @pytest.mark.asyncio
    async def test_wildcard_subdomain_allowed(self, secure_client):
        """Subdomains matching wildcard patterns should be allowed.

        Note: This test only validates domain allowlist, not DNS resolution.
        DNS resolution is tested separately in test_dns_rebinding_protection.py
        """
        # Test the allowlist directly without DNS resolution
        allowlist = secure_client._ssrf_protector._allowlist
        assert allowlist.is_allowed("subdomain.etherscan.io") is True


# =============================================================================
# Response Size Limiting Tests
# =============================================================================


class TestResponseSizeLimiting:
    """Tests for response size limiting integration.

    Requirements: 3.1
    """

    def test_small_response_accepted(self, resource_limiter):
        """Responses under the size limit should be accepted."""
        # Should not raise
        resource_limiter.check_response_size(1000)

    def test_large_response_rejected(self, resource_limiter):
        """Responses over the size limit should be rejected."""
        # 2MB exceeds 1MB limit
        with pytest.raises(ResponseTooLargeError) as exc_info:
            resource_limiter.check_response_size(2 * 1024 * 1024)

        assert exc_info.value.size == 2 * 1024 * 1024
        assert exc_info.value.limit == 1024 * 1024

    @pytest.mark.asyncio
    async def test_read_response_with_limit_enforced(self, secure_client):
        """Response reading should enforce size limits."""
        # Create mock response with large content
        mock_response = AsyncMock()

        # Simulate chunked response that exceeds limit
        chunks = [b"x" * 8192 for _ in range(200)]  # ~1.6MB total

        class MockAsyncIterator:
            def __init__(self, data):
                self.data = data
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.data):
                    raise StopAsyncIteration
                chunk = self.data[self.index]
                self.index += 1
                return chunk

        mock_response.content.iter_chunked = lambda _: MockAsyncIterator(chunks)

        with pytest.raises(ResponseTooLargeError):
            await secure_client._read_response_with_limit(mock_response)


# =============================================================================
# URL Resolution Tests
# =============================================================================


class TestURLResolution:
    """Tests for URL resolution and redirect handling."""

    def test_resolve_absolute_redirect(self, secure_client):
        """Absolute redirect URLs should be returned unchanged."""
        base = "https://api.etherscan.io/api"
        redirect = "https://api.etherscan.io/v2/api"

        result = secure_client._resolve_redirect_url(base, redirect)

        assert result == redirect

    def test_resolve_relative_redirect(self, secure_client):
        """Relative redirect URLs should be resolved against base."""
        base = "https://api.etherscan.io/api"
        redirect = "/v2/api"

        result = secure_client._resolve_redirect_url(base, redirect)

        assert result == "https://api.etherscan.io/v2/api"

    def test_resolve_relative_path_redirect(self, secure_client):
        """Relative path redirects should be resolved correctly."""
        base = "https://api.etherscan.io/api/v1"
        redirect = "../v2"

        result = secure_client._resolve_redirect_url(base, redirect)

        assert result == "https://api.etherscan.io/v2"

# =============================================================================
# URL Detection Tests
# =============================================================================


class TestURLDetection:
    """Tests for URL detection in parameter values."""

    def test_https_url_detected(self, secure_client):
        """HTTPS URLs should be detected."""
        assert secure_client._looks_like_url("https://example.com")

    def test_http_url_detected(self, secure_client):
        """HTTP URLs should be detected."""
        assert secure_client._looks_like_url("http://example.com")

    def test_ftp_url_detected(self, secure_client):
        """FTP URLs should be detected."""
        assert secure_client._looks_like_url("ftp://example.com")

    def test_file_url_detected(self, secure_client):
        """File URLs should be detected."""
        assert secure_client._looks_like_url("file:///etc/passwd")

    def test_protocol_relative_url_detected(self, secure_client):
        """Protocol-relative URLs should be detected."""
        assert secure_client._looks_like_url("//example.com/path")

    def test_normal_value_not_detected_as_url(self, secure_client):
        """Normal parameter values should not be detected as URLs."""
        assert not secure_client._looks_like_url("0x123456")
        assert not secure_client._looks_like_url("account")
        assert not secure_client._looks_like_url("100")

    def test_empty_value_not_detected_as_url(self, secure_client):
        """Empty values should not be detected as URLs."""
        assert not secure_client._looks_like_url("")
        assert not secure_client._looks_like_url(None)


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestContextManager:
    """Tests for async context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager_closes_session(
        self, ssrf_protector, resource_limiter, sanitizer
    ):
        """Context manager should close session on exit."""
        client = SecureHTTPClient(
            ssrf_protector=ssrf_protector,
            resource_limiter=resource_limiter,
            sanitizer=sanitizer,
        )

        async with client:
            # Create a session
            session = await client._get_session()
            assert session is not None
            assert not session.closed

        # Session should be closed after context exit
        assert client._session is None


# =============================================================================
# Allowed Parameter Keys Tests
# =============================================================================


class TestAllowedParameterKeys:
    """Tests for the ALLOWED_PARAM_KEYS constant."""

    def test_common_explorer_params_allowed(self, secure_client):
        """Common explorer API parameters should be in allowlist."""
        expected_keys = {
            "module",
            "action",
            "contractaddress",
            "address",
            "page",
            "offset",
            "sort",
            "startblock",
            "endblock",
            "apikey",
        }

        for key in expected_keys:
            assert key in secure_client.ALLOWED_PARAM_KEYS, (
                f"Expected '{key}' to be in ALLOWED_PARAM_KEYS"
            )

    def test_dangerous_params_not_allowed(self, secure_client):
        """Potentially dangerous parameters should not be allowed."""
        dangerous_keys = {
            "url",
            "redirect",
            "callback",
            "next",
            "return",
            "file",
            "path",
        }

        for key in dangerous_keys:
            assert key not in secure_client.ALLOWED_PARAM_KEYS, (
                f"Dangerous key '{key}' should not be in ALLOWED_PARAM_KEYS"
            )


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in SecureHTTPClient."""

    @pytest.mark.asyncio
    async def test_ssrf_error_not_retried(
        self, ssrf_protector, resource_limiter, sanitizer
    ):
        """SSRF errors should not be retried."""
        client = SecureHTTPClient(
            ssrf_protector=ssrf_protector,
            resource_limiter=resource_limiter,
            sanitizer=sanitizer,
        )

        with pytest.raises(DomainNotAllowedError):
            await client.get_with_retry(
                url="https://evil.com/api",
                max_retries=3,
            )

        await client.close()

    @pytest.mark.asyncio
    async def test_invalid_param_error_not_retried(
        self, ssrf_protector, resource_limiter, sanitizer
    ):
        """Invalid parameter errors should not be retried."""
        client = SecureHTTPClient(
            ssrf_protector=ssrf_protector,
            resource_limiter=resource_limiter,
            sanitizer=sanitizer,
        )

        with pytest.raises(InvalidParameterError):
            await client.get_with_retry(
                url="https://api.etherscan.io/api",
                params={"invalid_key": "value"},
                max_retries=3,
            )

        await client.close()
