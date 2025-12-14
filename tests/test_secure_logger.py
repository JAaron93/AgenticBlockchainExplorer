"""
Unit tests for SecureLogger.

Tests for info/warning/error sanitization and exception stack trace filtering.

Requirements: 1.3, 1.4
"""

import logging
from io import StringIO

import pytest

from core.security.credential_sanitizer import CredentialSanitizer
from core.security.secure_logger import SecureLogger


@pytest.fixture
def sanitizer():
    """Create a CredentialSanitizer with default config."""
    return CredentialSanitizer()


@pytest.fixture
def logger_with_handler():
    """Create a logger with a StringIO handler to capture output."""
    logger = logging.getLogger("test_secure_logger")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger, stream


@pytest.fixture
def secure_logger(sanitizer, logger_with_handler):
    """Create a SecureLogger wrapping the test logger."""
    logger, stream = logger_with_handler
    return SecureLogger(logger, sanitizer), stream


class TestSecureLoggerInfoSanitization:
    """Tests for info() method sanitization. Requirements: 1.3"""

    def test_info_sanitizes_api_key_in_message(self, secure_logger):
        """Test that API keys in info messages are redacted."""
        slogger, stream = secure_logger
        api_key = "abcdefghijklmnopqrstuvwxyz123456"

        slogger.info(f"Request with apikey={api_key}")

        output = stream.getvalue()
        assert api_key not in output
        assert "[REDACTED]" in output

    def test_info_sanitizes_jwt_token(self, secure_logger):
        """Test that JWT tokens in info messages are redacted."""
        slogger, stream = secure_logger
        jwt = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
            "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        )

        slogger.info(f"Auth token: {jwt}")

        output = stream.getvalue()
        assert jwt not in output
        assert "[REDACTED]" in output

    def test_info_sanitizes_args(self, secure_logger):
        """Test that credentials in format args are redacted."""
        slogger, stream = secure_logger
        api_key = "abcdefghijklmnopqrstuvwxyz123456"

        slogger.info("API key is: %s", api_key)

        output = stream.getvalue()
        assert api_key not in output
        assert "[REDACTED]" in output

    def test_info_preserves_non_credential_content(self, secure_logger):
        """Test that non-credential content is preserved."""
        slogger, stream = secure_logger

        slogger.info("Normal message without credentials")

        output = stream.getvalue()
        assert "Normal message without credentials" in output


class TestSecureLoggerWarningSanitization:
    """Tests for warning() method sanitization. Requirements: 1.3"""

    def test_warning_sanitizes_api_key(self, secure_logger):
        """Test that API keys in warning messages are redacted."""
        slogger, stream = secure_logger
        api_key = "abcdefghijklmnopqrstuvwxyz123456"

        slogger.warning(f"Failed request with key {api_key}")

        output = stream.getvalue()
        assert api_key not in output
        assert "[REDACTED]" in output
        assert "WARNING" in output

    def test_warning_sanitizes_dict_extra(self, secure_logger):
        """Test that credentials in extra dict are redacted."""
        slogger, stream = secure_logger
        api_key = "abcdefghijklmnopqrstuvwxyz123456"

        slogger.warning(
            "Request failed",
            extra={"apikey": api_key, "status": 401}
        )

        output = stream.getvalue()
        assert api_key not in output


class TestSecureLoggerErrorSanitization:
    """Tests for error() method sanitization. Requirements: 1.3"""

    def test_error_sanitizes_api_key(self, secure_logger):
        """Test that API keys in error messages are redacted."""
        slogger, stream = secure_logger
        api_key = "abcdefghijklmnopqrstuvwxyz123456"

        slogger.error(f"Error with apikey={api_key}")

        output = stream.getvalue()
        assert api_key not in output
        assert "[REDACTED]" in output
        assert "ERROR" in output

    def test_error_sanitizes_multiple_credentials(self, secure_logger):
        """Test that multiple credentials are all redacted."""
        slogger, stream = secure_logger
        api_key1 = "abcdefghijklmnopqrstuvwxyz123456"
        api_key2 = "zyxwvutsrqponmlkjihgfedcba654321"

        slogger.error(f"Keys: {api_key1} and {api_key2}")

        output = stream.getvalue()
        assert api_key1 not in output
        assert api_key2 not in output


class TestSecureLoggerExceptionSanitization:
    """Tests for exception() with stack trace filtering. Requirements: 1.4"""

    def test_exception_sanitizes_message(self, secure_logger):
        """Test that credentials in exception messages are redacted."""
        slogger, stream = secure_logger
        api_key = "abcdefghijklmnopqrstuvwxyz123456"

        try:
            raise ValueError(f"Invalid key: {api_key}")
        except ValueError:
            slogger.exception("Error occurred")

        output = stream.getvalue()
        assert api_key not in output
        assert "[REDACTED]" in output

    def test_exception_sanitizes_stack_trace(self, secure_logger):
        """Test that credentials in stack traces are redacted."""
        slogger, stream = secure_logger
        secret_value = "supersecretpassword12345678901234"

        try:
            # Create a local variable with credential in trace
            password = secret_value
            raise RuntimeError(f"Auth failed with {password}")
        except RuntimeError:
            slogger.exception("Authentication error")

        output = stream.getvalue()
        assert secret_value not in output

    def test_exception_includes_traceback_info(self, secure_logger):
        """Test that exception logging includes traceback structure."""
        slogger, stream = secure_logger

        try:
            raise ValueError("Test error")
        except ValueError:
            slogger.exception("Caught exception")

        output = stream.getvalue()
        assert "ValueError" in output
        assert "Test error" in output

    def test_exception_without_active_exception(self, secure_logger):
        """Test exception() when no exception is active."""
        slogger, stream = secure_logger

        # Call exception() outside of except block
        slogger.exception("No active exception", exc_info=False)

        output = stream.getvalue()
        assert "No active exception" in output


class TestSecureLoggerProperties:
    """Tests for SecureLogger properties and utility methods."""

    def test_name_property(self, secure_logger):
        """Test that name property returns underlying logger name."""
        slogger, _ = secure_logger
        assert slogger.name == "test_secure_logger"

    def test_level_property(self, secure_logger):
        """Test that level property returns underlying logger level."""
        slogger, _ = secure_logger
        assert slogger.level == logging.DEBUG

    def test_set_level(self, secure_logger):
        """Test that setLevel updates underlying logger."""
        slogger, _ = secure_logger
        slogger.setLevel(logging.WARNING)
        assert slogger.level == logging.WARNING

    def test_is_enabled_for(self, secure_logger):
        """Test isEnabledFor method."""
        slogger, _ = secure_logger
        slogger.setLevel(logging.WARNING)

        assert slogger.isEnabledFor(logging.ERROR) is True
        assert slogger.isEnabledFor(logging.DEBUG) is False

    def test_get_effective_level(self, secure_logger):
        """Test getEffectiveLevel method."""
        slogger, _ = secure_logger
        assert slogger.getEffectiveLevel() == logging.DEBUG


class TestSecureLoggerDebugAndCritical:
    """Tests for debug() and critical() methods."""

    def test_debug_sanitizes_credentials(self, secure_logger):
        """Test that debug messages are sanitized."""
        slogger, stream = secure_logger
        api_key = "abcdefghijklmnopqrstuvwxyz123456"

        slogger.debug(f"Debug: key={api_key}")

        output = stream.getvalue()
        assert api_key not in output
        assert "[REDACTED]" in output

    def test_critical_sanitizes_credentials(self, secure_logger):
        """Test that critical messages are sanitized."""
        slogger, stream = secure_logger
        api_key = "abcdefghijklmnopqrstuvwxyz123456"

        slogger.critical(f"Critical: key={api_key}")

        output = stream.getvalue()
        assert api_key not in output
        assert "[REDACTED]" in output
        assert "CRITICAL" in output


class TestSecureLoggerLogMethod:
    """Tests for the generic log() method."""

    def test_log_at_info_level(self, secure_logger):
        """Test log() at INFO level sanitizes credentials."""
        slogger, stream = secure_logger
        api_key = "abcdefghijklmnopqrstuvwxyz123456"

        slogger.log(logging.INFO, f"Log: key={api_key}")

        output = stream.getvalue()
        assert api_key not in output
        assert "[REDACTED]" in output
        assert "INFO" in output
