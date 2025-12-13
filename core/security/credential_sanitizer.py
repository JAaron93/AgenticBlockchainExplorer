"""Credential sanitization for secure logging and error handling.

This module provides the CredentialSanitizer class that detects and redacts
sensitive credentials from strings, dictionaries, and URLs.

Requirements: 1.1, 1.2, 1.5
"""

import re
from typing import Any, Dict, List, Optional, Pattern
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from config.models import CredentialSanitizerConfig


class CredentialSanitizer:
    """Detects and redacts credentials from strings.

    This class provides methods to sanitize sensitive credential values
    from various data formats including plain strings, dictionaries,
    and URLs with query parameters.

    The sanitizer uses configurable patterns to detect:
    - Known sensitive parameter names (apikey, token, secret, etc.)
    - Known sensitive header names (Authorization, X-API-Key, etc.)
    - Credential value patterns (32+ char alphanumeric, JWT tokens, etc.)

    Requirements: 1.1, 1.2, 1.5
    """

    def __init__(self, config: Optional[CredentialSanitizerConfig] = None):
        """Initialize with configurable patterns.

        Args:
            config: Configuration for credential detection patterns.
                   If None, uses default configuration.
        """
        self._config = config or CredentialSanitizerConfig()
        self._placeholder = self._config.redaction_placeholder

        # Compile regex patterns for efficient matching
        self._credential_patterns: List[Pattern[str]] = [
            re.compile(pattern) for pattern in self._config.credential_patterns
        ]

        # Create case-insensitive sets for name matching
        self._sensitive_param_names = {
            name.lower() for name in self._config.sensitive_param_names
        }
        self._sensitive_header_names = {
            name.lower() for name in self._config.sensitive_header_names
        }

    def sanitize(self, text: str) -> str:
        """Remove all detected credentials from text.

        Scans the input text for patterns matching known credential formats
        and replaces them with the redaction placeholder.

        Args:
            text: Input text that may contain credentials.

        Returns:
            Text with all detected credentials replaced by placeholder.

        Requirements: 1.1, 1.2
        """
        if not text:
            return text

        result = text

        # Replace matches for each credential pattern
        for pattern in self._credential_patterns:
            result = pattern.sub(self._placeholder, result)

        return result

    def sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize all string values in a dictionary.

        Traverses the dictionary structure and sanitizes:
        - Values of keys that match sensitive parameter/header names
        - String values that match credential patterns
        - Nested dictionaries and lists

        Args:
            data: Dictionary that may contain credential values.

        Returns:
            New dictionary with credentials redacted.

        Requirements: 1.2, 1.5
        """
        if not isinstance(data, dict):
            return data

        result: Dict[str, Any] = {}

        for key, value in data.items():
            sanitized_value = self._sanitize_value(key, value)
            result[key] = sanitized_value

        return result

    def _sanitize_value(self, key: str, value: Any) -> Any:
        """Sanitize a single value based on its key and type.

        Args:
            key: The dictionary key for this value.
            value: The value to potentially sanitize.

        Returns:
            Sanitized value.
        """
        # Check if key indicates a sensitive field
        if self._is_sensitive_key(key):
            if isinstance(value, str):
                return self._placeholder
            elif isinstance(value, (int, float)):
                return self._placeholder
            # For complex types, still try to redact
            elif value is not None:
                return self._placeholder

        # Handle different value types
        if isinstance(value, str):
            # Check if value looks like a credential
            if self._matches_credential_pattern(value):
                return self._placeholder
            return self.sanitize(value)
        elif isinstance(value, dict):
            return self.sanitize_dict(value)
        elif isinstance(value, list):
            return [
                self._sanitize_value(key, item) for item in value
            ]
        else:
            return value

    def sanitize_url(self, url: str) -> str:
        """Sanitize URL query parameters containing credentials.

        Parses the URL and redacts values of query parameters that
        match sensitive parameter names or credential patterns.

        Args:
            url: URL that may contain credentials in query parameters.

        Returns:
            URL with credential query parameter values redacted.

        Requirements: 1.3
        """
        if not url:
            return url

        try:
            parsed = urlparse(url)

            # If no query string, return as-is
            if not parsed.query:
                return url

            # Parse query parameters
            query_params = parse_qs(parsed.query, keep_blank_values=True)

            # Sanitize each parameter
            sanitized_params: Dict[str, List[str]] = {}
            for key, values in query_params.items():
                if self._is_sensitive_key(key):
                    # Redact all values for sensitive keys
                    sanitized_params[key] = [self._placeholder] * len(values)
                else:
                    # Check each value for credential patterns
                    sanitized_values = []
                    for val in values:
                        if self._matches_credential_pattern(val):
                            sanitized_values.append(self._placeholder)
                        else:
                            sanitized_values.append(val)
                    sanitized_params[key] = sanitized_values

            # Rebuild query string (flatten single-value lists)
            flat_params: Dict[str, str] = {}
            for key, values in sanitized_params.items():
                if len(values) == 1:
                    flat_params[key] = values[0]
                else:
                    # For multiple values, join with comma
                    flat_params[key] = ",".join(values)

            # Reconstruct URL
            new_query = urlencode(flat_params)
            sanitized_url = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                new_query,
                parsed.fragment,
            ))

            return sanitized_url

        except Exception:
            # If URL parsing fails, apply basic string sanitization
            return self.sanitize(url)

    def is_credential(self, key: str, value: str) -> bool:
        """Check if a key-value pair appears to be a credential.

        Args:
            key: The parameter or header name.
            value: The value to check.

        Returns:
            True if the key-value pair appears to be a credential.

        Requirements: 1.6, 1.7, 1.8
        """
        # Check if key is a known sensitive name
        if self._is_sensitive_key(key):
            return True

        # Check if value matches credential patterns
        if isinstance(value, str) and self._matches_credential_pattern(value):
            return True

        return False

    def _is_sensitive_key(self, key: str) -> bool:
        """Check if a key name indicates a sensitive field.

        Args:
            key: The key name to check.

        Returns:
            True if the key matches a sensitive parameter or header name.
        """
        key_lower = key.lower()
        return (
            key_lower in self._sensitive_param_names
            or key_lower in self._sensitive_header_names
        )

    def _matches_credential_pattern(self, value: str) -> bool:
        """Check if a value matches any credential pattern.

        Args:
            value: The string value to check.

        Returns:
            True if the value matches a credential pattern.
        """
        if not value or len(value) < 32:
            # Most credentials are at least 32 chars
            return False

        for pattern in self._credential_patterns:
            if pattern.fullmatch(value):
                return True

        return False
