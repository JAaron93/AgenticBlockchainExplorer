"""
Property-based tests for credential sanitization.

These tests use Hypothesis to verify correctness properties defined in the design document.

**Feature: agent-security-hardening, Property 1: Credential Sanitization Completeness**
**Validates: Requirements 1.1, 1.2, 1.3, 1.5**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

from config.models import CredentialSanitizerConfig
from core.security.credential_sanitizer import CredentialSanitizer


# =============================================================================
# Test Data Strategies
# =============================================================================


def alphanumeric_credential(min_size: int = 32, max_size: int = 64):
    """Generate alphanumeric strings that match the credential pattern [a-zA-Z0-9]{32,}."""
    return st.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        min_size=min_size,
        max_size=max_size,
    )


def jwt_token():
    """Generate JWT-like tokens matching eyJ[a-zA-Z0-9_-]+.[a-zA-Z0-9_-]+.[a-zA-Z0-9_-]+."""
    jwt_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    return st.tuples(
        st.text(alphabet=jwt_chars, min_size=10, max_size=50),
        st.text(alphabet=jwt_chars, min_size=10, max_size=50),
        st.text(alphabet=jwt_chars, min_size=10, max_size=50),
    ).map(lambda parts: f"eyJ{parts[0]}.{parts[1]}.{parts[2]}")


def credential_value():
    """Generate credential values that match configured patterns."""
    return st.one_of(
        alphanumeric_credential(),
        jwt_token(),
    )


def sensitive_param_name():
    """Generate sensitive parameter names from the default config."""
    return st.sampled_from([
        "apikey", "api_key", "API_KEY", "token", "auth_token",
        "secret", "password", "client_secret",
    ])


def sensitive_header_name():
    """Generate sensitive header names from the default config."""
    return st.sampled_from([
        "Authorization", "X-API-Key", "X-Auth-Token",
    ])


def non_credential_text():
    """Generate text that does NOT match credential patterns.
    
    This generates short strings (< 32 chars) that won't match the
    alphanumeric credential pattern, and avoids JWT-like patterns.
    """
    return st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789 .,!?-_",
        min_size=0,
        max_size=30,
    ).filter(lambda x: not x.startswith("eyJ"))


@st.composite
def text_with_embedded_credential(draw):
    """Generate text with a credential embedded somewhere in it."""
    prefix = draw(non_credential_text())
    credential = draw(credential_value())
    suffix = draw(non_credential_text())
    return prefix + credential + suffix, credential


@st.composite
def dict_with_sensitive_key(draw):
    """Generate a dictionary with a sensitive key containing a credential."""
    key = draw(sensitive_param_name())
    value = draw(credential_value())
    # Add some non-sensitive keys too
    other_keys = draw(st.dictionaries(
        keys=st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=10).filter(
            lambda k: k.lower() not in ["apikey", "api_key", "token", "auth_token", 
                                         "secret", "password", "client_secret"]
        ),
        values=st.text(min_size=0, max_size=20),
        min_size=0,
        max_size=3,
    ))
    result = {key: value, **other_keys}
    return result, key, value


@st.composite
def url_with_credential_param(draw):
    """Generate a URL with a credential in query parameters."""
    base_url = draw(st.sampled_from([
        "https://api.etherscan.io/api",
        "https://api.bscscan.com/api",
        "https://api.polygonscan.com/api",
    ]))
    param_name = draw(sensitive_param_name())
    credential = draw(alphanumeric_credential())
    # Add some non-sensitive params
    other_params = draw(st.dictionaries(
        keys=st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=10).filter(
            lambda k: k.lower() not in ["apikey", "api_key", "token", "auth_token",
                                         "secret", "password", "client_secret"]
        ),
        values=st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=10),
        min_size=0,
        max_size=2,
    ))
    
    # Build query string
    params = {param_name: credential, **other_params}
    query_string = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{base_url}?{query_string}"
    
    return url, credential


# =============================================================================
# Property Tests
# =============================================================================


class TestCredentialSanitizationCompleteness:
    """
    Property tests for credential sanitization completeness.
    
    **Feature: agent-security-hardening, Property 1: Credential Sanitization Completeness**
    
    For any string containing a credential pattern (API key, token, secret),
    the sanitized output SHALL NOT contain that credential value.
    
    **Validates: Requirements 1.1, 1.2, 1.3, 1.5**
    """

    @settings(max_examples=100)
    @given(data=text_with_embedded_credential())
    def test_property_1_sanitize_removes_credential_from_text(self, data):
        """
        **Feature: agent-security-hardening, Property 1: Credential Sanitization Completeness**
        
        For any text containing a credential pattern, sanitize() SHALL remove
        the credential value from the output.
        
        **Validates: Requirements 1.1, 1.2**
        """
        text, credential = data
        sanitizer = CredentialSanitizer()
        
        # Sanitize the text
        sanitized = sanitizer.sanitize(text)
        
        # The credential should NOT appear in the sanitized output
        assert credential not in sanitized, (
            f"Credential '{credential[:10]}...' still present in sanitized output"
        )
        
        # The placeholder should appear instead
        assert sanitizer._placeholder in sanitized, (
            f"Placeholder '{sanitizer._placeholder}' not found in sanitized output"
        )

    @settings(max_examples=100)
    @given(credential=credential_value())
    def test_property_1_standalone_credential_is_redacted(self, credential):
        """
        **Feature: agent-security-hardening, Property 1: Credential Sanitization Completeness**
        
        For any standalone credential value, sanitize() SHALL replace it
        entirely with the redaction placeholder.
        
        **Validates: Requirements 1.1**
        """
        sanitizer = CredentialSanitizer()
        
        # Sanitize the credential directly
        sanitized = sanitizer.sanitize(credential)
        
        # The credential should be completely replaced
        assert credential not in sanitized, (
            f"Credential '{credential[:10]}...' still present after sanitization"
        )
        assert sanitizer._placeholder in sanitized, (
            f"Placeholder not found in sanitized output"
        )

    @settings(max_examples=100)
    @given(data=dict_with_sensitive_key())
    def test_property_1_sanitize_dict_redacts_sensitive_keys(self, data):
        """
        **Feature: agent-security-hardening, Property 1: Credential Sanitization Completeness**
        
        For any dictionary with a sensitive key (apikey, token, secret, etc.),
        sanitize_dict() SHALL redact the value regardless of its content.
        
        **Validates: Requirements 1.2, 1.5**
        """
        input_dict, sensitive_key, credential = data
        sanitizer = CredentialSanitizer()
        
        # Sanitize the dictionary
        sanitized = sanitizer.sanitize_dict(input_dict)
        
        # The credential value should NOT appear in the sanitized dict
        assert sanitized[sensitive_key] == sanitizer._placeholder, (
            f"Sensitive key '{sensitive_key}' value not redacted. "
            f"Expected '{sanitizer._placeholder}', got '{sanitized[sensitive_key][:20]}...'"
        )
        
        # Original credential should not be anywhere in the sanitized dict values
        for key, value in sanitized.items():
            if isinstance(value, str):
                assert credential not in value, (
                    f"Credential found in sanitized dict under key '{key}'"
                )

    @settings(max_examples=100)
    @given(data=url_with_credential_param())
    def test_property_1_sanitize_url_redacts_credential_params(self, data):
        """
        **Feature: agent-security-hardening, Property 1: Credential Sanitization Completeness**
        
        For any URL with a credential in query parameters, sanitize_url()
        SHALL redact the credential value from the URL.
        
        **Validates: Requirements 1.3**
        """
        from urllib.parse import quote
        
        url, credential = data
        sanitizer = CredentialSanitizer()
        
        # Sanitize the URL
        sanitized_url = sanitizer.sanitize_url(url)
        
        # The credential should NOT appear in the sanitized URL
        assert credential not in sanitized_url, (
            f"Credential '{credential[:10]}...' still present in sanitized URL"
        )
        
        # The placeholder should appear in the URL (may be URL-encoded)
        placeholder = sanitizer._placeholder
        url_encoded_placeholder = quote(placeholder, safe='')
        assert placeholder in sanitized_url or url_encoded_placeholder in sanitized_url, (
            f"Placeholder '{placeholder}' (or URL-encoded '{url_encoded_placeholder}') "
            f"not found in sanitized URL: {sanitized_url}"
        )

    @settings(max_examples=100)
    @given(
        key=sensitive_param_name(),
        value=credential_value(),
    )
    def test_property_1_is_credential_detects_sensitive_keys(self, key, value):
        """
        **Feature: agent-security-hardening, Property 1: Credential Sanitization Completeness**
        
        For any key-value pair where the key is a sensitive parameter name,
        is_credential() SHALL return True.
        
        **Validates: Requirements 1.5**
        """
        sanitizer = CredentialSanitizer()
        
        # is_credential should detect sensitive keys
        assert sanitizer.is_credential(key, value) is True, (
            f"is_credential() failed to detect sensitive key '{key}'"
        )

    @settings(max_examples=100)
    @given(credential=credential_value())
    def test_property_1_is_credential_detects_credential_patterns(self, credential):
        """
        **Feature: agent-security-hardening, Property 1: Credential Sanitization Completeness**
        
        For any value matching a credential pattern, is_credential() SHALL
        return True even with a non-sensitive key name.
        
        **Validates: Requirements 1.5**
        """
        sanitizer = CredentialSanitizer()
        
        # Use a non-sensitive key name
        non_sensitive_key = "some_random_field"
        
        # is_credential should detect the credential pattern in the value
        assert sanitizer.is_credential(non_sensitive_key, credential) is True, (
            f"is_credential() failed to detect credential pattern in value"
        )

    @settings(max_examples=100)
    @given(
        credentials=st.lists(credential_value(), min_size=1, max_size=5),
        separator=st.sampled_from([" ", ", ", "\n", " | ", "; "]),
    )
    def test_property_1_multiple_credentials_all_redacted(self, credentials, separator):
        """
        **Feature: agent-security-hardening, Property 1: Credential Sanitization Completeness**
        
        For any text containing multiple credentials, sanitize() SHALL
        redact ALL credential values.
        
        **Validates: Requirements 1.1, 1.2**
        """
        # Ensure credentials are unique for this test
        unique_credentials = list(set(credentials))
        assume(len(unique_credentials) >= 1)
        
        text = separator.join(unique_credentials)
        sanitizer = CredentialSanitizer()
        
        # Sanitize the text
        sanitized = sanitizer.sanitize(text)
        
        # NONE of the credentials should appear in the sanitized output
        for cred in unique_credentials:
            assert cred not in sanitized, (
                f"Credential '{cred[:10]}...' still present in sanitized output "
                f"containing multiple credentials"
            )

    @settings(max_examples=100)
    @given(
        nested_depth=st.integers(min_value=1, max_value=5),
        credential=credential_value(),
        sensitive_key=sensitive_param_name(),
    )
    def test_property_1_nested_dict_credentials_redacted(
        self, nested_depth, credential, sensitive_key
    ):
        """
        **Feature: agent-security-hardening, Property 1: Credential Sanitization Completeness**
        
        For any nested dictionary structure containing credentials,
        sanitize_dict() SHALL redact credentials at all nesting levels.
        
        **Validates: Requirements 1.2, 1.5**
        """
        sanitizer = CredentialSanitizer()
        
        # Build a nested dictionary with the credential at the deepest level
        nested_dict = {sensitive_key: credential}
        for i in range(nested_depth):
            nested_dict = {f"level_{i}": nested_dict}
        
        # Sanitize the nested dictionary
        sanitized = sanitizer.sanitize_dict(nested_dict)
        
        # Helper to check if credential exists anywhere in nested structure
        def contains_credential(obj, cred):
            if isinstance(obj, str):
                return cred in obj
            elif isinstance(obj, dict):
                return any(contains_credential(v, cred) for v in obj.values())
            elif isinstance(obj, list):
                return any(contains_credential(item, cred) for item in obj)
            return False
        
        # The credential should NOT appear anywhere in the sanitized structure
        assert not contains_credential(sanitized, credential), (
            f"Credential found in nested dict at depth {nested_depth}"
        )

