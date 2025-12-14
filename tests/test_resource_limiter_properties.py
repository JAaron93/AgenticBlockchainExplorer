"""
Property-based tests for resource limiter response size enforcement.

These tests use Hypothesis to verify correctness properties defined in the design document.

**Feature: agent-security-hardening, Property 4: Response Size Enforcement**
**Validates: Requirements 3.1, 3.2**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

from config.models import ResourceLimitConfig
from core.security.resource_limiter import (
    ResourceLimiter,
    ResponseTooLargeError,
)


# =============================================================================
# Test Data Strategies
# =============================================================================


def valid_response_size_limit():
    """Generate valid response size limits within config bounds.
    
    ResourceLimitConfig allows:
    - min: 1024 (1KB)
    - max: 100 * 1024 * 1024 (100MB)
    """
    return st.integers(min_value=1024, max_value=100 * 1024 * 1024)


def response_size_exceeding_limit(limit: int):
    """Generate response sizes that exceed the given limit."""
    # Generate sizes from limit+1 up to limit*2 (capped at reasonable max)
    max_size = min(limit * 2, 200 * 1024 * 1024)  # Cap at 200MB for test sanity
    return st.integers(min_value=limit + 1, max_value=max_size)


def response_size_within_limit(limit: int):
    """Generate response sizes that are within the given limit."""
    return st.integers(min_value=0, max_value=limit)


@st.composite
def limit_and_exceeding_size(draw):
    """Generate a limit and a size that exceeds it."""
    limit = draw(valid_response_size_limit())
    # Size must exceed limit but stay reasonable
    max_exceeding = min(limit * 2, 200 * 1024 * 1024)
    size = draw(st.integers(min_value=limit + 1, max_value=max_exceeding))
    return limit, size


@st.composite
def limit_and_within_size(draw):
    """Generate a limit and a size that is within it."""
    limit = draw(valid_response_size_limit())
    size = draw(st.integers(min_value=0, max_value=limit))
    return limit, size


# =============================================================================
# Property Tests
# =============================================================================


class TestResponseSizeEnforcement:
    """
    Property tests for response size enforcement.
    
    **Feature: agent-security-hardening, Property 4: Response Size Enforcement**
    
    For any API response exceeding the configured size limit, the resource
    limiter SHALL abort the request before fully reading the response.
    
    **Validates: Requirements 3.1, 3.2**
    """

    @settings(max_examples=100)
    @given(data=limit_and_exceeding_size())
    def test_property_4_response_exceeding_limit_raises_error(self, data):
        """
        **Feature: agent-security-hardening, Property 4: Response Size Enforcement**
        
        For any response size exceeding the configured limit, check_response_size()
        SHALL raise ResponseTooLargeError.
        
        **Validates: Requirements 3.1, 3.2**
        """
        limit, size = data
        
        # Create config with the generated limit
        config = ResourceLimitConfig(max_response_size_bytes=limit)
        limiter = ResourceLimiter(config)
        
        # Checking a size that exceeds the limit should raise an error
        with pytest.raises(ResponseTooLargeError) as exc_info:
            limiter.check_response_size(size)
        
        # Verify the error contains correct information
        assert exc_info.value.size == size, (
            f"Error size {exc_info.value.size} doesn't match input size {size}"
        )
        assert exc_info.value.limit == limit, (
            f"Error limit {exc_info.value.limit} doesn't match config limit {limit}"
        )

    @settings(max_examples=100)
    @given(data=limit_and_within_size())
    def test_property_4_response_within_limit_passes(self, data):
        """
        **Feature: agent-security-hardening, Property 4: Response Size Enforcement**
        
        For any response size within the configured limit, check_response_size()
        SHALL NOT raise an error.
        
        **Validates: Requirements 3.1**
        """
        limit, size = data
        
        # Create config with the generated limit
        config = ResourceLimitConfig(max_response_size_bytes=limit)
        limiter = ResourceLimiter(config)
        
        # Checking a size within the limit should NOT raise an error
        # This should complete without exception
        limiter.check_response_size(size)

    @settings(max_examples=100)
    @given(limit=valid_response_size_limit())
    def test_property_4_exact_limit_passes(self, limit):
        """
        **Feature: agent-security-hardening, Property 4: Response Size Enforcement**
        
        For any response size exactly equal to the configured limit,
        check_response_size() SHALL NOT raise an error (boundary condition).
        
        **Validates: Requirements 3.1**
        """
        config = ResourceLimitConfig(max_response_size_bytes=limit)
        limiter = ResourceLimiter(config)
        
        # Checking a size exactly at the limit should NOT raise an error
        limiter.check_response_size(limit)

    @settings(max_examples=100)
    @given(limit=valid_response_size_limit())
    def test_property_4_one_byte_over_limit_raises(self, limit):
        """
        **Feature: agent-security-hardening, Property 4: Response Size Enforcement**
        
        For any response size exactly one byte over the configured limit,
        check_response_size() SHALL raise ResponseTooLargeError.
        
        **Validates: Requirements 3.1, 3.2**
        """
        config = ResourceLimitConfig(max_response_size_bytes=limit)
        limiter = ResourceLimiter(config)
        
        # Checking a size one byte over the limit should raise an error
        with pytest.raises(ResponseTooLargeError) as exc_info:
            limiter.check_response_size(limit + 1)
        
        assert exc_info.value.size == limit + 1
        assert exc_info.value.limit == limit

    @settings(max_examples=100)
    @given(
        limit=valid_response_size_limit(),
        sizes=st.lists(st.integers(min_value=0, max_value=200 * 1024 * 1024), min_size=1, max_size=10),
    )
    def test_property_4_consistent_enforcement_across_multiple_checks(self, limit, sizes):
        """
        **Feature: agent-security-hardening, Property 4: Response Size Enforcement**
        
        For any sequence of response size checks, the limiter SHALL consistently
        enforce the limit - sizes over limit always raise, sizes within always pass.
        
        **Validates: Requirements 3.1, 3.2**
        """
        config = ResourceLimitConfig(max_response_size_bytes=limit)
        limiter = ResourceLimiter(config)
        
        for size in sizes:
            if size > limit:
                # Should raise for sizes over limit
                with pytest.raises(ResponseTooLargeError):
                    limiter.check_response_size(size)
            else:
                # Should pass for sizes within limit
                limiter.check_response_size(size)

    @settings(max_examples=100)
    @given(data=limit_and_exceeding_size())
    def test_property_4_error_message_contains_size_info(self, data):
        """
        **Feature: agent-security-hardening, Property 4: Response Size Enforcement**
        
        For any response exceeding the limit, the error message SHALL contain
        both the actual size and the configured limit for debugging.
        
        **Validates: Requirements 3.2**
        """
        limit, size = data
        
        config = ResourceLimitConfig(max_response_size_bytes=limit)
        limiter = ResourceLimiter(config)
        
        with pytest.raises(ResponseTooLargeError) as exc_info:
            limiter.check_response_size(size)
        
        error_message = str(exc_info.value)
        
        # Error message should contain the actual size
        assert str(size) in error_message, (
            f"Error message should contain actual size {size}: {error_message}"
        )
        
        # Error message should contain the limit
        assert str(limit) in error_message, (
            f"Error message should contain limit {limit}: {error_message}"
        )

    @settings(max_examples=100)
    @given(limit=valid_response_size_limit())
    def test_property_4_max_response_size_property_matches_config(self, limit):
        """
        **Feature: agent-security-hardening, Property 4: Response Size Enforcement**
        
        The max_response_size property SHALL always return the configured limit.
        
        **Validates: Requirements 3.1**
        """
        config = ResourceLimitConfig(max_response_size_bytes=limit)
        limiter = ResourceLimiter(config)
        
        assert limiter.max_response_size == limit, (
            f"max_response_size {limiter.max_response_size} doesn't match config {limit}"
        )

    @settings(max_examples=100)
    @given(size=st.integers(min_value=0, max_value=20 * 1024 * 1024))
    def test_property_4_default_config_enforces_10mb_limit(self, size):
        """
        **Feature: agent-security-hardening, Property 4: Response Size Enforcement**
        
        With default configuration, the limiter SHALL enforce a 10MB limit.
        
        **Validates: Requirements 3.1**
        """
        # Default config has 10MB limit
        default_limit = 10 * 1024 * 1024
        limiter = ResourceLimiter()  # Uses default config
        
        assert limiter.max_response_size == default_limit
        
        if size > default_limit:
            with pytest.raises(ResponseTooLargeError):
                limiter.check_response_size(size)
        else:
            limiter.check_response_size(size)
