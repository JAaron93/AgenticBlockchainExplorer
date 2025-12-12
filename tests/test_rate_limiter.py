"""Tests for rate limiting functionality."""

import asyncio
import pytest
from api.rate_limiter import RateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_allows_requests_under_limit():
    """Test that requests under the limit are allowed."""
    limiter = RateLimiter(requests_per_minute=10, window_seconds=60.0)

    # Make 5 requests - all should be allowed
    for i in range(5):
        allowed, remaining, retry_after = await limiter.is_allowed("test_user")
        assert allowed is True
        assert remaining == 10 - (i + 1)
        assert retry_after == 0


@pytest.mark.asyncio
async def test_rate_limiter_blocks_requests_over_limit():
    """Test that requests over the limit are blocked."""
    limiter = RateLimiter(requests_per_minute=5, window_seconds=60.0)

    # Make 5 requests - all should be allowed
    for _ in range(5):
        allowed, _, _ = await limiter.is_allowed("test_user")
        assert allowed is True

    # 6th request should be blocked
    allowed, remaining, retry_after = await limiter.is_allowed("test_user")
    assert allowed is False
    assert remaining == 0
    assert retry_after > 0


@pytest.mark.asyncio
async def test_rate_limiter_tracks_users_separately():
    """Test that different users have separate rate limits."""
    limiter = RateLimiter(requests_per_minute=2, window_seconds=60.0)

    # User 1 makes 2 requests
    for _ in range(2):
        allowed, _, _ = await limiter.is_allowed("user1")
        assert allowed is True

    # User 1 is now blocked
    allowed, _, _ = await limiter.is_allowed("user1")
    assert allowed is False

    # User 2 can still make requests
    allowed, remaining, _ = await limiter.is_allowed("user2")
    assert allowed is True
    assert remaining == 1


@pytest.mark.asyncio
async def test_rate_limiter_headers():
    """Test that rate limit headers are generated correctly."""
    limiter = RateLimiter(requests_per_minute=100, window_seconds=60.0)

    headers = limiter.get_headers(remaining=50, retry_after=0)
    assert headers["X-RateLimit-Limit"] == "100"
    assert headers["X-RateLimit-Remaining"] == "50"
    assert headers["X-RateLimit-Window"] == "60"
    assert "Retry-After" not in headers

    # With retry_after
    headers = limiter.get_headers(remaining=0, retry_after=30)
    assert headers["Retry-After"] == "30"
