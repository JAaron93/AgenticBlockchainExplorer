"""Rate limiting middleware for FastAPI.

Implements per-user rate limiting based on configuration.
Uses in-memory storage for simplicity (use Redis in production for distributed systems).
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


@dataclass
class RateLimitState:
    """Tracks rate limit state for a single user/key."""
    
    requests: list = field(default_factory=list)
    
    def add_request(self, timestamp: float) -> None:
        """Add a request timestamp."""
        self.requests.append(timestamp)
    
    def cleanup_old_requests(self, window_seconds: float, current_time: float) -> None:
        """Remove requests older than the window."""
        cutoff = current_time - window_seconds
        self.requests = [ts for ts in self.requests if ts > cutoff]
    
    def request_count(self) -> int:
        """Get current request count."""
        return len(self.requests)


class RateLimiter:
    """In-memory rate limiter with sliding window algorithm.
    
    Tracks requests per user/key and enforces rate limits.
    Thread-safe for async operations.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 100,
        window_seconds: float = 60.0,
    ):
        """Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests allowed per window.
            window_seconds: Time window in seconds (default: 60).
        """
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self._states: Dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._lock = asyncio.Lock()
        self._cleanup_counter = 0
        self._cleanup_interval = 100  # Cleanup every N requests
    
    async def is_allowed(self, key: str) -> tuple[bool, int, int]:
        """Check if a request is allowed for the given key.
        
        Args:
            key: Unique identifier (e.g., user_id, IP address).
            
        Returns:
            Tuple of (allowed, remaining_requests, retry_after_seconds).
        """
        current_time = time.time()
        
        async with self._lock:
            state = self._states[key]
            
            # Cleanup old requests
            state.cleanup_old_requests(self.window_seconds, current_time)
            
            # Check if under limit
            current_count = state.request_count()
            
            if current_count >= self.requests_per_minute:
                # Calculate retry-after based on oldest request in window
                if state.requests:
                    oldest = min(state.requests)
                    retry_after = int(oldest + self.window_seconds - current_time) + 1
                else:
                    retry_after = int(self.window_seconds)
                
                return False, 0, max(1, retry_after)
            
            # Allow request and record it
            state.add_request(current_time)
            remaining = self.requests_per_minute - state.request_count()
            
            # Periodic cleanup of stale entries
            self._cleanup_counter += 1
            if self._cleanup_counter >= self._cleanup_interval:
                self._cleanup_counter = 0
                await self._cleanup_stale_entries(current_time)
            
            return True, remaining, 0
    
    async def _cleanup_stale_entries(self, current_time: float) -> None:
        """Remove entries with no recent requests."""
        cutoff = current_time - self.window_seconds * 2
        stale_keys = [
            key for key, state in self._states.items()
            if not state.requests or max(state.requests) < cutoff
        ]
        for key in stale_keys:
            del self._states[key]
        
        if stale_keys:
            logger.debug(f"Cleaned up {len(stale_keys)} stale rate limit entries")
    
    def get_headers(self, remaining: int, retry_after: int = 0) -> Dict[str, str]:
        """Get rate limit headers for response.
        
        Args:
            remaining: Remaining requests in window.
            retry_after: Seconds until rate limit resets (if limited).
            
        Returns:
            Dictionary of rate limit headers.
        """
        headers = {
            "X-RateLimit-Limit": str(self.requests_per_minute),
            "X-RateLimit-Remaining": str(max(0, remaining)),
            "X-RateLimit-Window": str(int(self.window_seconds)),
        }
        if retry_after > 0:
            headers["Retry-After"] = str(retry_after)
        return headers


# Global rate limiter instance (initialized on app startup)
_rate_limiter: Optional[RateLimiter] = None


def init_rate_limiter(requests_per_minute: int = 100) -> RateLimiter:
    """Initialize the global rate limiter.
    
    Args:
        requests_per_minute: Maximum requests per minute per user.
        
    Returns:
        The initialized RateLimiter instance.
    """
    global _rate_limiter
    _rate_limiter = RateLimiter(requests_per_minute=requests_per_minute)
    logger.info(f"Rate limiter initialized: {requests_per_minute} requests/minute")
    return _rate_limiter


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance.
    
    Returns:
        The RateLimiter instance.
        
    Raises:
        RuntimeError: If rate limiter not initialized.
    """
    if _rate_limiter is None:
        raise RuntimeError("Rate limiter not initialized")
    return _rate_limiter


def _extract_rate_limit_key(request: Request) -> str:
    """Extract the rate limit key from a request.
    
    Uses user_id if authenticated, otherwise falls back to IP address.
    
    Args:
        request: FastAPI request object.
        
    Returns:
        Rate limit key string.
    """
    # Try to get user_id from request state (set by auth middleware)
    if hasattr(request.state, "user") and request.state.user:
        return f"user:{request.state.user.user_id}"
    
    # Fall back to IP address
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        ip = forwarded_for.split(",")[0].strip()
    elif request.client:
        ip = request.client.host
    else:
        ip = "unknown"
    
    return f"ip:{ip}"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting.
    
    Applies rate limiting to all requests based on user ID or IP address.
    Excludes health check and static endpoints.
    """
    
    # Endpoints excluded from rate limiting
    EXCLUDED_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json"}
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Process request with rate limiting.
        
        Args:
            request: Incoming request.
            call_next: Next middleware/handler in chain.
            
        Returns:
            Response with rate limit headers.
        """
        # Skip rate limiting for excluded paths
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)
        
        # Skip rate limiting for OPTIONS (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        try:
            rate_limiter = get_rate_limiter()
        except RuntimeError:
            # Rate limiter not initialized, skip limiting
            return await call_next(request)
        
        # Extract rate limit key
        key = _extract_rate_limit_key(request)
        
        # Check rate limit
        allowed, remaining, retry_after = await rate_limiter.is_allowed(key)
        
        if not allowed:
            logger.warning(
                f"Rate limit exceeded for {key}",
                extra={
                    "key": key,
                    "path": request.url.path,
                    "method": request.method,
                    "retry_after": retry_after,
                }
            )
            
            headers = rate_limiter.get_headers(remaining, retry_after)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                headers=headers,
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        headers = rate_limiter.get_headers(remaining)
        for header, value in headers.items():
            response.headers[header] = value
        
        return response
