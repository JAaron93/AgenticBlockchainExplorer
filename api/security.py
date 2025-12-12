"""Security middleware for FastAPI.

Implements security headers, CORS configuration, and CSRF protection.
"""

import hashlib
import hmac
import logging
import secrets
import time
from typing import Callable, Dict, List, Optional, Set

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses.
    
    Adds headers for:
    - HSTS (HTTP Strict Transport Security)
    - X-Frame-Options (clickjacking protection)
    - X-Content-Type-Options (MIME sniffing protection)
    - X-XSS-Protection (XSS filter)
    - Content-Security-Policy (CSP)
    - Referrer-Policy
    - Permissions-Policy
    """
    
    def __init__(
        self,
        app,
        hsts_max_age: int = 31536000,  # 1 year
        include_subdomains: bool = True,
        frame_options: str = "DENY",
        content_type_options: str = "nosniff",
        xss_protection: str = "1; mode=block",
        referrer_policy: str = "strict-origin-when-cross-origin",
        csp_directives: Optional[Dict[str, str]] = None,
        is_production: bool = False,
    ):
        """Initialize security headers middleware.
        
        Args:
            app: FastAPI application.
            hsts_max_age: HSTS max-age in seconds.
            include_subdomains: Include subdomains in HSTS.
            frame_options: X-Frame-Options value.
            content_type_options: X-Content-Type-Options value.
            xss_protection: X-XSS-Protection value.
            referrer_policy: Referrer-Policy value.
            csp_directives: Custom CSP directives.
            is_production: Whether running in production (enables HSTS).
        """
        super().__init__(app)
        self.hsts_max_age = hsts_max_age
        self.include_subdomains = include_subdomains
        self.frame_options = frame_options
        self.content_type_options = content_type_options
        self.xss_protection = xss_protection
        self.referrer_policy = referrer_policy
        self.is_production = is_production
        
        # Build CSP header
        default_csp = {
            "default-src": "'self'",
            "script-src": "'self'",
            "style-src": "'self' 'unsafe-inline'",
            "img-src": "'self' data: https:",
            "font-src": "'self'",
            "connect-src": "'self'",
            "frame-ancestors": "'none'",
            "base-uri": "'self'",
            "form-action": "'self'",
        }
        if csp_directives:
            default_csp.update(csp_directives)
        
        self.csp_header = "; ".join(
            f"{key} {value}" for key, value in default_csp.items()
        )
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Add security headers to response.
        
        Args:
            request: Incoming request.
            call_next: Next middleware/handler.
            
        Returns:
            Response with security headers.
        """
        response = await call_next(request)
        
        # HSTS - only in production with HTTPS
        if self.is_production:
            hsts_value = f"max-age={self.hsts_max_age}"
            if self.include_subdomains:
                hsts_value += "; includeSubDomains"
            response.headers["Strict-Transport-Security"] = hsts_value
        
        # Clickjacking protection
        response.headers["X-Frame-Options"] = self.frame_options
        
        # MIME sniffing protection
        response.headers["X-Content-Type-Options"] = self.content_type_options
        
        # XSS filter (legacy, but still useful)
        response.headers["X-XSS-Protection"] = self.xss_protection
        
        # Content Security Policy
        response.headers["Content-Security-Policy"] = self.csp_header
        
        # Referrer Policy
        response.headers["Referrer-Policy"] = self.referrer_policy
        
        # Permissions Policy (formerly Feature-Policy)
        response.headers["Permissions-Policy"] = (
            "accelerometer=(), camera=(), geolocation=(), "
            "gyroscope=(), magnetometer=(), microphone=(), "
            "payment=(), usb=()"
        )
        
        return response


class CORSConfig:
    """Configuration for CORS middleware."""
    
    def __init__(
        self,
        allowed_origins: List[str],
        allow_credentials: bool = True,
        allowed_methods: Optional[List[str]] = None,
        allowed_headers: Optional[List[str]] = None,
        expose_headers: Optional[List[str]] = None,
        max_age: int = 600,
    ):
        """Initialize CORS configuration.
        
        Args:
            allowed_origins: List of allowed origins.
            allow_credentials: Allow credentials (cookies, auth headers).
            allowed_methods: Allowed HTTP methods.
            allowed_headers: Allowed request headers.
            expose_headers: Headers to expose to browser.
            max_age: Preflight cache duration in seconds.
        """
        self.allowed_origins: Set[str] = set(allowed_origins)
        self.allow_credentials = allow_credentials
        self.allowed_methods = allowed_methods or [
            "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"
        ]
        self.allowed_headers = allowed_headers or [
            "Authorization", "Content-Type", "X-CSRF-Token",
            "X-Requested-With", "Accept", "Origin"
        ]
        self.expose_headers = expose_headers or [
            "X-RateLimit-Limit", "X-RateLimit-Remaining",
            "X-RateLimit-Window", "Retry-After"
        ]
        self.max_age = max_age
    
    def is_origin_allowed(self, origin: Optional[str]) -> bool:
        """Check if an origin is allowed.
        
        Args:
            origin: Origin header value.
            
        Returns:
            True if origin is allowed.
        """
        if not origin:
            return False
        return origin in self.allowed_origins


# Global CORS config (initialized on app startup)
_cors_config: Optional[CORSConfig] = None


def init_cors_config(
    allowed_origins: List[str],
    allow_credentials: bool = True,
) -> CORSConfig:
    """Initialize global CORS configuration.
    
    Args:
        allowed_origins: List of allowed origins.
        allow_credentials: Allow credentials.
        
    Returns:
        The initialized CORSConfig.
    """
    global _cors_config
    _cors_config = CORSConfig(
        allowed_origins=allowed_origins,
        allow_credentials=allow_credentials,
    )
    logger.info(f"CORS configured for origins: {allowed_origins}")
    return _cors_config


def get_cors_config() -> Optional[CORSConfig]:
    """Get the global CORS configuration."""
    return _cors_config


class CSRFProtection:
    """CSRF protection using double-submit cookie pattern.
    
    Generates and validates CSRF tokens for state-changing requests.
    """
    
    # Methods that require CSRF protection
    PROTECTED_METHODS = {"POST", "PUT", "DELETE", "PATCH"}
    
    # Paths excluded from CSRF protection (e.g., API endpoints with JWT auth)
    EXCLUDED_PATHS = {"/callback", "/api/"}
    
    def __init__(self, secret_key: str, token_lifetime: int = 3600):
        """Initialize CSRF protection.
        
        Args:
            secret_key: Secret key for token signing.
            token_lifetime: Token lifetime in seconds.
        """
        self.secret_key = secret_key.encode()
        self.token_lifetime = token_lifetime
    
    def generate_token(self) -> str:
        """Generate a new CSRF token.
        
        Returns:
            CSRF token string.
        """
        # Generate random token with timestamp
        random_bytes = secrets.token_bytes(32)
        timestamp = str(int(time.time())).encode()
        
        # Create HMAC signature
        message = random_bytes + b":" + timestamp
        signature = hmac.new(self.secret_key, message, hashlib.sha256).hexdigest()
        
        # Combine into token
        token = f"{random_bytes.hex()}:{timestamp.decode()}:{signature}"
        return token
    
    def validate_token(self, token: str) -> bool:
        """Validate a CSRF token.
        
        Args:
            token: Token to validate.
            
        Returns:
            True if token is valid.
        """
        try:
            parts = token.split(":")
            if len(parts) != 3:
                return False
            
            random_hex, timestamp_str, signature = parts
            random_bytes = bytes.fromhex(random_hex)
            timestamp = int(timestamp_str)
            
            # Check token age
            if time.time() - timestamp > self.token_lifetime:
                logger.debug("CSRF token expired")
                return False
            
            # Verify signature
            message = random_bytes + b":" + timestamp_str.encode()
            expected_signature = hmac.new(
                self.secret_key, message, hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                logger.debug("CSRF token signature mismatch")
                return False
            
            return True
            
        except (ValueError, TypeError) as e:
            logger.debug(f"CSRF token validation error: {e}")
            return False
    
    def should_protect(self, request: Request) -> bool:
        """Check if request should be CSRF protected.
        
        Args:
            request: FastAPI request.
            
        Returns:
            True if CSRF protection should be applied.
        """
        # Only protect state-changing methods
        if request.method not in self.PROTECTED_METHODS:
            return False
        
        # Skip API endpoints that use JWT authentication
        path = request.url.path
        for excluded in self.EXCLUDED_PATHS:
            if path.startswith(excluded):
                return False
        
        return True


# Global CSRF protection instance
_csrf_protection: Optional[CSRFProtection] = None


def init_csrf_protection(secret_key: str) -> CSRFProtection:
    """Initialize global CSRF protection.
    
    Args:
        secret_key: Secret key for token signing.
        
    Returns:
        The initialized CSRFProtection.
    """
    global _csrf_protection
    _csrf_protection = CSRFProtection(secret_key=secret_key)
    logger.info("CSRF protection initialized")
    return _csrf_protection


def get_csrf_protection() -> Optional[CSRFProtection]:
    """Get the global CSRF protection instance."""
    return _csrf_protection


class CSRFMiddleware(BaseHTTPMiddleware):
    """Middleware for CSRF protection.
    
    Validates CSRF tokens for state-changing requests on non-API endpoints.
    API endpoints are expected to use JWT authentication instead.
    """
    
    CSRF_HEADER = "X-CSRF-Token"
    CSRF_COOKIE = "csrf_token"
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Process request with CSRF validation.
        
        Args:
            request: Incoming request.
            call_next: Next middleware/handler.
            
        Returns:
            Response.
        """
        csrf = get_csrf_protection()
        
        if csrf is None:
            return await call_next(request)
        
        # Check if request needs CSRF protection
        if csrf.should_protect(request):
            # Get token from header
            token = request.headers.get(self.CSRF_HEADER)
            
            if not token:
                logger.warning(
                    f"Missing CSRF token for {request.method} {request.url.path}",
                    extra={"path": request.url.path, "method": request.method}
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="CSRF token missing",
                )
            
            if not csrf.validate_token(token):
                logger.warning(
                    f"Invalid CSRF token for {request.method} {request.url.path}",
                    extra={"path": request.url.path, "method": request.method}
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid CSRF token",
                )
        
        response = await call_next(request)
        
        # Set new CSRF token cookie for GET requests
        if request.method == "GET" and csrf:
            new_token = csrf.generate_token()
            response.set_cookie(
                key=self.CSRF_COOKIE,
                value=new_token,
                httponly=False,  # JavaScript needs to read this
                secure=True,
                samesite="lax",
                max_age=csrf.token_lifetime,
            )
        
        return response
