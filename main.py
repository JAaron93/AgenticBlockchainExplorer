"""
Main entry point for the blockchain stablecoin explorer application.

Initializes FastAPI app with authentication, database, and API routes.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from api.routes import auth_router, agent_router, results_router
from api.rate_limiter import RateLimitMiddleware, init_rate_limiter
from api.security import (
    init_cors_config,
    init_csrf_protection,
    get_cors_config,
)
from config.loader import ConfigurationManager
from config.models import Config
from core.auth0_manager import init_auth0, close_auth0
from core.database import init_database, close_database
from core.logging import configure_logging, get_logger

# Global config reference
_config: Config | None = None

# Deferred logger - will be configured after config is loaded
logger: logging.Logger | None = None


def get_config() -> Config:
    """Get the loaded configuration."""
    if _config is None:
        raise RuntimeError("Configuration not loaded")
    return _config


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown."""
    global _config, logger

    try:
        # Load configuration FIRST before any logging
        config_manager = ConfigurationManager()
        _config = config_manager.load_config()
        config_manager.validate_config(_config)

        # Configure structured logging with settings from config
        configure_logging(
            level=_config.logging.level,
            fmt=_config.logging.format,
            service_name="blockchain-stablecoin-explorer"
        )
        logger = get_logger(__name__)

        # Now we can log startup messages with the correct level
        logger.info("Starting Blockchain Stablecoin Explorer...")
        logger.info("Configuration loaded successfully", extra={
            "explorers": [e.name for e in _config.explorers],
            "stablecoins": list(_config.stablecoins.keys()),
            "environment": _config.app.env,
        })

        # Initialize database
        await init_database(_config.database)
        logger.info("Database connection established")

        # Initialize Auth0
        init_auth0(_config.auth0)
        logger.info("Auth0 manager initialized")

        # Initialize rate limiter
        init_rate_limiter(requests_per_minute=_config.rate_limit.per_minute)
        logger.info(
            f"Rate limiter initialized: {_config.rate_limit.per_minute} requests/minute"
        )

        # Initialize CORS configuration
        init_cors_config(
            allowed_origins=_config.cors.allowed_origins,
            allow_credentials=_config.cors.allow_credentials,
        )
        logger.info(f"CORS configured for origins: {_config.cors.allowed_origins}")

        # Initialize CSRF protection
        init_csrf_protection(secret_key=_config.app.secret_key)
        logger.info("CSRF protection initialized")

        # Initialize collector locks
        from collectors.base import init_collector_locks
        init_collector_locks()
        logger.info("Collector locks initialized")

        logger.info("Application startup complete")

    except Exception as e:
        # Ensure we have a logger even if config failed
        if logger is None:
            configure_logging()
            logger = get_logger(__name__)
        logger.error(f"Failed to start application: {e}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down application...")

    await close_auth0()
    await close_database()

    logger.info("Application shutdown complete")


app = FastAPI(
    title="Blockchain Stablecoin Explorer",
    description="""
## Overview

The Blockchain Stablecoin Explorer is an autonomous agent that collects and analyzes 
USDC and USDT stablecoin usage data from multiple blockchain networks.

## Features

- **Multi-chain Support**: Ethereum, BSC, and Polygon networks
- **Activity Classification**: Transactions, store of value, and other activities
- **Structured Output**: JSON format for easy analysis
- **Secure Access**: Auth0-based authentication and authorization

## Authentication

This API uses OAuth 2.0 with Auth0 for authentication. To access protected endpoints:

1. Navigate to `/login` to initiate the OAuth flow
2. After authentication, you'll receive an access token
3. Include the token in the `Authorization` header: `Bearer <token>`

## Permissions

| Permission | Description |
|------------|-------------|
| `run:agent` | Trigger data collection runs |
| `view:results` | View collection results |
| `download:data` | Download JSON outputs |
| `admin:config` | Administrative access |

## Rate Limiting

API requests are rate limited per user. See configuration for current limits.
""",
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "Health",
            "description": "Health check endpoints for monitoring and load balancers",
        },
        {
            "name": "Authentication",
            "description": "OAuth 2.0 authentication flow with Auth0",
        },
        {
            "name": "Agent Control",
            "description": "Trigger and monitor data collection runs",
        },
        {
            "name": "Results",
            "description": "Access and download collection results",
        },
    ],
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled errors."""
    if logger:
        logger.error(
            f"Unhandled exception: {exc}",
            exc_info=True,
            extra={
                "path": str(request.url.path),
                "method": request.method,
                "error_type": type(exc).__name__,
            }
        )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal error occurred"},
    )


# CORS middleware - configured after config is loaded
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    """Add CORS headers based on configuration."""
    cors_config = get_cors_config()
    
    # Handle preflight requests
    if request.method == "OPTIONS" and cors_config:
        origin = request.headers.get("origin")
        if origin and cors_config.is_origin_allowed(origin):
            allow_creds = str(cors_config.allow_credentials).lower()
            allow_methods = ", ".join(cors_config.allowed_methods)
            allow_headers = ", ".join(cors_config.allowed_headers)
            expose_headers = ", ".join(cors_config.expose_headers)
            return JSONResponse(
                status_code=200,
                content={},
                headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Credentials": allow_creds,
                    "Access-Control-Allow-Methods": allow_methods,
                    "Access-Control-Allow-Headers": allow_headers,
                    "Access-Control-Expose-Headers": expose_headers,
                    "Access-Control-Max-Age": str(cors_config.max_age),
                },
            )

    response = await call_next(request)

    # Get CORS config if available
    if cors_config:
        origin = request.headers.get("origin")
        if origin and cors_config.is_origin_allowed(origin):
            allow_creds = str(cors_config.allow_credentials).lower()
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = allow_creds
            response.headers["Access-Control-Allow-Methods"] = \
                ", ".join(cors_config.allowed_methods)
            response.headers["Access-Control-Allow-Headers"] = \
                ", ".join(cors_config.allowed_headers)
            response.headers["Access-Control-Expose-Headers"] = \
                ", ".join(cors_config.expose_headers)
    
    return response


# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)

    # Determine if we're in production
    is_production = _config and _config.app.env == "production"

    # HSTS - only in production with HTTPS
    if is_production:
        response.headers["Strict-Transport-Security"] = \
            "max-age=31536000; includeSubDomains"

    # Clickjacking protection
    response.headers["X-Frame-Options"] = "DENY"

    # MIME sniffing protection
    response.headers["X-Content-Type-Options"] = "nosniff"

    # XSS filter (legacy, but still useful)
    response.headers["X-XSS-Protection"] = "1; mode=block"

    # Content Security Policy
    csp = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )
    response.headers["Content-Security-Policy"] = csp

    # Referrer Policy
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # Permissions Policy
    response.headers["Permissions-Policy"] = (
        "accelerometer=(), camera=(), geolocation=(), "
        "gyroscope=(), magnetometer=(), microphone=(), "
        "payment=(), usb=()"
    )

    return response


# Add security middlewares (order matters - first added = last executed)
# Rate limiting should be checked early
app.add_middleware(RateLimitMiddleware)

# Security headers should be added to all responses
# Note: SecurityHeadersMiddleware is configured dynamically based on environment
# It will be added after config is loaded in lifespan


# Include routers
app.include_router(auth_router)
app.include_router(agent_router)
app.include_router(results_router)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - basic service info.
    
    Returns basic information about the service including name and version.
    """
    return {
        "status": "ok",
        "service": "Blockchain Stablecoin Explorer",
        "version": "1.0.0",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint for load balancers.
    
    Returns a simple healthy status. Use this for basic liveness checks.
    """
    return {"status": "healthy"}


@app.get("/health/ready", tags=["Health"])
async def readiness_check():
    """Readiness check - verifies all dependencies are available.
    
    Checks:
    - Database connectivity
    - Auth0 configuration
    - Configuration loaded
    """
    from core.database import get_database
    from core.auth0_manager import get_auth0_manager
    from sqlalchemy import text
    
    checks = {
        "database": "unknown",
        "auth0": "unknown",
        "config": "unknown",
    }
    all_healthy = True
    
    # Check configuration
    try:
        get_config()  # Verify config is loaded
        checks["config"] = "healthy"
    except Exception as e:
        checks["config"] = "unhealthy"
        all_healthy = False
    
    # Check database connectivity
    try:
        db = get_database()
        async with db.session() as session:
            await session.execute(text("SELECT 1"))
        checks["database"] = "healthy"
    except Exception as e:
        checks["database"] = f"unhealthy: {str(e)}"
        all_healthy = False
    
    # Check Auth0 configuration
    try:
        auth0 = get_auth0_manager()
        if auth0.domain and auth0.client_id:
            checks["auth0"] = "healthy"
        else:
            checks["auth0"] = "unhealthy: missing configuration"
            all_healthy = False
    except Exception as e:
        checks["auth0"] = f"unhealthy: {str(e)}"
        all_healthy = False
    
    status_code = 200 if all_healthy else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if all_healthy else "unhealthy",
            "checks": checks,
            "version": "1.0.0",
        }
    )


@app.get("/health/live", tags=["Health"])
async def liveness_check():
    """Liveness check - verifies the application is running.
    
    This is a simple check that always returns healthy if the app is running.
    Used by Kubernetes/container orchestrators to detect if the app needs restart.
    """
    return {"status": "alive"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
