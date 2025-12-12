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
from config.loader import ConfigurationManager
from config.models import Config
from core.auth0_manager import init_auth0, close_auth0
from core.database import init_database, close_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global config reference
_config: Config | None = None


def get_config() -> Config:
    """Get the loaded configuration."""
    if _config is None:
        raise RuntimeError("Configuration not loaded")
    return _config


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown."""
    global _config
    
    # Startup
    logger.info("Starting Blockchain Stablecoin Explorer...")
    
    try:
        # Load configuration
        config_manager = ConfigurationManager()
        _config = config_manager.load_config()
        config_manager.validate_config(_config)
        logger.info("Configuration loaded successfully")
        
        # Initialize database
        await init_database(_config.database)
        logger.info("Database connection established")
        
        # Initialize Auth0
        init_auth0(_config.auth0)
        logger.info("Auth0 manager initialized")
        
        # Configure logging level from config
        log_level = getattr(logging, _config.logging.level, logging.INFO)
        logging.getLogger().setLevel(log_level)
        
        logger.info("Application startup complete")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    
    await close_auth0()
    await close_database()
    
    logger.info("Application shutdown complete")


app = FastAPI(
    title="Blockchain Stablecoin Explorer",
    description="Autonomous agent for collecting and analyzing stablecoin usage data",
    version="1.0.0",
    lifespan=lifespan,
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal error occurred"},
    )


# CORS middleware - configured after config is loaded
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    """Add CORS headers based on configuration."""
    response = await call_next(request)
    
    # Get CORS config if available
    if _config:
        origin = request.headers.get("origin")
        if origin and origin in _config.cors.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
    
    return response


# Include routers
app.include_router(auth_router)
app.include_router(agent_router)
app.include_router(results_router)


@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {
        "status": "ok",
        "service": "Blockchain Stablecoin Explorer",
        "version": "1.0.0",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
