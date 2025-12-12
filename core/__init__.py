"""Core module for the blockchain stablecoin explorer agent."""

from core.database import (
    DatabaseConnection,
    init_database,
    get_database,
    close_database,
)
from core.db_manager import DatabaseManager, InvalidUUIDError, InvalidStatusError
from core.auth0_manager import (
    Auth0Manager,
    Auth0Error,
    TokenValidationError,
    TokenExpiredError,
    InsufficientPermissionsError,
    UserInfo,
    init_auth0,
    get_auth0_manager,
    close_auth0,
)
from core.orchestrator import AgentOrchestrator, CollectionReport, RunConfig

__all__ = [
    # Database
    "DatabaseConnection",
    "DatabaseManager",
    "InvalidUUIDError",
    "InvalidStatusError",
    "init_database",
    "get_database",
    "close_database",
    # Auth0
    "Auth0Manager",
    "Auth0Error",
    "TokenValidationError",
    "TokenExpiredError",
    "InsufficientPermissionsError",
    "UserInfo",
    "init_auth0",
    "get_auth0_manager",
    "close_auth0",
    # Orchestrator
    "AgentOrchestrator",
    "CollectionReport",
    "RunConfig",
]
