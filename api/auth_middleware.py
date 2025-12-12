"""FastAPI authentication middleware for Auth0 integration.

Provides dependencies for protecting endpoints with JWT authentication
and permission-based authorization.
"""

import logging
from typing import Callable, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from core.auth0_manager import (
    Auth0Manager,
    UserInfo,
    TokenValidationError,
    TokenExpiredError,
    InsufficientPermissionsError,
    get_auth0_manager,
)

logger = logging.getLogger(__name__)

# HTTP Bearer token security scheme
security = HTTPBearer(auto_error=False)


class AuthenticationError(HTTPException):
    """HTTP 401 Unauthorized exception."""

    def __init__(
        self,
        detail: str = "Authentication required",
        headers: Optional[dict] = None
    ):
        if headers is None:
            headers = {"WWW-Authenticate": "Bearer"}
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers=headers
        )


class AuthorizationError(HTTPException):
    """HTTP 403 Forbidden exception."""

    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail
        )


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth0_manager: Auth0Manager = Depends(get_auth0_manager)
) -> UserInfo:
    """FastAPI dependency to get the current authenticated user.

    Extracts and validates the JWT token from the Authorization header.

    Args:
        request: FastAPI request object.
        credentials: HTTP Bearer credentials from Authorization header.
        auth0_manager: Auth0 manager instance.

    Returns:
        UserInfo object with user details and permissions.

    Raises:
        AuthenticationError: If token is missing, invalid, or expired.
    """
    # Check for token in Authorization header
    if credentials is None:
        logger.warning(
            f"Missing authorization header for {request.method} {request.url}"
        )
        raise AuthenticationError("Missing authorization token")

    token = credentials.credentials

    try:
        user_info = await auth0_manager.verify_token(token)

        # Store user info in request state for later use
        request.state.user = user_info

        logger.debug(
            f"Authenticated user {user_info.user_id} "
            f"for {request.method} {request.url}"
        )

        return user_info

    except TokenExpiredError:
        logger.warning(
            f"Expired token for {request.method} {request.url}"
        )
        raise AuthenticationError(
            "Token has expired",
            headers={"WWW-Authenticate": "Bearer error=\"invalid_token\""}
        )

    except TokenValidationError as e:
        logger.warning(
            f"Invalid token for {request.method} {request.url}: {e.message}"
        )
        raise AuthenticationError(
            f"Invalid token: {e.message}",
            headers={
                "WWW-Authenticate": f"Bearer error=\"{e.error_code}\""
            }
        )


def requires_auth(
    permission: Optional[str] = None
) -> Callable:
    """Dependency factory for authentication with optional permission.

    This can be used as a dependency to protect endpoints:

    Example:
        @app.get("/protected")
        async def protected_endpoint(
            user: UserInfo = Depends(requires_auth())
        ):
            return {"user_id": user.user_id}

        @app.post("/admin")
        async def admin_endpoint(
            user: UserInfo = Depends(requires_auth("admin:config"))
        ):
            return {"user_id": user.user_id}

    Args:
        permission: Optional permission string required to access the endpoint.

    Returns:
        FastAPI dependency function.
    """
    async def dependency(
        user: UserInfo = Depends(get_current_user),
        auth0_manager: Auth0Manager = Depends(get_auth0_manager)
    ) -> UserInfo:
        if permission:
            if not auth0_manager.check_permission(user, permission):
                logger.warning(
                    f"User {user.user_id} lacks permission: {permission}"
                )
                raise AuthorizationError(
                    f"Missing required permission: {permission}"
                )
        return user

    return dependency


def requires_permission(permission: str) -> Callable:
    """Dependency factory for requiring a specific permission.

    Shorthand for requires_auth(permission=permission).

    Example:
        @app.post("/api/agent/run")
        async def run_agent(
            user: UserInfo = Depends(requires_permission("run:agent"))
        ):
            return {"status": "started"}

    Args:
        permission: Permission string required to access the endpoint.

    Returns:
        FastAPI dependency function.
    """
    return requires_auth(permission=permission)


class PermissionChecker:
    """Callable class for permission checking as a dependency.

    This provides an alternative way to check permissions that can be
    used with FastAPI's dependency injection.

    Example:
        run_agent_permission = PermissionChecker("run:agent")

        @app.post("/api/agent/run")
        async def run_agent(
            user: UserInfo = Depends(run_agent_permission)
        ):
            return {"status": "started"}
    """

    def __init__(self, permission: str):
        """Initialize permission checker.

        Args:
            permission: Required permission string.
        """
        self.permission = permission

    async def __call__(
        self,
        user: UserInfo = Depends(get_current_user),
        auth0_manager: Auth0Manager = Depends(get_auth0_manager)
    ) -> UserInfo:
        """Check if user has the required permission.

        Args:
            user: Authenticated user info.
            auth0_manager: Auth0 manager instance.

        Returns:
            UserInfo if permission check passes.

        Raises:
            AuthorizationError: If user lacks the required permission.
        """
        try:
            auth0_manager.require_permission(user, self.permission)
            return user
        except InsufficientPermissionsError:
            raise AuthorizationError(
                f"Missing required permission: {self.permission}"
            )


# Pre-defined permission checkers for common permissions
run_agent_permission = PermissionChecker("run:agent")
view_results_permission = PermissionChecker("view:results")
download_data_permission = PermissionChecker("download:data")
admin_config_permission = PermissionChecker("admin:config")


async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth0_manager: Auth0Manager = Depends(get_auth0_manager)
) -> Optional[UserInfo]:
    """FastAPI dependency to optionally get the current user.

    Unlike get_current_user, this does not raise an error if no token
    is provided. Useful for endpoints that have different behavior
    for authenticated vs anonymous users.

    Args:
        request: FastAPI request object.
        credentials: HTTP Bearer credentials from Authorization header.
        auth0_manager: Auth0 manager instance.

    Returns:
        UserInfo if authenticated, None otherwise.
    """
    if credentials is None:
        return None

    try:
        user_info = await auth0_manager.verify_token(credentials.credentials)
        request.state.user = user_info
        return user_info
    except (TokenValidationError, TokenExpiredError):
        return None


def get_client_info(request: Request) -> dict:
    """Extract client information from request for audit logging.

    Args:
        request: FastAPI request object.

    Returns:
        Dictionary with ip_address and user_agent.
    """
    # Get client IP, considering X-Forwarded-For header for proxied requests
    forwarded_for = request.headers.get("X-Forwarded-For")
    ip_address: Optional[str] = None
    if forwarded_for:
        # Take the first IP in the chain (original client)
        ip_address = forwarded_for.split(",")[0].strip()
    elif request.client:
        ip_address = request.client.host

    user_agent = request.headers.get("User-Agent")

    return {
        "ip_address": ip_address,
        "user_agent": user_agent
    }
