"""Auth0 integration manager for token verification and permission checking.

Provides JWT token validation using python-jose and Auth0 JWKS.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import aiohttp
from jose import jwt, JWTError, ExpiredSignatureError
from jose.exceptions import JWTClaimsError

from config.models import Auth0Config

logger = logging.getLogger(__name__)


class Auth0Error(Exception):
    """Base exception for Auth0-related errors."""

    pass


class TokenValidationError(Auth0Error):
    """Raised when token validation fails."""

    def __init__(self, message: str, error_code: str = "invalid_token"):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


class TokenExpiredError(TokenValidationError):
    """Raised when the token has expired."""

    def __init__(self, message: str = "Token has expired"):
        super().__init__(message, error_code="token_expired")


class InsufficientPermissionsError(Auth0Error):
    """Raised when user lacks required permissions."""

    def __init__(self, required_permission: str, user_permissions: List[str]):
        self.required_permission = required_permission
        self.user_permissions = user_permissions
        message = f"Missing required permission: {required_permission}"
        super().__init__(message)


@dataclass
class UserInfo:
    """Represents authenticated user information from JWT token."""

    user_id: str  # Auth0 'sub' claim
    email: Optional[str] = None
    name: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    raw_claims: Dict[str, Any] = field(default_factory=dict)

    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions


class Auth0Manager:
    """Manages Auth0 authentication and authorization.

    Handles JWT token verification using Auth0's JWKS (JSON Web Key Set)
    and provides methods for permission checking.
    """

    def __init__(self, config: Auth0Config):
        """Initialize Auth0 manager.

        Args:
            config: Auth0 configuration containing domain, client_id, etc.
        """
        self._config = config
        self._jwks: Optional[Dict[str, Any]] = None
        self._jwks_uri = f"https://{config.domain}/.well-known/jwks.json"
        self._issuer = f"https://{config.domain}/"
        self._algorithms = ["RS256"]

    @property
    def domain(self) -> str:
        """Get the Auth0 domain."""
        return self._config.domain

    @property
    def client_id(self) -> str:
        """Get the Auth0 client ID."""
        return self._config.client_id

    @property
    def audience(self) -> str:
        """Get the Auth0 API audience."""
        return self._config.audience

    @property
    def callback_url(self) -> str:
        """Get the OAuth callback URL."""
        return str(self._config.callback_url)

    @property
    def logout_url(self) -> str:
        """Get the logout redirect URL."""
        return str(self._config.logout_url)

    async def _fetch_jwks(self) -> Dict[str, Any]:
        """Fetch JWKS from Auth0.

        Returns:
            JWKS dictionary containing public keys.

        Raises:
            Auth0Error: If JWKS cannot be fetched.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self._jwks_uri, timeout=10
                ) as response:
                    if response.status != 200:
                        raise Auth0Error(
                            f"Failed to fetch JWKS: HTTP {response.status}"
                        )
                    return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Failed to fetch JWKS from {self._jwks_uri}: {e}")
            raise Auth0Error(f"Failed to fetch JWKS: {e}") from e

    async def _get_jwks(self) -> Dict[str, Any]:
        """Get JWKS, fetching from Auth0 if not cached.

        Returns:
            JWKS dictionary.
        """
        if self._jwks is None:
            self._jwks = await self._fetch_jwks()
        return self._jwks

    async def _refresh_jwks(self) -> Dict[str, Any]:
        """Force refresh of JWKS cache.

        Returns:
            Fresh JWKS dictionary.
        """
        self._jwks = await self._fetch_jwks()
        return self._jwks

    def _get_signing_key(
        self, token: str, jwks: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract the signing key from JWKS that matches the token's kid.

        Args:
            token: JWT token string.
            jwks: JWKS dictionary.

        Returns:
            The matching key from JWKS.

        Raises:
            TokenValidationError: If no matching key is found.
        """
        try:
            unverified_header = jwt.get_unverified_header(token)
        except JWTError as e:
            raise TokenValidationError(f"Invalid token header: {e}")

        kid = unverified_header.get("kid")
        if not kid:
            raise TokenValidationError("Token header missing 'kid' claim")

        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                return key

        raise TokenValidationError(
            f"Unable to find matching key for kid: {kid}",
            error_code="invalid_key"
        )

    async def verify_token(self, token: str) -> UserInfo:
        """Verify a JWT token and extract user information.

        Args:
            token: JWT token string (without 'Bearer ' prefix).

        Returns:
            UserInfo object with user details and permissions.

        Raises:
            TokenValidationError: If token is invalid.
            TokenExpiredError: If token has expired.
        """
        if not token:
            raise TokenValidationError("Token is required")

        # Remove 'Bearer ' prefix if present
        if token.startswith("Bearer "):
            token = token[7:]

        jwks = await self._get_jwks()

        try:
            signing_key = self._get_signing_key(token, jwks)
        except TokenValidationError:
            # Key not found, try refreshing JWKS (key rotation)
            logger.info("Signing key not found, refreshing JWKS")
            jwks = await self._refresh_jwks()
            signing_key = self._get_signing_key(token, jwks)

        try:
            payload = jwt.decode(
                token,
                signing_key,
                algorithms=self._algorithms,
                audience=self._config.audience,
                issuer=self._issuer,
                options={
                    "verify_signature": True,
                    "verify_aud": True,
                    "verify_iss": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "require_exp": True,
                    "require_iat": True,
                }
            )
        except ExpiredSignatureError:
            logger.warning("Token has expired")
            raise TokenExpiredError()
        except JWTClaimsError as e:
            logger.warning(f"Token claims validation failed: {e}")
            raise TokenValidationError(f"Invalid token claims: {e}")
        except JWTError as e:
            logger.warning(f"Token validation failed: {e}")
            raise TokenValidationError(f"Token validation failed: {e}")

        # Extract user information from payload
        user_id = payload.get("sub")
        if not user_id:
            raise TokenValidationError("Token missing 'sub' claim")

        # Extract permissions from token
        # Auth0 includes these in access tokens
        permissions = payload.get("permissions", [])

        # Extract optional user info claims
        email = payload.get("email") or payload.get(f"{self._issuer}email")
        name = payload.get("name") or payload.get(f"{self._issuer}name")

        logger.debug(f"Token verified for user: {user_id}")

        return UserInfo(
            user_id=user_id,
            email=email,
            name=name,
            permissions=permissions,
            raw_claims=payload
        )

    def get_user_permissions(self, user_info: UserInfo) -> List[str]:
        """Get permissions for a user from their token info.

        Args:
            user_info: UserInfo object from verify_token.

        Returns:
            List of permission strings.
        """
        return user_info.permissions

    def check_permission(
        self,
        user_info: UserInfo,
        permission: str
    ) -> bool:
        """Check if a user has a specific permission.

        Args:
            user_info: UserInfo object from verify_token.
            permission: Permission string to check.

        Returns:
            True if user has the permission, False otherwise.
        """
        return user_info.has_permission(permission)

    def require_permission(
        self,
        user_info: UserInfo,
        permission: str
    ) -> None:
        """Require a user to have a specific permission.

        Args:
            user_info: UserInfo object from verify_token.
            permission: Required permission string.

        Raises:
            InsufficientPermissionsError: If user lacks the permission.
        """
        if not self.check_permission(user_info, permission):
            logger.warning(
                f"User {user_info.user_id} lacks permission: {permission}. "
                f"Has: {user_info.permissions}"
            )
            raise InsufficientPermissionsError(
                permission, user_info.permissions
            )

    def get_authorization_url(self, state: Optional[str] = None) -> str:
        """Generate Auth0 authorization URL for login.

        Args:
            state: Optional state parameter for CSRF protection.

        Returns:
            Authorization URL to redirect user to.
        """
        params = {
            "response_type": "code",
            "client_id": self._config.client_id,
            "redirect_uri": str(self._config.callback_url),
            "scope": "openid profile email",
            "audience": self._config.audience,
        }

        if state:
            params["state"] = state

        query_string = urlencode(params)
        return f"https://{self._config.domain}/authorize?{query_string}"

    def get_logout_url(self, return_to: Optional[str] = None) -> str:
        """Generate Auth0 logout URL.

        Args:
            return_to: URL to redirect to after logout.

        Returns:
            Logout URL to redirect user to.
        """
        return_url = return_to or str(self._config.logout_url)
        params = {
            "client_id": self._config.client_id,
            "returnTo": return_url,
        }
        query_string = urlencode(params)
        return f"https://{self._config.domain}/v2/logout?{query_string}"

    async def exchange_code_for_tokens(
        self,
        code: str
    ) -> Dict[str, Any]:
        """Exchange authorization code for tokens.

        Args:
            code: Authorization code from Auth0 callback.

        Returns:
            Dictionary containing access_token, id_token, etc.

        Raises:
            Auth0Error: If token exchange fails.
        """
        token_url = f"https://{self._config.domain}/oauth/token"

        payload = {
            "grant_type": "authorization_code",
            "client_id": self._config.client_id,
            "client_secret": self._config.client_secret,
            "code": code,
            "redirect_uri": str(self._config.callback_url),
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    token_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                ) as response:
                    data = await response.json()

                    if response.status != 200:
                        error = data.get("error", "unknown_error")
                        error_desc = data.get(
                            "error_description",
                            "Token exchange failed"
                        )
                        logger.error(
                            f"Token exchange failed: {error} - {error_desc}"
                        )
                        raise Auth0Error(
                            f"Token exchange failed: {error_desc}"
                        )

                    return data
        except aiohttp.ClientError as e:
            logger.error(f"Token exchange request failed: {e}")
            raise Auth0Error(f"Token exchange request failed: {e}") from e


# Global Auth0 manager instance
_auth0_manager: Optional[Auth0Manager] = None


def get_auth0_manager() -> Auth0Manager:
    """Get the global Auth0 manager instance.

    Returns:
        Auth0Manager instance.

    Raises:
        RuntimeError: If Auth0 manager has not been initialized.
    """
    if _auth0_manager is None:
        raise RuntimeError(
            "Auth0 manager not initialized. Call init_auth0() first."
        )
    return _auth0_manager


def init_auth0(config: Auth0Config) -> Auth0Manager:
    """Initialize the global Auth0 manager.

    Args:
        config: Auth0 configuration.

    Returns:
        Initialized Auth0Manager instance.
    """
    global _auth0_manager
    _auth0_manager = Auth0Manager(config)
    logger.info(f"Auth0 manager initialized for domain: {config.domain}")
    return _auth0_manager


async def close_auth0() -> None:
    """Close the global Auth0 manager and clean up resources."""
    global _auth0_manager
    if _auth0_manager is not None:
        # Clear any cached data
        _auth0_manager._jwks = None
    _auth0_manager = None
    logger.info("Auth0 manager closed")
