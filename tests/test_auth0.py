"""Unit tests for Auth0 token validation with mock tokens."""

import time
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from core.auth0_manager import (
    Auth0Manager,
    UserInfo,
    TokenValidationError,
    TokenExpiredError,
    InsufficientPermissionsError,
    Auth0Error,
    init_auth0,
    get_auth0_manager,
    close_auth0,
)
from config.models import Auth0Config


@pytest.fixture
def auth0_config():
    """Create a test Auth0 configuration."""
    return Auth0Config(
        domain="test.auth0.com",
        client_id="test_client_id",
        client_secret="test_client_secret",
        audience="https://test-api",
        callback_url="http://localhost:8000/callback",
        logout_url="http://localhost:8000",
    )


@pytest.fixture
def auth0_manager(auth0_config):
    """Create an Auth0Manager instance."""
    return Auth0Manager(auth0_config)


@pytest.fixture
def mock_jwks():
    """Create mock JWKS response."""
    return {
        "keys": [
            {
                "kty": "RSA",
                "kid": "test-key-id",
                "use": "sig",
                "n": "test-modulus",
                "e": "AQAB",
            }
        ]
    }


@pytest.fixture
def user_info():
    """Create a test UserInfo object."""
    return UserInfo(
        user_id="auth0|123456789",
        email="test@example.com",
        name="Test User",
        permissions=["run:agent", "view:results"],
        raw_claims={"sub": "auth0|123456789"},
    )


class TestUserInfo:
    """Tests for UserInfo dataclass."""

    def test_has_permission_returns_true_for_existing(self, user_info):
        """has_permission returns True for existing permission."""
        assert user_info.has_permission("run:agent") is True
        assert user_info.has_permission("view:results") is True

    def test_has_permission_returns_false_for_missing(self, user_info):
        """has_permission returns False for missing permission."""
        assert user_info.has_permission("admin:config") is False
        assert user_info.has_permission("nonexistent") is False

    def test_user_info_attributes(self, user_info):
        """UserInfo has correct attributes."""
        assert user_info.user_id == "auth0|123456789"
        assert user_info.email == "test@example.com"
        assert user_info.name == "Test User"
        assert len(user_info.permissions) == 2


class TestAuth0ManagerProperties:
    """Tests for Auth0Manager properties."""

    def test_domain_property(self, auth0_manager, auth0_config):
        """domain property returns correct value."""
        assert auth0_manager.domain == auth0_config.domain

    def test_client_id_property(self, auth0_manager, auth0_config):
        """client_id property returns correct value."""
        assert auth0_manager.client_id == auth0_config.client_id

    def test_audience_property(self, auth0_manager, auth0_config):
        """audience property returns correct value."""
        assert auth0_manager.audience == auth0_config.audience

    def test_callback_url_property(self, auth0_manager, auth0_config):
        """callback_url property returns correct value."""
        assert auth0_manager.callback_url == str(auth0_config.callback_url)

    def test_logout_url_property(self, auth0_manager, auth0_config):
        """logout_url property returns correct value."""
        assert auth0_manager.logout_url == str(auth0_config.logout_url)


class TestGetAuthorizationUrl:
    """Tests for get_authorization_url method."""

    def test_authorization_url_contains_required_params(self, auth0_manager):
        """Authorization URL contains all required parameters."""
        url = auth0_manager.get_authorization_url()

        assert "https://test.auth0.com/authorize" in url
        assert "response_type=code" in url
        assert "client_id=test_client_id" in url
        assert "redirect_uri=" in url
        assert "scope=openid+profile+email" in url
        assert "audience=" in url

    def test_authorization_url_with_state(self, auth0_manager):
        """Authorization URL includes state parameter when provided."""
        url = auth0_manager.get_authorization_url(state="csrf-token-123")

        assert "state=csrf-token-123" in url


class TestGetLogoutUrl:
    """Tests for get_logout_url method."""

    def test_logout_url_format(self, auth0_manager):
        """Logout URL has correct format."""
        url = auth0_manager.get_logout_url()

        assert "https://test.auth0.com/v2/logout" in url
        assert "client_id=test_client_id" in url
        assert "returnTo=" in url

    def test_logout_url_with_custom_return(self, auth0_manager):
        """Logout URL uses custom return URL when provided."""
        url = auth0_manager.get_logout_url(return_to="https://custom.com")

        assert "returnTo=https%3A%2F%2Fcustom.com" in url


class TestCheckPermission:
    """Tests for permission checking methods."""

    def test_check_permission_returns_true(self, auth0_manager, user_info):
        """check_permission returns True for existing permission."""
        assert auth0_manager.check_permission(user_info, "run:agent") is True

    def test_check_permission_returns_false(self, auth0_manager, user_info):
        """check_permission returns False for missing permission."""
        assert auth0_manager.check_permission(user_info, "admin:config") is False

    def test_get_user_permissions(self, auth0_manager, user_info):
        """get_user_permissions returns user's permissions."""
        permissions = auth0_manager.get_user_permissions(user_info)

        assert permissions == ["run:agent", "view:results"]

    def test_require_permission_passes(self, auth0_manager, user_info):
        """require_permission passes for existing permission."""
        # Should not raise
        auth0_manager.require_permission(user_info, "run:agent")

    def test_require_permission_raises(self, auth0_manager, user_info):
        """require_permission raises for missing permission."""
        with pytest.raises(InsufficientPermissionsError) as exc_info:
            auth0_manager.require_permission(user_info, "admin:config")

        assert exc_info.value.required_permission == "admin:config"
        assert exc_info.value.user_permissions == ["run:agent", "view:results"]


class TestVerifyToken:
    """Tests for verify_token method."""

    @pytest.mark.asyncio
    async def test_verify_token_empty_raises(self, auth0_manager):
        """Empty token raises TokenValidationError."""
        with pytest.raises(TokenValidationError) as exc_info:
            await auth0_manager.verify_token("")

        assert "Token is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_verify_token_strips_bearer_prefix(self, auth0_manager, mock_jwks):
        """Token with Bearer prefix is handled correctly."""
        # Mock the JWKS fetch and JWT decode
        with patch.object(auth0_manager, "_fetch_jwks", return_value=mock_jwks):
            with patch("core.auth0_manager.jwt") as mock_jwt:
                mock_jwt.get_unverified_header.return_value = {"kid": "test-key-id"}
                mock_jwt.decode.return_value = {
                    "sub": "auth0|123",
                    "permissions": ["run:agent"],
                    "exp": time.time() + 3600,
                    "iat": time.time(),
                }

                result = await auth0_manager.verify_token("Bearer test-token")

                # Verify decode was called without Bearer prefix
                mock_jwt.decode.assert_called_once()
                call_args = mock_jwt.decode.call_args
                assert call_args[0][0] == "test-token"

    @pytest.mark.asyncio
    async def test_verify_token_success(self, auth0_manager, mock_jwks):
        """Successful token verification returns UserInfo."""
        with patch.object(auth0_manager, "_fetch_jwks", return_value=mock_jwks):
            with patch("core.auth0_manager.jwt") as mock_jwt:
                mock_jwt.get_unverified_header.return_value = {"kid": "test-key-id"}
                mock_jwt.decode.return_value = {
                    "sub": "auth0|123456",
                    "email": "user@example.com",
                    "name": "Test User",
                    "permissions": ["run:agent", "view:results"],
                    "exp": time.time() + 3600,
                    "iat": time.time(),
                }

                result = await auth0_manager.verify_token("valid-token")

                assert isinstance(result, UserInfo)
                assert result.user_id == "auth0|123456"
                assert result.email == "user@example.com"
                assert result.name == "Test User"
                assert "run:agent" in result.permissions

    @pytest.mark.asyncio
    async def test_verify_token_expired_raises(self, auth0_manager, mock_jwks):
        """Expired token raises TokenExpiredError."""
        from jose import ExpiredSignatureError

        with patch.object(auth0_manager, "_fetch_jwks", return_value=mock_jwks):
            with patch("core.auth0_manager.jwt") as mock_jwt:
                mock_jwt.get_unverified_header.return_value = {"kid": "test-key-id"}
                mock_jwt.decode.side_effect = ExpiredSignatureError("Token expired")

                with pytest.raises(TokenExpiredError):
                    await auth0_manager.verify_token("expired-token")

    @pytest.mark.asyncio
    async def test_verify_token_invalid_header_raises(self, auth0_manager, mock_jwks):
        """Token with invalid header raises TokenValidationError."""
        from jose import JWTError

        with patch.object(auth0_manager, "_fetch_jwks", return_value=mock_jwks):
            with patch("core.auth0_manager.jwt") as mock_jwt:
                mock_jwt.get_unverified_header.side_effect = JWTError("Invalid header")

                with pytest.raises(TokenValidationError) as exc_info:
                    await auth0_manager.verify_token("invalid-token")

                assert "Invalid token header" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_verify_token_missing_kid_raises(self, auth0_manager, mock_jwks):
        """Token without kid in header raises TokenValidationError."""
        with patch.object(auth0_manager, "_fetch_jwks", return_value=mock_jwks):
            with patch("core.auth0_manager.jwt") as mock_jwt:
                mock_jwt.get_unverified_header.return_value = {}  # No kid

                with pytest.raises(TokenValidationError) as exc_info:
                    await auth0_manager.verify_token("no-kid-token")

                assert "missing 'kid'" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_verify_token_key_not_found_refreshes_jwks(
        self, auth0_manager, mock_jwks
    ):
        """When key not found, JWKS is refreshed."""
        jwks_with_new_key = {
            "keys": [
                {"kty": "RSA", "kid": "new-key-id", "use": "sig", "n": "n", "e": "AQAB"}
            ]
        }

        fetch_count = 0

        async def mock_fetch():
            nonlocal fetch_count
            fetch_count += 1
            if fetch_count == 1:
                return mock_jwks  # First call returns old JWKS
            return jwks_with_new_key  # Second call returns new JWKS

        with patch.object(auth0_manager, "_fetch_jwks", side_effect=mock_fetch):
            with patch("core.auth0_manager.jwt") as mock_jwt:
                mock_jwt.get_unverified_header.return_value = {"kid": "new-key-id"}
                mock_jwt.decode.return_value = {
                    "sub": "auth0|123",
                    "permissions": [],
                    "exp": time.time() + 3600,
                    "iat": time.time(),
                }

                result = await auth0_manager.verify_token("token-with-new-key")

                assert fetch_count == 2  # JWKS was refreshed
                assert result.user_id == "auth0|123"

    @pytest.mark.asyncio
    async def test_verify_token_missing_sub_raises(self, auth0_manager, mock_jwks):
        """Token without sub claim raises TokenValidationError."""
        with patch.object(auth0_manager, "_fetch_jwks", return_value=mock_jwks):
            with patch("core.auth0_manager.jwt") as mock_jwt:
                mock_jwt.get_unverified_header.return_value = {"kid": "test-key-id"}
                mock_jwt.decode.return_value = {
                    # No 'sub' claim
                    "permissions": [],
                    "exp": time.time() + 3600,
                    "iat": time.time(),
                }

                with pytest.raises(TokenValidationError) as exc_info:
                    await auth0_manager.verify_token("no-sub-token")

                assert "missing 'sub'" in str(exc_info.value)


class TestFetchJwks:
    """Tests for JWKS fetching."""

    @pytest.mark.asyncio
    async def test_fetch_jwks_success(self, auth0_manager, mock_jwks):
        """Successful JWKS fetch returns keys."""
        with patch("core.auth0_manager.aiohttp.ClientSession") as mock_session:
            # Create mock response
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_jwks)

            # Create async context manager for get()
            mock_get_cm = MagicMock()
            mock_get_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get_cm.__aexit__ = AsyncMock(return_value=None)

            # Create async context manager for session
            mock_session_instance = MagicMock()
            mock_session_instance.get.return_value = mock_get_cm
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            result = await auth0_manager._fetch_jwks()

            assert result == mock_jwks

    @pytest.mark.asyncio
    async def test_fetch_jwks_http_error_raises(self, auth0_manager):
        """HTTP error when fetching JWKS raises Auth0Error."""
        with patch("core.auth0_manager.aiohttp.ClientSession") as mock_session:
            mock_response = MagicMock()
            mock_response.status = 500

            mock_get_cm = MagicMock()
            mock_get_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get_cm.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.get.return_value = mock_get_cm
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            with pytest.raises(Auth0Error) as exc_info:
                await auth0_manager._fetch_jwks()

            assert "HTTP 500" in str(exc_info.value)


class TestExchangeCodeForTokens:
    """Tests for exchange_code_for_tokens method."""

    @pytest.mark.asyncio
    async def test_exchange_code_success(self, auth0_manager):
        """Successful code exchange returns tokens."""
        token_response = {
            "access_token": "access-token-123",
            "id_token": "id-token-456",
            "token_type": "Bearer",
            "expires_in": 86400,
        }

        with patch("core.auth0_manager.aiohttp.ClientSession") as mock_session:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=token_response)

            mock_post_cm = MagicMock()
            mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post_cm.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.post.return_value = mock_post_cm
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            result = await auth0_manager.exchange_code_for_tokens("auth-code-123")

            assert result["access_token"] == "access-token-123"
            assert result["id_token"] == "id-token-456"

    @pytest.mark.asyncio
    async def test_exchange_code_error_raises(self, auth0_manager):
        """Error during code exchange raises Auth0Error."""
        error_response = {
            "error": "invalid_grant",
            "error_description": "Invalid authorization code",
        }

        with patch("core.auth0_manager.aiohttp.ClientSession") as mock_session:
            mock_response = MagicMock()
            mock_response.status = 400
            mock_response.json = AsyncMock(return_value=error_response)

            mock_post_cm = MagicMock()
            mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post_cm.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.post.return_value = mock_post_cm
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            with pytest.raises(Auth0Error) as exc_info:
                await auth0_manager.exchange_code_for_tokens("invalid-code")

            assert "Invalid authorization code" in str(exc_info.value)


class TestGlobalAuth0Manager:
    """Tests for global Auth0 manager functions."""

    @pytest.mark.asyncio
    async def test_init_and_get_auth0_manager(self, auth0_config):
        """init_auth0 creates manager accessible via get_auth0_manager."""
        # Clean up any existing manager
        await close_auth0()

        manager = init_auth0(auth0_config)

        assert manager is not None
        assert get_auth0_manager() is manager

        # Clean up
        await close_auth0()

    @pytest.mark.asyncio
    async def test_get_auth0_manager_not_initialized_raises(self):
        """get_auth0_manager raises when not initialized."""
        # Ensure manager is closed
        await close_auth0()

        with pytest.raises(RuntimeError) as exc_info:
            get_auth0_manager()

        assert "not initialized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_close_auth0_clears_manager(self, auth0_config):
        """close_auth0 clears the global manager."""
        init_auth0(auth0_config)

        await close_auth0()

        with pytest.raises(RuntimeError):
            get_auth0_manager()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
