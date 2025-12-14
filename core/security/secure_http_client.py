"""Secure HTTP client with integrated security protections.

This module provides the SecureHTTPClient class that wraps aiohttp with
comprehensive security validations including SSRF protection, response
size limiting, credential sanitization, and redirect validation.

Requirements: 1.1, 1.2, 2.4, 2.5, 2.8, 2.9, 3.1, 3.2
"""

import asyncio
import logging
from typing import Any, Dict, Optional, Set

import aiohttp

from core.security.credential_sanitizer import CredentialSanitizer
from core.security.resource_limiter import (
    ResourceLimiter,
    ResponseTooLargeError,
)
from core.security.secure_logger import SecureLogger
from core.security.ssrf_protector import (
    SSRFProtector,
    SSRFError,
)


logger = logging.getLogger(__name__)


class SecureHTTPClientError(Exception):
    """Base class for SecureHTTPClient errors."""
    pass


class InvalidParameterError(SecureHTTPClientError):
    """Raised when request parameters are invalid."""
    pass


class SecureHTTPClient:
    """HTTP client with integrated security protections.

    This class wraps aiohttp to provide comprehensive security validations:
    - SSRF protection via domain allowlist and private IP blocking
    - Response size limiting to prevent resource exhaustion
    - Credential sanitization for logging
    - Redirect validation with DNS rebinding protection

    Requirements: 1.1, 1.2, 2.4, 2.5, 2.8, 2.9, 3.1, 3.2
    """

    # Allowed parameter keys for explorer APIs
    # Only these keys are permitted in request parameters
    ALLOWED_PARAM_KEYS: Set[str] = {
        "module",
        "action",
        "contractaddress",
        "address",
        "page",
        "offset",
        "sort",
        "startblock",
        "endblock",
        "apikey",
        "tag",
        "txhash",
        "blockno",
        "timestamp",
        "closest",
    }

    # Maximum number of redirects to follow
    MAX_REDIRECTS = 5

    def __init__(
        self,
        ssrf_protector: SSRFProtector,
        resource_limiter: ResourceLimiter,
        sanitizer: CredentialSanitizer,
        secure_logger: Optional[SecureLogger] = None,
        timeout: float = 30.0,
    ):
        """Initialize with security components.

        Args:
            ssrf_protector: SSRFProtector instance for URL validation.
            resource_limiter: ResourceLimiter for response size checking.
            sanitizer: CredentialSanitizer for logging.
            secure_logger: Optional SecureLogger for sanitized logging.
            timeout: Default request timeout in seconds.
        """
        self._ssrf_protector = ssrf_protector
        self._resource_limiter = resource_limiter
        self._sanitizer = sanitizer
        self._logger = secure_logger
        self._default_timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session.

        Returns:
            aiohttp.ClientSession instance.
        """
        if self._session is None or self._session.closed:
            # Create session with redirect disabled (handle manually)
            timeout = aiohttp.ClientTimeout(total=self._default_timeout)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                raise_for_status=False,
            )
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "SecureHTTPClient":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self, exc_type: Any, exc_val: Any, exc_tb: Any
    ) -> None:
        """Async context manager exit."""
        await self.close()

    def _validate_params(
        self, params: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Validate and sanitize request parameters.

        Validates that:
        1. All param keys are in ALLOWED_PARAM_KEYS allowlist
        2. Param value types are str, int, or bool only
        3. No param value looks like a URL (potential SSRF via param)

        Args:
            params: Dictionary of request parameters.

        Returns:
            Sanitized params dict with all values as strings.

        Raises:
            InvalidParameterError: If param key not in allowlist.

        Requirements: 2.4, 2.5
        """
        if not params:
            return {}

        validated: Dict[str, str] = {}

        for key, value in params.items():
            # Check key is in allowlist
            key_lower = key.lower()
            allowed_lower = {k.lower() for k in self.ALLOWED_PARAM_KEYS}
            if key_lower not in allowed_lower:
                raise InvalidParameterError(
                    f"Parameter key '{key}' is not in the allowed list."
                )

            # Check value type
            if not isinstance(value, (str, int, bool)):
                type_name = type(value).__name__
                raise InvalidParameterError(
                    f"Parameter '{key}' has invalid type {type_name}. "
                    "Only str, int, and bool are allowed."
                )

            # Convert to string
            if isinstance(value, bool):
                str_value = str(value).lower()
            else:
                str_value = str(value)

            # Check if value looks like a URL (potential SSRF via param)
            if self._looks_like_url(str_value):
                raise InvalidParameterError(
                    f"Parameter '{key}' value appears to be a URL."
                )

            validated[key] = str_value

        return validated

    def _looks_like_url(self, value: str) -> bool:
        """Check if a value looks like a URL.

        Args:
            value: String value to check.

        Returns:
            True if value appears to be a URL.
        """
        if not value:
            return False

        value_lower = value.lower().strip()

        # Check for common URL schemes
        url_schemes = ("http://", "https://", "ftp://", "file://")
        if value_lower.startswith(url_schemes):
            return True

        # Check for URL-like patterns without scheme
        if value_lower.startswith("//"):
            return True

        return False

    def _sanitize_params_for_logging(
        self, params: Dict[str, Any]
    ) -> Dict[str, str]:
        """Create a copy of params with credentials redacted for logging.

        Args:
            params: Dictionary of request parameters.

        Returns:
            Dictionary with credential values redacted.

        Requirements: 1.1, 1.2
        """
        if not params:
            return {}

        return self._sanitizer.sanitize_dict(params)

    async def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make GET request with all security validations.

        Parameter handling:
        1. Validate all param keys are in ALLOWED_PARAM_KEYS allowlist
        2. Validate param value types (str, int, bool only)
        3. URL-encode all param values to prevent injection
        4. Run CredentialSanitizer on param values for logging
        5. Check if any param value looks like a URL and reject
        6. Log only redacted params at debug level

        Args:
            url: Target URL (must be HTTPS and in domain allowlist).
            params: Optional query parameters.
            timeout: Optional request timeout (uses default if not specified).
            headers: Optional HTTP headers.

        Returns:
            Parsed JSON response as dictionary.

        Raises:
            InvalidParameterError: If param key not in allowlist.
            SSRFError: If URL fails SSRF validation.
            ResponseTooLargeError: If response exceeds size limit.
            asyncio.TimeoutError: If request exceeds timeout.
            aiohttp.ClientError: If HTTP request fails.

        Requirements: 1.1, 1.2, 2.4, 2.5, 3.1, 3.2
        """
        # Validate and sanitize parameters
        validated_params = self._validate_params(params)

        # Log sanitized request details
        if self._logger:
            sanitized_params = self._sanitize_params_for_logging(
                validated_params
            )
            sanitized_url = self._sanitizer.sanitize_url(url)
            self._logger.debug(
                f"GET request to {sanitized_url} params: {sanitized_params}"
            )

        # Validate URL against SSRF protections
        await self._ssrf_protector.validate_request(url)

        # Get session and make request
        session = await self._get_session()
        if timeout is not None:
            effective_timeout = timeout
        else:
            effective_timeout = self._default_timeout

        # Build full URL for redirect tracking
        current_url = url
        redirect_count = 0
        original_resolved_ip: Optional[str] = None

        while True:
            try:
                # Create timeout for this request
                request_timeout = aiohttp.ClientTimeout(
                    total=effective_timeout
                )

                # Use params only on first request (not redirects)
                if redirect_count == 0:
                    req_params = validated_params
                else:
                    req_params = None
                async with session.get(
                    current_url,
                    params=req_params,
                    headers=headers,
                    timeout=request_timeout,
                    allow_redirects=False,
                ) as response:
                    # Check for redirect
                    if response.status in (301, 302, 303, 307, 308):
                        # Capture original IP for DNS rebinding check on first request
                        if redirect_count == 0:
                            try:
                                if response.connection and response.connection.transport:
                                    peername = response.connection.transport.get_extra_info("peername")
                                    if peername and len(peername) > 0:
                                        original_resolved_ip = peername[0]
                            except Exception as e:
                                # Log warning but don't fail if we can't get IP
                                if self._logger:
                                    self._logger.warning(
                                        f"Failed to capture original IP for DNS rebinding check: {e}"
                                    )

                        redirect_count += 1
                        if redirect_count > self.MAX_REDIRECTS:
                            max_redir = self.MAX_REDIRECTS
                            raise SSRFError(
                                f"Too many redirects (max {max_redir})"
                            )

                        # Get redirect location
                        redirect_url = response.headers.get("Location")
                        if not redirect_url:
                            raise SSRFError(
                                "Redirect response missing Location header"
                            )

                        # Handle relative redirects
                        redirect_url = self._resolve_redirect_url(
                            current_url, redirect_url
                        )

                        # Validate redirect with SSRF and rebinding checks
                        await self._handle_redirect(
                            current_url,
                            redirect_url,
                            original_resolved_ip,
                        )

                        if self._logger:
                            sanitized_redir = self._sanitizer.sanitize_url(
                                redirect_url
                            )
                            self._logger.debug(
                                f"Following redirect to {sanitized_redir}"
                            )

                        current_url = redirect_url
                        continue

                    # Check response size before reading body
                    content_len = response.headers.get("Content-Length")
                    if content_len:
                        try:
                            size = int(content_len)
                            self._resource_limiter.check_response_size(size)
                        except ValueError:
                            # Invalid Content-Length, check actual size
                            pass

                    # Read response with size limit
                    body = await self._read_response_with_limit(response)

                    # Parse JSON response
                    try:
                        import json
                        data = json.loads(body)
                    except json.JSONDecodeError as e:
                        raise SecureHTTPClientError(
                            f"Invalid JSON response: {e}"
                        )

                    return data

            except aiohttp.ClientError as e:
                if self._logger:
                    err_type = type(e).__name__
                    self._logger.error(f"HTTP request failed: {err_type}")
                raise

    async def _read_response_with_limit(
        self, response: aiohttp.ClientResponse
    ) -> bytes:
        """Read response body with size limit enforcement.

        Args:
            response: aiohttp response object.

        Returns:
            Response body as bytes.

        Raises:
            ResponseTooLargeError: If response exceeds size limit.

        Requirements: 3.1, 3.2
        """
        max_size = self._resource_limiter.max_response_size
        chunks: list[bytes] = []
        total_size = 0

        async for chunk in response.content.iter_chunked(8192):
            total_size += len(chunk)
            if total_size > max_size:
                raise ResponseTooLargeError(
                    size=total_size,
                    limit=max_size,
                )
            chunks.append(chunk)

        return b"".join(chunks)

    def _resolve_redirect_url(self, base_url: str, redirect_url: str) -> str:
        """Resolve a potentially relative redirect URL.

        Args:
            base_url: The original request URL.
            redirect_url: The redirect Location header value.

        Returns:
            Absolute redirect URL.
        """
        from urllib.parse import urljoin

        # If redirect_url is already absolute, urljoin returns it unchanged
        return urljoin(base_url, redirect_url)

    async def _handle_redirect(
        self,
        original_url: str,
        redirect_url: str,
        original_resolved_ip: Optional[str] = None,
    ) -> None:
        """Handle redirect with SSRF validation and rebinding check.

        Validates the redirect target against:
        1. Domain allowlist
        2. HTTPS protocol requirement
        3. Private IP blocking
        4. DNS rebinding detection

        Args:
            original_url: The URL that returned the redirect.
            redirect_url: The redirect target URL.
            original_resolved_ip: Optional cached IP from original request.

        Raises:
            DomainNotAllowedError: If redirect domain not in allowlist.
            PrivateIPError: If redirect resolves to private IP.
            DNSRebindingError: If DNS rebinding attack detected.

        Requirements: 2.8, 2.9
        """
        await self._ssrf_protector.validate_redirect(
            original_url=original_url,
            redirect_url=redirect_url,
            original_resolved_ip=original_resolved_ip,
        )

    async def get_with_retry(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        backoff_multiplier: float = 2.0,
    ) -> Dict[str, Any]:
        """Make GET request with automatic retry on transient failures.

        Args:
            url: Target URL.
            params: Optional query parameters.
            timeout: Optional request timeout.
            headers: Optional HTTP headers.
            max_retries: Maximum number of retry attempts.
            backoff_base: Base delay for exponential backoff.
            backoff_multiplier: Multiplier for exponential backoff.

        Returns:
            Parsed JSON response as dictionary.

        Raises:
            Various exceptions if all retries fail.
        """
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                return await self.get(
                    url=url,
                    params=params,
                    timeout=timeout,
                    headers=headers,
                )
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e

                if attempt < max_retries:
                    delay = backoff_base * (backoff_multiplier ** attempt)
                    if self._logger:
                        attempt_num = attempt + 1
                        total = max_retries + 1
                        err_type = type(e).__name__
                        self._logger.warning(
                            f"Request failed ({attempt_num}/{total}), "
                            f"retrying in {delay:.1f}s: {err_type}"
                        )
                    await asyncio.sleep(delay)
                else:
                    if self._logger:
                        total = max_retries + 1
                        self._logger.error(
                            f"Request failed after {total} attempts"
                        )

            except (SSRFError, InvalidParameterError, ResponseTooLargeError):
                # Don't retry security errors
                raise

        # All retries exhausted
        if last_error:
            raise last_error
        raise SecureHTTPClientError("Request failed with unknown error")
