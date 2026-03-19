"""Base class for blockchain explorer collectors."""

import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional
import threading

import aiohttp

from config.models import ExplorerConfig, RetryConfig
from collectors.models import Transaction, Holder, ExplorerData, ActivityType


# Use standard logging to avoid circular imports
# The logging will be configured by core.logging when the app starts
logger = logging.getLogger(__name__)

# Lazy import for security components to avoid circular imports
_secure_http_client = None
_secure_http_client_lock = None
_secure_http_client_lock_init = threading.Lock()


def _get_secure_http_client_lock() -> asyncio.Lock:
    """Get or create the secure http client lock (async-safe)."""
    global _secure_http_client_lock
    if _secure_http_client_lock is None:
        with _secure_http_client_lock_init:
            if _secure_http_client_lock is None:
                _secure_http_client_lock = asyncio.Lock()
    return _secure_http_client_lock


async def _get_secure_http_client():
    """Get or create the secure HTTP client singleton (async-safe).

    Returns None if security components are not available, allowing
    fallback to standard aiohttp session.
    """
    global _secure_http_client
    async with _get_secure_http_client_lock():
        if _secure_http_client is None:
            try:
                from core.security.secure_http_client import SecureHTTPClient
                from core.security.ssrf_protector import (
                    SSRFProtector,
                    DomainAllowlist,
                )
                from core.security.resource_limiter import ResourceLimiter
                from core.security.credential_sanitizer import CredentialSanitizer
                from core.security.secure_logger import SecureLogger
                from config.models import (
                    SSRFProtectionConfig,
                    ResourceLimitConfig,
                    CredentialSanitizerConfig,
                )

                # Create security components with default configs
                ssrf_config = SSRFProtectionConfig()
                allowlist = DomainAllowlist(ssrf_config.allowed_domains)
                ssrf_protector = SSRFProtector(allowlist)

                resource_config = ResourceLimitConfig()
                resource_limiter = ResourceLimiter(resource_config)

                sanitizer_config = CredentialSanitizerConfig()
                sanitizer = CredentialSanitizer(sanitizer_config)

                secure_logger = SecureLogger(logger, sanitizer)

                _secure_http_client = SecureHTTPClient(
                    ssrf_protector=ssrf_protector,
                    resource_limiter=resource_limiter,
                    sanitizer=sanitizer,
                    secure_logger=secure_logger,
                )
                logger.info("SecureHTTPClient initialized for collectors")
            except Exception as e:
                logger.warning(f"Failed to initialize SecureHTTPClient: {e}")
                _secure_http_client = False  # Mark as failed, don't retry
    return _secure_http_client if _secure_http_client else None


# Lazy import for blockchain validator to avoid circular imports
_blockchain_validator = None


def _get_blockchain_validator():
    """Get or create the blockchain validator singleton."""
    global _blockchain_validator
    if _blockchain_validator is None:
        try:
            from core.security.blockchain_validator import BlockchainDataValidator

            _blockchain_validator = BlockchainDataValidator()
            logger.debug("BlockchainDataValidator initialized for ExplorerCollector")
        except Exception as e:
            logger.warning(f"Failed to initialize BlockchainDataValidator: {e}")
            _blockchain_validator = False  # Mark as failed, don't retry
    return _blockchain_validator if _blockchain_validator else None


# Lazy import for schema validator to avoid circular imports
_schema_validator = None
_schema_validator_lock = None
_schema_validator_lock_init = threading.Lock()


def _get_schema_validator_lock() -> asyncio.Lock:
    """Get or create the schema validator lock (async-safe)."""
    global _schema_validator_lock
    if _schema_validator_lock is None:
        with _schema_validator_lock_init:
            if _schema_validator_lock is None:
                _schema_validator_lock = asyncio.Lock()
    return _schema_validator_lock


def init_collector_locks():
    """Explicitly initialize asyncio locks on the current event loop.

    This should be called during application startup to ensure locks are
    created on the correct event loop, avoiding potential race conditions
    or loop binding issues with lazy initialization.
    """
    global _secure_http_client_lock, _schema_validator_lock

    # Initialize secure http client lock
    if _secure_http_client_lock is None:
        with _secure_http_client_lock_init:
            if _secure_http_client_lock is None:
                _secure_http_client_lock = asyncio.Lock()
                logger.debug("Initialized _secure_http_client_lock")

    # Initialize schema validator lock
    if _schema_validator_lock is None:
        with _schema_validator_lock_init:
            if _schema_validator_lock is None:
                _schema_validator_lock = asyncio.Lock()
                logger.debug("Initialized _schema_validator_lock")


async def _get_schema_validator():
    """Get or create the schema validator singleton (async-safe)."""
    global _schema_validator
    async with _get_schema_validator_lock():
        if _schema_validator is None:
            try:
                from core.security.schema_validator import (
                    ResponseSchemaValidator,
                    SchemaFallbackStrategy,
                )

                # Resolve schemas directory relative to this file
                # collectors/base.py -> ../schemas
                schema_dir = Path(__file__).resolve().parent.parent / "schemas"

                _schema_validator = ResponseSchemaValidator(
                    schema_directory=schema_dir,
                    fallback_strategy=SchemaFallbackStrategy.SKIP_VALIDATION,
                )
                _schema_validator.load_schemas()
                logger.info("Schema validator initialized for collectors")
            except Exception as e:
                logger.warning(f"Failed to initialize schema validator: {e}")
                _schema_validator = False  # Mark as failed, don't retry
    return _schema_validator if _schema_validator else None


class CollectorTimeoutError(Exception):
    """Raised when a collector request times out."""

    def __init__(self, service: str, timeout_seconds: float):
        self.service = service
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Request to {service} timed out after {timeout_seconds}s")


class ExplorerCollector(ABC):
    # Token decimals (override in subclasses)
    TOKEN_DECIMALS: dict[str, int] = {}
    """Abstract base class for blockchain explorer data collectors.
    
    Provides common functionality for rate limiting, retry logic,
    and response validation. Concrete implementations must implement
    the fetch_stablecoin_transactions and fetch_token_holders methods.
    """

    def __init__(
        self, config: ExplorerConfig, retry_config: Optional[RetryConfig] = None
    ):
        """Initialize the collector.

        Args:
            config: Explorer configuration with API details
            retry_config: Retry configuration (uses defaults if not provided)
        """
        self.config = config
        self.retry_config = retry_config or RetryConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._request_count = 0
        self._last_request_time: Optional[float] = None

    @property
    def name(self) -> str:
        """Get the explorer name."""
        return self.config.name

    @property
    def chain(self) -> str:
        """Get the blockchain chain."""
        return self.config.chain

    @property
    def base_url(self) -> str:
        """Get the base API URL."""
        return str(self.config.base_url)

    @property
    def api_key(self) -> str:
        """Get the API key."""
        return self.config.api_key

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                total=self.retry_config.request_timeout_seconds
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "ExplorerCollector":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def handle_rate_limit(self) -> None:
        """Handle rate limiting by waiting before the next request.

        Implements a simple rate limiting strategy by waiting
        the configured backoff time when rate limited.
        """
        wait_time = self.retry_config.backoff_seconds
        logger.warning(
            f"Rate limited on {self.name}. Waiting {wait_time} seconds before retry.",
            extra={"explorer": self.name, "wait_seconds": wait_time},
        )
        await asyncio.sleep(wait_time)

    async def validate_response(
        self,
        response: dict,
        endpoint: Optional[str] = None,
    ) -> bool:
        """Validate an API response.

        Performs both basic validation and JSON schema validation when
        an endpoint is specified.

        Args:
            response: The JSON response from the API
            endpoint: Optional endpoint name for schema validation
                      (e.g., "tokentx", "tokenholderlist")

        Returns:
            True if the response is valid, False otherwise
        """
        if not isinstance(response, dict):
            logger.error(
                f"Invalid response type from {self.name}: "
                f"expected dict, got {type(response)}",
                extra={"explorer": self.name},
            )
            return False

        # Check for common error indicators
        status = response.get("status")
        message = response.get("message", "")

        # Status "0" typically indicates an error in *scan APIs
        if status == "0":
            # "No transactions found" is not an error, just empty results
            if "No transactions found" in message or "No records found" in message:
                return True
            logger.warning(
                f"API error from {self.name}: {message}",
                extra={"explorer": self.name, "message": message},
            )
            return False

        # Perform schema validation if endpoint is specified
        if endpoint:
            schema_valid = await self._validate_response_schema(response, endpoint)
            if not schema_valid:
                return False

        return True

    async def _validate_response_schema(
        self,
        response: dict,
        endpoint: str,
    ) -> bool:
        """Validate response against JSON schema.

        Args:
            response: The API response to validate
            endpoint: The endpoint name (e.g., "tokentx")

        Returns:
            True if valid or validation skipped, False if invalid

        Requirements: 4.8, 4.9, 4.11
        """
        validator = await _get_schema_validator()
        if validator is None:
            # Schema validation not available, skip
            return True

        # Map explorer name to schema directory name
        explorer_name = self.name.lower()  # or the intended transformation

        # Detect schema version from response
        detected_version, version_source = validator.detect_response_version(response)
        expected_version = validator.get_schema_version(explorer_name, endpoint)

        # Log version mismatch at WARNING level
        if (
            detected_version != "unknown"
            and expected_version
            and detected_version != expected_version
        ):
            classification = validator.classify_version_mismatch(
                detected_version, expected_version
            )
            logger.warning(
                f"Schema version mismatch for {explorer_name}/{endpoint}: "
                f"detected {detected_version} (from {version_source}), "
                f"expected {expected_version}, classification: {classification.value}",
                extra={
                    "explorer": self.name,
                    "endpoint": endpoint,
                    "detected_version": detected_version,
                    "expected_version": expected_version,
                    "classification": classification.value,
                },
            )

        # Validate against schema
        result = validator.validate(response, explorer_name, endpoint)

        if not result.is_valid:
            # Log warning with field paths only (not raw values)
            visible_paths = result.field_paths[:10]
            logger.warning(
                f"Schema validation failed for {explorer_name}/{endpoint}: "
                f"{len(result.errors)} error(s) at paths: "
                f"{', '.join(visible_paths)}",
                extra={
                    "explorer": self.name,
                    "endpoint": endpoint,
                    "error_count": len(result.errors),
                    "field_paths": visible_paths,
                    "nesting_exceeded": result.nesting_depth_exceeded,
                },
            )
            return False

        return True

    def _is_rate_limit_error(self, response: dict) -> bool:
        """Check if the response indicates a rate limit error.

        Args:
            response: The JSON response from the API

        Returns:
            True if rate limited, False otherwise
        """
        message = response.get("message", "").lower()
        result = response.get("result", "")

        rate_limit_indicators = [
            "rate limit",
            "max rate limit",
            "too many requests",
            "exceeded the rate limit",
        ]

        for indicator in rate_limit_indicators:
            if indicator in message.lower() or (
                isinstance(result, str) and indicator in result.lower()
            ):
                return True

        return False

    async def _make_request(
        self,
        params: dict[str, Any],
        run_id: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> Optional[dict]:
        """Make an API request with retry logic and exponential backoff.

        Implements comprehensive error handling for:
        - Network errors (connection failures, DNS issues)
        - Rate limiting (HTTP 429 and API-level limits)
        - Timeouts (configurable request timeout)
        - API authentication errors (401, 403)
        - Invalid responses
        - Schema validation (when endpoint is specified)
        - SSRF protection (via SecureHTTPClient when available)

        Args:
            params: Query parameters for the API request
            run_id: Optional run ID for logging correlation
            endpoint: Optional endpoint name for schema validation
                      (e.g., "tokentx", "tokenholderlist")

        Returns:
            The JSON response if successful, None otherwise

        Requirements: 1.1, 2.4, 3.1
        """
        # Add API key to params
        params["apikey"] = self.api_key

        # Try to use SecureHTTPClient if available
        secure_client = await _get_secure_http_client()

        if secure_client is not None:
            return await self._make_secure_request(
                secure_client, params, run_id, endpoint
            )
        else:
            # Fallback to standard aiohttp session
            return await self._make_standard_request(params, run_id, endpoint)

    async def _make_secure_request(
        self,
        secure_client: Any,
        params: dict[str, Any],
        run_id: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> Optional[dict]:
        """Make request using SecureHTTPClient with SSRF protection.

        Args:
            secure_client: SecureHTTPClient instance
            params: Query parameters for the API request
            run_id: Optional run ID for logging correlation
            endpoint: Optional endpoint name for schema validation

        Returns:
            The JSON response if successful, None otherwise

        Requirements: 1.1, 2.4, 3.1
        """
        from core.security.ssrf_protector import SSRFError
        from core.security.resource_limiter import ResponseTooLargeError
        from core.security.secure_http_client import InvalidParameterError

        last_error: Optional[Exception] = None

        for attempt in range(self.retry_config.max_attempts):
            log_extra = {
                "explorer": self.name,
                "attempt": attempt + 1,
                "max_attempts": self.retry_config.max_attempts,
                "secure_client": True,
            }
            if run_id:
                log_extra["run_id"] = run_id

            try:
                logger.debug(
                    f"Making secure request to {self.name} (attempt {attempt + 1}/{self.retry_config.max_attempts})",
                    extra=log_extra,
                )

                data = await secure_client.get(
                    url=self.base_url,
                    params=params,
                    timeout=self.retry_config.request_timeout_seconds,
                )

                # Check for API-level rate limiting
                if self._is_rate_limit_error(data):
                    logger.warning(
                        f"API-level rate limit from {self.name}",
                        extra={**log_extra, "error_type": "api_rate_limit"},
                    )
                    await self.handle_rate_limit()
                    continue

                # Validate response structure and schema
                if not await self.validate_response(data, endpoint=endpoint):
                    if attempt < self.retry_config.max_attempts - 1:
                        backoff = self.retry_config.backoff_seconds * (2**attempt)
                        logger.info(
                            f"Retrying {self.name} in {backoff} seconds after validation failure",
                            extra={**log_extra, "backoff_seconds": backoff},
                        )
                        await asyncio.sleep(backoff)
                        continue
                    return None

                return data

            except SSRFError as e:
                # SSRF errors are security issues - don't retry
                logger.error(
                    f"SSRF protection blocked request to {self.name}: {e}",
                    extra={**log_extra, "error_type": "ssrf_blocked"},
                )
                return None

            except InvalidParameterError as e:
                # Invalid parameters - don't retry
                logger.error(
                    f"Invalid parameters for {self.name}: {e}",
                    extra={**log_extra, "error_type": "invalid_params"},
                )
                return None

            except ResponseTooLargeError as e:
                # Response too large - don't retry
                logger.error(
                    f"Response too large from {self.name}: {e}",
                    extra={**log_extra, "error_type": "response_too_large"},
                )
                return None

            except asyncio.TimeoutError:
                last_error = CollectorTimeoutError(
                    service=self.name,
                    timeout_seconds=self.retry_config.request_timeout_seconds,
                )
                logger.warning(
                    f"Request timeout to {self.name} after {self.retry_config.request_timeout_seconds}s (attempt {attempt + 1})",
                    extra={**log_extra, "error_type": "timeout"},
                )
            except aiohttp.ClientError as e:
                last_error = e
                logger.warning(
                    f"Network error from {self.name}: {e} (attempt {attempt + 1})",
                    extra={**log_extra, "error_type": "network", "error": str(e)},
                )
            except Exception as e:
                last_error = e
                logger.error(
                    f"Unexpected error from {self.name}: {e}",
                    extra={**log_extra, "error_type": "unexpected", "error": str(e)},
                    exc_info=True,
                )

            # Exponential backoff before retry
            if attempt < self.retry_config.max_attempts - 1:
                backoff = self.retry_config.backoff_seconds * (2**attempt)
                logger.info(
                    f"Retrying {self.name} in {backoff} seconds",
                    extra={**log_extra, "backoff_seconds": backoff},
                )
                await asyncio.sleep(backoff)

        logger.error(
            f"All {self.retry_config.max_attempts} attempts failed for {self.name}",
            extra={
                "explorer": self.name,
                "last_error": str(last_error),
                "error_type": type(last_error).__name__ if last_error else "unknown",
            },
        )
        return None

    async def _make_standard_request(
        self,
        params: dict[str, Any],
        run_id: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> Optional[dict]:
        """Make request using standard aiohttp session (fallback).

        This is the fallback when SecureHTTPClient is not available.

        Args:
            params: Query parameters for the API request
            run_id: Optional run ID for logging correlation
            endpoint: Optional endpoint name for schema validation

        Returns:
            The JSON response if successful, None otherwise
        """
        session = await self._get_session()

        last_error: Optional[Exception] = None

        for attempt in range(self.retry_config.max_attempts):
            log_extra = {
                "explorer": self.name,
                "attempt": attempt + 1,
                "max_attempts": self.retry_config.max_attempts,
            }
            if run_id:
                log_extra["run_id"] = run_id

            try:
                logger.debug(
                    f"Making request to {self.name} (attempt {attempt + 1}/{self.retry_config.max_attempts})",
                    extra=log_extra,
                )

                async with session.get(self.base_url, params=params) as response:
                    # Handle HTTP-level rate limiting (429)
                    if response.status == 429:
                        logger.warning(
                            f"HTTP 429 rate limit from {self.name}",
                            extra={**log_extra, "status_code": 429},
                        )
                        # Wait 60 seconds as per requirements
                        await self.handle_rate_limit()
                        continue

                    # Handle authentication errors (401, 403)
                    if response.status in (401, 403):
                        error_text = await response.text()
                        logger.error(
                            f"API authentication error from {self.name}: HTTP {response.status}",
                            extra={
                                **log_extra,
                                "status_code": response.status,
                                "error_type": "authentication",
                            },
                        )
                        # Don't retry auth errors - they won't succeed
                        return None

                    # Handle other HTTP errors
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(
                            f"HTTP {response.status} from {self.name}",
                            extra={
                                **log_extra,
                                "status_code": response.status,
                                "response_preview": error_text[:200]
                                if error_text
                                else None,
                            },
                        )
                        if attempt < self.retry_config.max_attempts - 1:
                            backoff = self.retry_config.backoff_seconds * (2**attempt)
                            logger.info(
                                f"Retrying {self.name} in {backoff} seconds after HTTP {response.status}",
                                extra={**log_extra, "backoff_seconds": backoff},
                            )
                            await asyncio.sleep(backoff)
                            continue
                        return None

                    # Parse JSON response
                    try:
                        data = await response.json()
                    except Exception as json_err:
                        logger.error(
                            f"Failed to parse JSON response from {self.name}: {json_err}",
                            extra={**log_extra, "error_type": "json_parse"},
                        )
                        if attempt < self.retry_config.max_attempts - 1:
                            backoff = self.retry_config.backoff_seconds * (2**attempt)
                            await asyncio.sleep(backoff)
                            continue
                        return None

                    # Check for API-level rate limiting
                    if self._is_rate_limit_error(data):
                        logger.warning(
                            f"API-level rate limit from {self.name}",
                            extra={**log_extra, "error_type": "api_rate_limit"},
                        )
                        # Wait 60 seconds as per requirements
                        await self.handle_rate_limit()
                        continue

                    # Validate response structure and schema
                    if not await self.validate_response(data, endpoint=endpoint):
                        if attempt < self.retry_config.max_attempts - 1:
                            backoff = self.retry_config.backoff_seconds * (2**attempt)
                            logger.info(
                                f"Retrying {self.name} in {backoff} seconds after validation failure",
                                extra={**log_extra, "backoff_seconds": backoff},
                            )
                            await asyncio.sleep(backoff)
                            continue
                        return None

                    return data

            except asyncio.TimeoutError:
                last_error = CollectorTimeoutError(
                    service=self.name,
                    timeout_seconds=self.retry_config.request_timeout_seconds,
                )
                logger.warning(
                    f"Request timeout to {self.name} after {self.retry_config.request_timeout_seconds}s (attempt {attempt + 1})",
                    extra={**log_extra, "error_type": "timeout"},
                )
            except aiohttp.ClientConnectorError as e:
                last_error = e
                logger.warning(
                    f"Connection error to {self.name}: {e} (attempt {attempt + 1})",
                    extra={**log_extra, "error_type": "connection", "error": str(e)},
                )
            except aiohttp.ClientError as e:
                last_error = e
                logger.warning(
                    f"Network error from {self.name}: {e} (attempt {attempt + 1})",
                    extra={**log_extra, "error_type": "network", "error": str(e)},
                )
            except Exception as e:
                last_error = e
                logger.error(
                    f"Unexpected error from {self.name}: {e}",
                    extra={**log_extra, "error_type": "unexpected", "error": str(e)},
                    exc_info=True,
                )

            # Exponential backoff before retry
            if attempt < self.retry_config.max_attempts - 1:
                backoff = self.retry_config.backoff_seconds * (2**attempt)
                logger.info(
                    f"Retrying {self.name} in {backoff} seconds",
                    extra={**log_extra, "backoff_seconds": backoff},
                )
                await asyncio.sleep(backoff)

        logger.error(
            f"All {self.retry_config.max_attempts} attempts failed for {self.name}",
            extra={
                "explorer": self.name,
                "last_error": str(last_error),
                "error_type": type(last_error).__name__ if last_error else "unknown",
            },
        )
        return None

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse a Unix timestamp string to datetime.

        Args:
            timestamp_str: Unix timestamp as string

        Returns:
            datetime object in UTC
        """
        return datetime.fromtimestamp(int(timestamp_str), tz=timezone.utc)

    def _parse_amount(self, value_str: str, stablecoin: str) -> Decimal:
        """Parse a token amount from raw value to decimal.

        Args:
            value_str: Raw token value as string (in smallest unit)
            stablecoin: The stablecoin symbol for decimal lookup

        Returns:
            Decimal amount in standard units
        """
        decimals = self.TOKEN_DECIMALS.get(stablecoin, 18)
        raw_value = Decimal(value_str)
        return raw_value / Decimal(10**decimals)

    def _classify_activity(
        self, from_address: str, to_address: str, amount: Decimal
    ) -> ActivityType:
        """Classify the activity type of a transaction.

        Args:
            from_address: Sender address
            to_address: Receiver address
            amount: Transaction amount

        Returns:
            ActivityType classification
        """
        # Zero address indicates minting or burning
        zero_address = "0x0000000000000000000000000000000000000000"

        if from_address.lower() == zero_address:
            return ActivityType.OTHER  # Minting
        if to_address.lower() == zero_address:
            return ActivityType.OTHER  # Burning

        # Standard transfer
        if amount > 0 and from_address and to_address:
            return ActivityType.TRANSACTION

        return ActivityType.UNKNOWN

    def _parse_transaction(
        self, tx_data: dict, stablecoin: str
    ) -> Optional[Transaction]:
        """Parse a transaction from API response data.

        Args:
            tx_data: Raw transaction data from API
            stablecoin: The stablecoin symbol

        Returns:
            Transaction object or None if parsing/validation fails
        """
        try:
            from_address = tx_data.get("from", "")
            to_address = tx_data.get("to", "")
            tx_hash = tx_data.get("hash", "")
            value = tx_data.get("value", "0")

            # Validate using BlockchainDataValidator if available
            validator = _get_blockchain_validator()
            if validator:
                if tx_hash and not validator.validate_tx_hash(tx_hash):
                    logger.warning(
                        f"Skipping record with invalid field: tx_hash",
                        extra={"explorer": self.name, "field": "tx_hash"},
                    )
                    return None

                if from_address and not validator.validate_address(from_address):
                    logger.warning(
                        f"Skipping record with invalid field: from_address",
                        extra={"explorer": self.name, "field": "from_address"},
                    )
                    return None

                if to_address and not validator.validate_address(to_address):
                    logger.warning(
                        f"Skipping record with invalid field: to_address",
                        extra={"explorer": self.name, "field": "to_address"},
                    )
                    return None

                from_address = validator.normalize_address(from_address)
                to_address = validator.normalize_address(to_address)

            amount = self._parse_amount(value, stablecoin)
            activity_type = self._classify_activity(from_address, to_address, amount)

            gas_used = None
            gas_price = None
            if tx_data.get("gasUsed"):
                gas_used = int(tx_data["gasUsed"])
            if tx_data.get("gasPrice"):
                gas_price = Decimal(tx_data["gasPrice"]) / Decimal(10**9)

            return Transaction(
                transaction_hash=tx_hash,
                block_number=int(tx_data.get("blockNumber", 0)),
                timestamp=self._parse_timestamp(tx_data.get("timeStamp", "0")),
                from_address=from_address,
                to_address=to_address,
                amount=amount,
                stablecoin=stablecoin,
                chain=self.chain,
                activity_type=activity_type,
                source_explorer=self.name,
                gas_used=gas_used,
                gas_price=gas_price,
            )
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(
                f"Failed to parse transaction from {self.name}: {e}",
                extra={"explorer": self.name, "error": str(e)},
            )
            return None

    async def fetch_stablecoin_transactions(
        self,
        stablecoin: str,
        contract_address: str,
        limit: int = 1000,
        run_id: Optional[str] = None,
    ) -> list[Transaction]:
        """Fetch stablecoin transactions from the explorer."""
        transactions: list[Transaction] = []

        log_extra = {
            "explorer": self.name,
            "stablecoin": stablecoin,
            "contract": contract_address,
        }
        if run_id:
            log_extra["run_id"] = run_id

        params = {
            "module": "account",
            "action": "tokentx",
            "contractaddress": contract_address,
            "page": 1,
            "offset": min(limit, 10000),
            "sort": "desc",
        }

        logger.debug(
            f"Fetching {stablecoin} transactions from {self.name}", extra=log_extra
        )

        response = await self._make_request(params, run_id, endpoint="tokentx")

        if response is None:
            logger.error(
                f"Failed to fetch {stablecoin} transactions from {self.name}",
                extra=log_extra,
            )
            return transactions

        result = response.get("result", [])

        if isinstance(result, str):
            if "No transactions found" in result:
                logger.info(
                    f"No {stablecoin} transactions found on {self.name}",
                    extra=log_extra,
                )
            else:
                logger.warning(
                    f"Unexpected result from {self.name}: {result}", extra=log_extra
                )
            return transactions

        for tx_data in result[:limit]:
            tx = self._parse_transaction(tx_data, stablecoin)
            if tx:
                transactions.append(tx)

        logger.info(
            f"Parsed {len(transactions)} {stablecoin} transactions from {self.name}",
            extra={**log_extra, "count": len(transactions)},
        )

        return transactions

    def _parse_holder(
        self, holder_data: dict, stablecoin: str, contract_address: str
    ) -> Optional[Holder]:
        """Parse a holder from API response data."""
        try:
            address = holder_data.get("TokenHolderAddress", "")
            balance_str = holder_data.get("TokenHolderQuantity", "0")

            validator = _get_blockchain_validator()
            if validator:
                if address and not validator.validate_address(address):
                    logger.warning(
                        f"Skipping record with invalid field: address",
                        extra={"explorer": self.name, "field": "address"},
                    )
                    return None
                address = validator.normalize_address(address)

            balance = self._parse_amount(balance_str, stablecoin)
            now = datetime.now(timezone.utc)

            return Holder(
                address=address,
                balance=balance,
                stablecoin=stablecoin,
                chain=self.chain,
                first_seen=now,
                last_activity=now,
                is_store_of_value=False,
                source_explorer=self.name,
            )
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(
                f"Failed to parse holder from {self.name}: {e}",
                extra={"explorer": self.name, "error": str(e)},
            )
            return None

    async def fetch_token_holders(
        self,
        stablecoin: str,
        contract_address: str,
        limit: int = 100,
        run_id: Optional[str] = None,
    ) -> list[Holder]:
        """Fetch token holders from the explorer."""
        holders: list[Holder] = []

        log_extra = {
            "explorer": self.name,
            "stablecoin": stablecoin,
            "contract": contract_address,
        }
        if run_id:
            log_extra["run_id"] = run_id

        params = {
            "module": "token",
            "action": "tokenholderlist",
            "contractaddress": contract_address,
            "page": 1,
            "offset": min(limit, 1000),
        }

        response = await self._make_request(params, run_id, endpoint="tokenholderlist")

        if response is None:
            logger.warning(
                f"Failed to fetch {stablecoin} holders from {self.name}",
                extra=log_extra,
            )
            return holders

        result = response.get("result", [])

        if isinstance(result, str):
            if "No token holder found" in result:
                logger.info(
                    f"No {stablecoin} holders found on {self.name}", extra=log_extra
                )
            elif "API Pro" in result or "upgrade" in result.lower():
                logger.info(
                    f"Token holder endpoint requires Pro API on {self.name}",
                    extra=log_extra,
                )
            else:
                logger.warning(
                    f"Unexpected result from {self.name}: {result}", extra=log_extra
                )
            return holders

        for holder_data in result[:limit]:
            holder = self._parse_holder(holder_data, stablecoin, contract_address)
            if holder:
                holders.append(holder)

        logger.info(
            f"Parsed {len(holders)} {stablecoin} holders from {self.name}",
            extra={**log_extra, "count": len(holders)},
        )

        return holders

    async def collect_all(
        self,
        stablecoins: dict[str, str],
        max_records: int = 1000,
        run_id: Optional[str] = None,
    ) -> ExplorerData:
        """Collect all data from this explorer for the given stablecoins.

        Implements graceful error handling to return partial results
        when some stablecoins fail to collect. Errors are logged and
        stored in the result for reporting.

        Args:
            stablecoins: Dict mapping stablecoin symbol to contract address
            max_records: Maximum records per stablecoin
            run_id: Optional run ID for logging correlation

        Returns:
            ExplorerData containing all collected transactions and holders,
            including any errors encountered during collection
        """
        import time

        start_time = time.time()

        result = ExplorerData(explorer_name=self.name, chain=self.chain)

        log_extra = {"explorer": self.name, "chain": self.chain}
        if run_id:
            log_extra["run_id"] = run_id

        logger.info(
            f"Starting data collection from {self.name} for {len(stablecoins)} stablecoins",
            extra={**log_extra, "stablecoins": list(stablecoins.keys())},
        )

        successful_coins = []
        failed_coins = []

        for stablecoin, contract_address in stablecoins.items():
            coin_success = True
            coin_log_extra = {
                **log_extra,
                "stablecoin": stablecoin,
                "contract": contract_address,
            }

            # Fetch transactions with error handling
            try:
                logger.info(
                    f"Fetching {stablecoin} transactions from {self.name}",
                    extra=coin_log_extra,
                )

                transactions = await self.fetch_stablecoin_transactions(
                    stablecoin=stablecoin,
                    contract_address=contract_address,
                    limit=max_records,
                    run_id=run_id,
                )

                # Validate and filter transactions
                valid_transactions = self._validate_transactions(
                    transactions, stablecoin, run_id
                )
                result.transactions.extend(valid_transactions)

                logger.info(
                    f"Fetched {len(valid_transactions)} valid {stablecoin} transactions from {self.name}",
                    extra={
                        **coin_log_extra,
                        "total_fetched": len(transactions),
                        "valid_count": len(valid_transactions),
                        "skipped": len(transactions) - len(valid_transactions),
                    },
                )

            except asyncio.TimeoutError as e:
                coin_success = False
                error_msg = (
                    f"Timeout fetching {stablecoin} transactions from {self.name}"
                )
                result.errors.append(error_msg)
                logger.warning(
                    error_msg, extra={**coin_log_extra, "error_type": "timeout"}
                )
            except aiohttp.ClientError as e:
                coin_success = False
                error_msg = f"Network error fetching {stablecoin} transactions: {e}"
                result.errors.append(error_msg)
                logger.warning(
                    error_msg,
                    extra={**coin_log_extra, "error_type": "network", "error": str(e)},
                )
            except Exception as e:
                coin_success = False
                error_msg = f"Error fetching {stablecoin} transactions: {e}"
                result.errors.append(error_msg)
                logger.error(
                    error_msg,
                    extra={
                        **coin_log_extra,
                        "error_type": type(e).__name__,
                        "error": str(e),
                    },
                    exc_info=True,
                )

            # Fetch holders with error handling
            try:
                logger.info(
                    f"Fetching {stablecoin} holders from {self.name}",
                    extra=coin_log_extra,
                )

                holders = await self.fetch_token_holders(
                    stablecoin=stablecoin,
                    contract_address=contract_address,
                    limit=min(max_records, 100),  # Holders typically limited
                    run_id=run_id,
                )

                # Validate and filter holders
                valid_holders = self._validate_holders(holders, stablecoin, run_id)
                result.holders.extend(valid_holders)

                logger.info(
                    f"Fetched {len(valid_holders)} valid {stablecoin} holders from {self.name}",
                    extra={
                        **coin_log_extra,
                        "total_fetched": len(holders),
                        "valid_count": len(valid_holders),
                        "skipped": len(holders) - len(valid_holders),
                    },
                )

            except asyncio.TimeoutError as e:
                # Don't mark as failed for holder timeout - transactions may have succeeded
                error_msg = f"Timeout fetching {stablecoin} holders from {self.name}"
                result.errors.append(error_msg)
                logger.warning(
                    error_msg, extra={**coin_log_extra, "error_type": "timeout"}
                )
            except aiohttp.ClientError as e:
                error_msg = f"Network error fetching {stablecoin} holders: {e}"
                result.errors.append(error_msg)
                logger.warning(
                    error_msg,
                    extra={**coin_log_extra, "error_type": "network", "error": str(e)},
                )
            except Exception as e:
                error_msg = f"Error fetching {stablecoin} holders: {e}"
                result.errors.append(error_msg)
                logger.error(
                    error_msg,
                    extra={
                        **coin_log_extra,
                        "error_type": type(e).__name__,
                        "error": str(e),
                    },
                    exc_info=True,
                )

            if coin_success:
                successful_coins.append(stablecoin)
            else:
                failed_coins.append(stablecoin)

        result.collection_time_seconds = time.time() - start_time

        # Log summary with success/failure breakdown
        log_level = "info" if not failed_coins else "warning"
        getattr(logger, log_level)(
            f"Completed collection from {self.name}: {result.total_records} records in {result.collection_time_seconds:.2f}s",
            extra={
                **log_extra,
                "total_records": result.total_records,
                "transactions": len(result.transactions),
                "holders": len(result.holders),
                "errors": len(result.errors),
                "successful_coins": successful_coins,
                "failed_coins": failed_coins,
                "duration_seconds": result.collection_time_seconds,
                "partial_success": bool(failed_coins and successful_coins),
            },
        )

        return result

    def _validate_transactions(
        self,
        transactions: list[Transaction],
        stablecoin: str,
        run_id: Optional[str] = None,
    ) -> list[Transaction]:
        """Validate transactions and skip invalid records with logging.

        Args:
            transactions: List of transactions to validate
            stablecoin: Stablecoin symbol for logging
            run_id: Optional run ID for logging correlation

        Returns:
            List of valid transactions
        """
        valid = []
        for tx in transactions:
            if self._is_valid_transaction(tx):
                valid.append(tx)
            else:
                logger.debug(
                    f"Skipping invalid transaction from {self.name}",
                    extra={
                        "explorer": self.name,
                        "stablecoin": stablecoin,
                        "tx_hash": tx.transaction_hash[:20]
                        if tx.transaction_hash
                        else "unknown",
                        "run_id": run_id,
                    },
                )
        return valid

    def _is_valid_transaction(self, tx: Transaction) -> bool:
        """Check if a transaction has all required fields.

        Args:
            tx: Transaction to validate

        Returns:
            True if valid, False otherwise
        """
        return bool(
            tx.transaction_hash
            and tx.from_address
            and tx.to_address
            and tx.timestamp
            and tx.stablecoin
            and tx.chain
        )

    def _validate_holders(
        self, holders: list[Holder], stablecoin: str, run_id: Optional[str] = None
    ) -> list[Holder]:
        """Validate holders and skip invalid records with logging.

        Args:
            holders: List of holders to validate
            stablecoin: Stablecoin symbol for logging
            run_id: Optional run ID for logging correlation

        Returns:
            List of valid holders
        """
        valid = []
        for holder in holders:
            if self._is_valid_holder(holder):
                valid.append(holder)
            else:
                logger.debug(
                    f"Skipping invalid holder from {self.name}",
                    extra={
                        "explorer": self.name,
                        "stablecoin": stablecoin,
                        "address": holder.address[:20] if holder.address else "unknown",
                        "run_id": run_id,
                    },
                )
        return valid

    def _is_valid_holder(self, holder: Holder) -> bool:
        """Check if a holder has all required fields.

        Args:
            holder: Holder to validate

        Returns:
            True if valid, False otherwise
        """
        return bool(
            holder.address
            and holder.stablecoin
            and holder.chain
            and holder.balance is not None
        )
