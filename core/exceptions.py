"""Custom exceptions for the blockchain stablecoin explorer.

Provides a hierarchy of exceptions for different error categories
to enable proper error handling and logging throughout the application.
"""

from typing import Any, Optional


class StablecoinExplorerError(Exception):
    """Base exception for all stablecoin explorer errors."""
    
    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None
    ):
        """Initialize the exception.
        
        Args:
            message: Human-readable error message.
            details: Optional dictionary with additional error context.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


# ==================== Network Errors ====================

class NetworkError(StablecoinExplorerError):
    """Base exception for network-related errors."""
    pass


class ConnectionError(NetworkError):
    """Raised when unable to connect to an external service."""
    
    def __init__(
        self,
        service: str,
        message: str,
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            message=f"Connection to {service} failed: {message}",
            details={
                "service": service,
                "original_error": str(original_error) if original_error else None,
            }
        )
        self.service = service
        self.original_error = original_error


class TimeoutError(NetworkError):
    """Raised when a request times out."""
    
    def __init__(
        self,
        service: str,
        timeout_seconds: float,
        operation: str = "request"
    ):
        super().__init__(
            message=f"{operation.capitalize()} to {service} timed out after {timeout_seconds}s",
            details={
                "service": service,
                "timeout_seconds": timeout_seconds,
                "operation": operation,
            }
        )
        self.service = service
        self.timeout_seconds = timeout_seconds


class RateLimitError(NetworkError):
    """Raised when rate limited by an external service."""
    
    def __init__(
        self,
        service: str,
        retry_after_seconds: Optional[int] = None
    ):
        message = f"Rate limited by {service}"
        if retry_after_seconds:
            message += f", retry after {retry_after_seconds}s"
        
        super().__init__(
            message=message,
            details={
                "service": service,
                "retry_after_seconds": retry_after_seconds,
            }
        )
        self.service = service
        self.retry_after_seconds = retry_after_seconds


# ==================== API Errors ====================

class APIError(StablecoinExplorerError):
    """Base exception for API-related errors."""
    pass


class APIResponseError(APIError):
    """Raised when an API returns an error response."""
    
    def __init__(
        self,
        service: str,
        status_code: Optional[int] = None,
        error_message: Optional[str] = None,
        response_body: Optional[str] = None
    ):
        message = f"API error from {service}"
        if status_code:
            message += f" (HTTP {status_code})"
        if error_message:
            message += f": {error_message}"
        
        super().__init__(
            message=message,
            details={
                "service": service,
                "status_code": status_code,
                "error_message": error_message,
                "response_body": response_body[:500] if response_body else None,
            }
        )
        self.service = service
        self.status_code = status_code
        self.error_message = error_message


class APIKeyError(APIError):
    """Raised when there's an issue with API key authentication."""
    
    def __init__(self, service: str, message: str = "Invalid or missing API key"):
        super().__init__(
            message=f"API key error for {service}: {message}",
            details={"service": service}
        )
        self.service = service



# ==================== Data Errors ====================

class DataError(StablecoinExplorerError):
    """Base exception for data-related errors."""
    pass


class DataValidationError(DataError):
    """Raised when data fails validation."""
    
    def __init__(
        self,
        field: str,
        value: Any,
        reason: str,
        record_type: str = "record"
    ):
        super().__init__(
            message=f"Validation failed for {record_type}.{field}: {reason}",
            details={
                "field": field,
                "value": str(value)[:100],  # Truncate long values
                "reason": reason,
                "record_type": record_type,
            }
        )
        self.field = field
        self.value = value
        self.reason = reason


class DataParsingError(DataError):
    """Raised when data cannot be parsed."""
    
    def __init__(
        self,
        source: str,
        data_type: str,
        reason: str,
        raw_data: Optional[str] = None
    ):
        super().__init__(
            message=f"Failed to parse {data_type} from {source}: {reason}",
            details={
                "source": source,
                "data_type": data_type,
                "reason": reason,
                "raw_data": raw_data[:200] if raw_data else None,
            }
        )
        self.source = source
        self.data_type = data_type


# ==================== Collection Errors ====================

class CollectionError(StablecoinExplorerError):
    """Base exception for data collection errors."""
    pass


class ExplorerError(CollectionError):
    """Raised when a blockchain explorer fails."""
    
    def __init__(
        self,
        explorer: str,
        chain: str,
        message: str,
        partial_data: bool = False
    ):
        super().__init__(
            message=f"Explorer {explorer} ({chain}) error: {message}",
            details={
                "explorer": explorer,
                "chain": chain,
                "partial_data": partial_data,
            }
        )
        self.explorer = explorer
        self.chain = chain
        self.partial_data = partial_data


class AllExplorersFailedError(CollectionError):
    """Raised when all explorers fail to collect data."""
    
    def __init__(self, explorer_errors: list[str]):
        super().__init__(
            message="All blockchain explorers failed to collect data",
            details={"explorer_errors": explorer_errors}
        )
        self.explorer_errors = explorer_errors


# ==================== Configuration Errors ====================

class ConfigurationError(StablecoinExplorerError):
    """Raised when there's a configuration error."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            message=f"Configuration error: {message}",
            details={"config_key": config_key} if config_key else {}
        )
        self.config_key = config_key


# ==================== Authentication Errors ====================

class AuthenticationError(StablecoinExplorerError):
    """Raised when authentication fails."""
    pass


class TokenValidationError(AuthenticationError):
    """Raised when JWT token validation fails."""
    
    def __init__(self, reason: str):
        super().__init__(
            message=f"Token validation failed: {reason}",
            details={"reason": reason}
        )
        self.reason = reason


class PermissionDeniedError(AuthenticationError):
    """Raised when user lacks required permissions."""
    
    def __init__(self, required_permission: str, user_id: Optional[str] = None):
        super().__init__(
            message=f"Permission denied: requires '{required_permission}'",
            details={
                "required_permission": required_permission,
                "user_id": user_id,
            }
        )
        self.required_permission = required_permission
