"""Security components for the blockchain stablecoin explorer.

This package provides security hardening components:

Currently implemented:
- CredentialSanitizer: Detects and redacts credentials from strings/dicts/URLs
- SecureLogger: Logger wrapper with automatic credential sanitization
- DomainAllowlist: Validates domains against configurable allowlist
- SSRFProtector: Protects against SSRF attacks with IP blocking
- ResourceLimiter: Monitors and enforces resource consumption limits
- GracefulTerminator: Handles graceful shutdown on timeout/resource exhaustion
- CircuitBreaker: Circuit breaker pattern for API resilience
- ExponentialBackoff: Exponential backoff with jitter for retries
- BlockchainDataValidator: Validates blockchain data formats
- TimeoutManager: Manages hierarchical timeouts for agent runs

Additional components will be added as the security hardening spec progresses:
- Schema validation
- Secure HTTP client
"""

from core.security.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitBreakerState,
    ExponentialBackoff,
)
from core.security.credential_sanitizer import CredentialSanitizer
from core.security.graceful_terminator import (
    ExplorerDataOutput,
    GracefulTerminator,
    TerminationReport,
)
from core.security.resource_limiter import (
    CPUTimeLimitExceededError,
    FileTooLargeError,
    MemoryLimitExceededError,
    ResourceLimitError,
    ResourceLimiter,
    ResponseTooLargeError,
)
from core.security.secure_logger import SecureLogger
from core.security.ssrf_protector import (
    DomainAllowlist,
    DomainNotAllowedError,
    DNSRebindingError,
    DNSResolutionError,
    PrivateIPError,
    ProtocolNotAllowedError,
    SSRFError,
    SSRFProtector,
)
from core.security.blockchain_validator import (
    BlockchainDataValidator,
    BlockchainValidationError,
    ValidationResult,
)
from core.security.safe_path_handler import (
    InvalidFilenameError,
    PathTraversalError,
    SafePathHandler,
)
from core.security.timeout_manager import (
    CollectionTimeoutError,
    OverallTimeoutError,
    TimeoutError,
    TimeoutManager,
)
from core.security.schema_validator import (
    ResponseSchemaValidator,
    SchemaFallbackStrategy,
    SchemaLoadError,
    SchemaValidationError,
    SchemaVersionClassification,
    ValidationResult as SchemaValidationResult,
)

__all__ = [
    # Circuit breaker and retry
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "CircuitBreakerState",
    "ExponentialBackoff",
    # Credential sanitization
    "CredentialSanitizer",
    "SecureLogger",
    # SSRF protection
    "DomainAllowlist",
    "DomainNotAllowedError",
    "DNSRebindingError",
    "DNSResolutionError",
    "PrivateIPError",
    "ProtocolNotAllowedError",
    "SSRFError",
    "SSRFProtector",
    # Resource limiting
    "CPUTimeLimitExceededError",
    "FileTooLargeError",
    "MemoryLimitExceededError",
    "ResourceLimitError",
    "ResourceLimiter",
    "ResponseTooLargeError",
    # Graceful termination
    "ExplorerDataOutput",
    "GracefulTerminator",
    "TerminationReport",
    # Blockchain data validation
    "BlockchainDataValidator",
    "BlockchainValidationError",
    "ValidationResult",
    # Safe file path handling
    "InvalidFilenameError",
    "PathTraversalError",
    "SafePathHandler",
    # Timeout management
    "CollectionTimeoutError",
    "OverallTimeoutError",
    "TimeoutError",
    "TimeoutManager",
    # Schema validation
    "ResponseSchemaValidator",
    "SchemaFallbackStrategy",
    "SchemaLoadError",
    "SchemaValidationError",
    "SchemaValidationResult",
    "SchemaVersionClassification",
]
