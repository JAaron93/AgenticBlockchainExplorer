"""Pydantic models for configuration schema validation."""

import logging
from typing import ClassVar, Dict, List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, HttpUrl, ValidationInfo
from enum import Enum
from eth_utils import is_address, to_checksum_address

# Use standard logging here since this module is loaded before our logging is configured
logger = logging.getLogger(__name__)


class ExplorerType(str, Enum):
    """Type of blockchain explorer."""
    API = "api"
    SCRAPER = "scraper"


class ExplorerConfig(BaseModel):
    """Configuration for a blockchain explorer."""

    ALLOWED_CHAINS: ClassVar[List[str]] = ["ethereum", "bsc", "polygon"]
    
    name: str = Field(..., description="Name of the explorer (e.g., 'etherscan')")
    base_url: HttpUrl = Field(..., description="Base URL for the explorer API")
    api_key: str = Field(..., description="API key for authentication")
    type: ExplorerType = Field(default=ExplorerType.API, description="Type of explorer")
    chain: str = Field(..., description="Blockchain network (e.g., 'ethereum', 'bsc', 'polygon')")
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate explorer name is not empty."""
        if not v or not v.strip():
            raise ValueError("Explorer name cannot be empty")
        return v.lower().strip()
    
    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key is not empty."""
        if not v or not v.strip():
            raise ValueError("API key cannot be empty")
        return v.strip()
    
    @field_validator("chain")
    @classmethod
    def validate_chain(cls, v: str) -> str:
        """Validate chain name."""
        chain_lower = v.lower().strip()
        if chain_lower not in cls.ALLOWED_CHAINS:
            raise ValueError(f"Chain must be one of {cls.ALLOWED_CHAINS}, got '{v}'")
        return chain_lower


class StablecoinConfig(BaseModel):
    """Configuration for stablecoin contract addresses."""
    
    ethereum: str = Field(..., description="Contract address on Ethereum")
    bsc: str = Field(..., description="Contract address on BSC")
    polygon: str = Field(..., description="Contract address on Polygon")
    
    @field_validator("ethereum", "bsc", "polygon")
    @classmethod
    def validate_address(cls, v: str) -> str:
        """Validate Ethereum address format with EIP-55 checksum."""
        if not v or not v.strip():
            raise ValueError("Contract address cannot be empty")
        
        address = v.strip()
        
        # Validate address using eth_utils
        if not is_address(address):
            raise ValueError(f"Invalid Ethereum address: '{address}'")
        
        # Return checksummed address for normalization
        return to_checksum_address(address)


class Auth0Config(BaseModel):
    """Configuration for Auth0 authentication."""
    
    domain: str = Field(..., description="Auth0 domain")
    client_id: str = Field(..., description="Auth0 client ID")
    client_secret: str = Field(..., description="Auth0 client secret")
    audience: str = Field(..., description="Auth0 API audience/identifier")
    callback_url: HttpUrl = Field(..., description="OAuth callback URL")
    logout_url: HttpUrl = Field(..., description="Logout redirect URL")
    
    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v: str) -> str:
        """Validate Auth0 domain format."""
        if not v or not v.strip():
            raise ValueError("Auth0 domain cannot be empty")
        
        domain = v.strip()
        
        # Check if it looks like a valid Auth0 domain
        allowed_suffixes = (".auth0.com", ".us.auth0.com", ".eu.auth0.com")
        if not domain.endswith(allowed_suffixes):
            logger.warning(
                f"Auth0 domain '{domain}' does not end with standard "
                f"Auth0 suffixes {allowed_suffixes}. Custom domains are "
                f"allowed but may require additional configuration."
            )
        
        return domain
    
    @field_validator("client_id", "client_secret", "audience")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """Validate field is not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class OutputConfig(BaseModel):
    """Configuration for output settings."""
    
    directory: str = Field(default="./output", description="Output directory path")
    max_records_per_explorer: int = Field(
        default=1000,
        ge=1,
        le=100000,
        description="Maximum records to collect per explorer"
    )
    
    @field_validator("directory")
    @classmethod
    def validate_directory(cls, v: str) -> str:
        """Validate directory path."""
        if not v or not v.strip():
            raise ValueError("Output directory cannot be empty")
        return v.strip()


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""
    
    max_attempts: int = Field(default=3, ge=1, le=10, description="Maximum retry attempts")
    backoff_seconds: int = Field(default=60, ge=1, le=300, description="Backoff time in seconds")
    request_timeout_seconds: int = Field(default=30, ge=5, le=120, description="Request timeout")
    max_concurrent_requests: int = Field(default=5, ge=1, le=20, description="Max concurrent requests")


class DatabaseConfig(BaseModel):
    """Configuration for database connection."""
    
    url: str = Field(..., description="Database connection URL")
    pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, le=100, description="Max overflow connections")
    
    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate database URL."""
        if not v or not v.strip():
            raise ValueError("Database URL cannot be empty")
        
        url = v.strip()
        
        # Basic validation for PostgreSQL URL
        if not url.startswith("postgresql://") and not url.startswith("postgresql+asyncpg://"):
            raise ValueError("Database URL must start with 'postgresql://' or 'postgresql+asyncpg://'")
        
        return url


class AppConfig(BaseModel):
    """Configuration for application settings."""
    
    env: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Application environment"
    )
    host: str = Field(default="0.0.0.0", description="Application host")
    port: int = Field(default=8000, ge=1, le=65535, description="Application port")
    debug: bool = Field(default=False, description="Debug mode")
    secret_key: str = Field(..., description="Secret key for session management")
    
    @field_validator("debug")
    @classmethod
    def validate_debug(cls, v: bool, info: ValidationInfo) -> bool:
        """Warn if debug mode is enabled in production."""
        if v and info.data.get("env") == "production":
            logger.warning(
                "Debug mode is enabled in production environment. "
                "This could expose sensitive information and should be disabled."
            )
        return v
    
    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Validate secret key."""
        if not v or not v.strip():
            raise ValueError("Secret key cannot be empty")
        
        key = v.strip()
        
        if len(key) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        
        return key


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting."""
    
    per_minute: int = Field(default=100, ge=1, le=10000, description="Requests per minute per user")


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    format: Literal["json", "text"] = Field(default="json", description="Log format")


class CORSConfig(BaseModel):
    """Configuration for CORS."""
    
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins"
    )
    allow_credentials: bool = Field(default=True, description="Allow credentials")
    
    @field_validator("allowed_origins")
    @classmethod
    def validate_origins(cls, v: List[str]) -> List[str]:
        """Validate CORS origins."""
        if not v:
            raise ValueError("At least one CORS origin must be specified")
        
        # Validate each origin
        for origin in v:
            if not origin or not origin.strip():
                raise ValueError("CORS origin cannot be empty")
        
        return [origin.strip() for origin in v]


class SessionConfig(BaseModel):
    """Configuration for session management."""

    timeout_hours: int = Field(
        default=24, ge=1, le=168, description="Session timeout in hours"
    )
    cookie_secure: bool = Field(default=True, description="Secure cookie flag")
    cookie_httponly: bool = Field(
        default=True, description="HttpOnly cookie flag"
    )
    cookie_samesite: Literal["lax", "strict", "none"] = Field(
        default="lax",
        description="SameSite cookie attribute"
    )


class CredentialSanitizerConfig(BaseModel):
    """Configuration for credential sanitization.

    Defines patterns and names used to detect and redact sensitive
    credentials from logs, error messages, and API responses.

    Requirements: 1.6, 1.7, 1.8, 1.9
    """

    sensitive_param_names: List[str] = Field(
        default=[
            "apikey",
            "api_key",
            "API_KEY",
            "token",
            "auth_token",
            "secret",
            "password",
            "client_secret",
        ],
        description="Parameter names that indicate credential values",
    )
    sensitive_header_names: List[str] = Field(
        default=["Authorization", "X-API-Key", "X-Auth-Token"],
        description="HTTP header names that contain credentials",
    )
    credential_patterns: List[str] = Field(
        default=[
            r"[a-zA-Z0-9]{32,}",  # 32+ char alphanumeric (API keys)
            r"eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+",  # JWT
        ],
        description="Regex patterns matching credential formats",
    )
    redaction_placeholder: str = Field(
        default="[REDACTED]",
        description="Placeholder text to replace redacted credentials",
    )


class SSRFProtectionConfig(BaseModel):
    """Configuration for SSRF (Server-Side Request Forgery) protection.

    Defines the domain allowlist and security settings for outbound
    HTTP requests to prevent SSRF attacks.

    Requirements: 2.2, 2.3
    """

    allowed_domains: List[str] = Field(
        default=[
            "api.etherscan.io",
            "*.etherscan.io",
            "api.bscscan.com",
            "*.bscscan.com",
            "api.polygonscan.com",
            "*.polygonscan.com",
        ],
        description="List of allowed domains (supports wildcard patterns)",
    )
    require_https: bool = Field(
        default=True,
        description="Require HTTPS protocol for all outbound requests",
    )
    block_private_ips: bool = Field(
        default=True,
        description="Block requests resolving to private/internal IP ranges",
    )

    @field_validator("allowed_domains")
    @classmethod
    def validate_allowed_domains(cls, v: List[str]) -> List[str]:
        """Validate domain allowlist is not empty and patterns are valid."""
        if not v:
            raise ValueError(
                "SSRF protection requires at least one allowed domain"
            )

        validated = []
        for domain in v:
            domain = domain.strip().lower()
            if not domain:
                raise ValueError("Domain pattern cannot be empty")

            # Validate wildcard pattern syntax
            if "*" in domain:
                # Only allow wildcard at the start for subdomain matching
                if not domain.startswith("*."):
                    raise ValueError(
                        f"Invalid wildcard pattern '{domain}': "
                        "wildcard must be at start (e.g., '*.example.com')"
                    )
                # Ensure there's a valid domain after the wildcard
                base_domain = domain[2:]  # Remove "*."
                if not base_domain or "." not in base_domain:
                    raise ValueError(
                        f"Invalid wildcard pattern '{domain}': "
                        "must have valid base domain"
                    )

            validated.append(domain)

        return validated


class ResourceLimitConfig(BaseModel):
    """Configuration for resource consumption limits.

    Defines limits for response sizes, file sizes, and memory usage
    to protect against resource exhaustion attacks.

    Requirements: 3.1, 3.5, 3.6
    """

    max_response_size_bytes: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        ge=1024,  # Minimum 1KB
        le=100 * 1024 * 1024,  # Maximum 100MB
        description="Maximum allowed API response body size in bytes",
    )
    max_output_file_size_bytes: int = Field(
        default=100 * 1024 * 1024,  # 100MB
        ge=1024,  # Minimum 1KB
        le=1024 * 1024 * 1024,  # Maximum 1GB
        description="Maximum allowed output file size in bytes",
    )
    max_memory_usage_mb: int = Field(
        default=512,
        ge=64,  # Minimum 64MB
        le=8192,  # Maximum 8GB
        description="Maximum allowed process memory usage in megabytes",
    )
    max_cpu_time_seconds: int = Field(
        default=3600,  # 1 hour
        ge=60,  # Minimum 1 minute
        le=86400,  # Maximum 24 hours
        description="Maximum allowed CPU time in seconds",
    )


class TimeoutConfig(BaseModel):
    """Configuration for collection timeout enforcement.

    Defines timeouts for overall agent runs, per-stablecoin collections,
    and graceful shutdown operations.

    Requirements: 6.4, 6.5
    """

    overall_run_timeout_seconds: int = Field(
        default=1800,  # 30 minutes
        ge=60,  # Minimum 1 minute
        le=86400,  # Maximum 24 hours
        description="Maximum total runtime for the entire agent run across all stablecoins and explorers",
    )
    per_collection_timeout_seconds: int = Field(
        default=180,  # 3 minutes
        ge=30,  # Minimum 30 seconds
        le=3600,  # Maximum 1 hour
        description="Maximum time for collecting data for a single stablecoin from a single explorer",
    )
    shutdown_timeout_seconds: int = Field(
        default=30,
        ge=5,  # Minimum 5 seconds
        le=300,  # Maximum 5 minutes
        description="Maximum time to wait for graceful shutdown (file writes, cleanup)",
    )

    @model_validator(mode="after")
    def validate_timeout_relationships(self) -> "TimeoutConfig":
        """Validate that timeout values have sensible relationships.

        Ensures:
        - per_collection_timeout < overall_run_timeout
        - shutdown_timeout < overall_run_timeout
        - At least one collection can complete within overall timeout

        Note: Full validation against actual collection count happens at runtime
        in TimeoutManager, which may dynamically adjust per_collection_timeout.
        """
        if self.per_collection_timeout_seconds >= self.overall_run_timeout_seconds:
            raise ValueError(
                f"per_collection_timeout_seconds ({self.per_collection_timeout_seconds}) "
                f"must be less than overall_run_timeout_seconds ({self.overall_run_timeout_seconds})"
            )

        if self.shutdown_timeout_seconds >= self.overall_run_timeout_seconds:
            raise ValueError(
                f"shutdown_timeout_seconds ({self.shutdown_timeout_seconds}) "
                f"must be less than overall_run_timeout_seconds ({self.overall_run_timeout_seconds})"
            )

        # Ensure there's enough time for at least one collection plus shutdown
        min_required = self.per_collection_timeout_seconds + self.shutdown_timeout_seconds
        if min_required >= self.overall_run_timeout_seconds:
            raise ValueError(
                f"overall_run_timeout_seconds ({self.overall_run_timeout_seconds}) must be greater than "
                f"per_collection_timeout_seconds + shutdown_timeout_seconds ({min_required})"
            )

        return self


class SecurityConfig(BaseModel):
    """Combined security configuration for agent hardening.

    Aggregates all security-related configurations including credential
    sanitization, SSRF protection, resource limits, and timeouts.

    Requirements: 1.6, 1.7, 1.8, 1.9, 2.2, 2.3, 3.1, 6.4, 6.5
    """

    credential_sanitizer: CredentialSanitizerConfig = Field(
        default_factory=CredentialSanitizerConfig,
        description="Configuration for credential detection and redaction",
    )
    ssrf_protection: SSRFProtectionConfig = Field(
        default_factory=SSRFProtectionConfig,
        description="Configuration for SSRF attack prevention",
    )
    resource_limits: ResourceLimitConfig = Field(
        default_factory=ResourceLimitConfig,
        description="Configuration for resource consumption limits",
    )
    timeouts: TimeoutConfig = Field(
        default_factory=TimeoutConfig,
        description="Configuration for collection timeout enforcement",
    )


class SecurityConfig(BaseModel):
    """Combined security configuration.

    Aggregates all security-related configuration models into a single
    configuration section for the main Config model.

    Requirements: 1.6, 1.7, 1.8, 1.9, 2.2, 2.3, 3.1, 6.4, 6.5
    """

    credential_sanitizer: CredentialSanitizerConfig = Field(
        default_factory=CredentialSanitizerConfig,
        description="Configuration for credential detection and redaction",
    )
    ssrf_protection: SSRFProtectionConfig = Field(
        default_factory=SSRFProtectionConfig,
        description="Configuration for SSRF protection and domain allowlisting",
    )
    resource_limits: ResourceLimitConfig = Field(
        default_factory=ResourceLimitConfig,
        description="Configuration for resource consumption limits",
    )
    timeouts: TimeoutConfig = Field(
        default_factory=TimeoutConfig,
        description="Configuration for collection timeout enforcement",
    )


class Config(BaseModel):
    """Main configuration model."""

    explorers: List[ExplorerConfig] = Field(
        ..., description="List of blockchain explorers"
    )
    stablecoins: Dict[str, StablecoinConfig] = Field(
        ..., description="Stablecoin contract addresses"
    )
    auth0: Auth0Config = Field(..., description="Auth0 configuration")
    database: DatabaseConfig = Field(..., description="Database configuration")
    app: AppConfig = Field(..., description="Application configuration")
    output: OutputConfig = Field(
        default_factory=OutputConfig, description="Output configuration"
    )
    retry: RetryConfig = Field(
        default_factory=RetryConfig, description="Retry configuration"
    )
    rate_limit: RateLimitConfig = Field(
        default_factory=RateLimitConfig, description="Rate limit configuration"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )
    cors: CORSConfig = Field(
        default_factory=CORSConfig, description="CORS configuration"
    )
    session: SessionConfig = Field(
        default_factory=SessionConfig, description="Session configuration"
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security hardening configuration",
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security hardening configuration",
    )
    
    @field_validator("explorers")
    @classmethod
    def validate_explorers(cls, v: List[ExplorerConfig]) -> List[ExplorerConfig]:
        """Validate explorers list."""
        if not v or len(v) == 0:
            raise ValueError("At least one explorer must be configured")
        
        # Check for duplicate explorer names
        names = [explorer.name for explorer in v]
        if len(names) != len(set(names)):
            raise ValueError("Explorer names must be unique")
        
        return v
    
    @field_validator("stablecoins")
    @classmethod
    def validate_stablecoins(cls, v: Dict[str, StablecoinConfig]) -> Dict[str, StablecoinConfig]:
        """Validate stablecoins configuration."""
        if not v or len(v) == 0:
            raise ValueError("At least one stablecoin must be configured")
        
        # Check for common stablecoins
        required_stablecoins = ["USDC", "USDT"]
        for coin in required_stablecoins:
            if coin not in v:
                raise ValueError(f"Stablecoin '{coin}' must be configured")
        
        return v
    
    # Note: Explorer-stablecoin chain validation is enforced by StablecoinConfig
    # requiring all chain fields (ethereum, bsc, polygon) to be present.
    # No additional model_validator needed since Pydantic's required-field
    # validation already guarantees all chains have addresses.

    model_config = {
        "json_schema_extra": {
            "example": {
                "explorers": [
                    {
                        "name": "etherscan",
                        "base_url": "https://api.etherscan.io/api",
                        "api_key": "FAKE_API_KEY_DO_NOT_USE",
                        "type": "api",
                        "chain": "ethereum"
                    }
                ],
                "stablecoins": {
                    "USDC": {
                        "ethereum": "0x0000000000000000000000000000000000000000",
                        "bsc": "0x0000000000000000000000000000000000000000",
                        "polygon": "0x0000000000000000000000000000000000000000"
                    }
                },
                "auth0": {
                    "domain": "example.invalid",
                    "client_id": "REPLACE_WITH_AUTH0_CLIENT_ID",
                    "client_secret": "REPLACE_WITH_AUTH0_CLIENT_SECRET",
                    "audience": "https://example.invalid/api",
                    "callback_url": "http://localhost:8000/callback",
                    "logout_url": "http://localhost:8000"
                },
                "database": {
                    "url": "postgresql://user:pass@localhost:5432/REPLACE_DB",
                    "pool_size": 10,
                    "max_overflow": 20
                },
                "app": {
                    "env": "development",
                    "host": "0.0.0.0",
                    "port": 8000,
                    "debug": True,
                    "secret_key": "INVALID_SECRET_REPLACE_WITH_SECURE_32_CHAR_KEY"
                }
            }
        }
    }
