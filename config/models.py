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
    
    timeout_hours: int = Field(default=24, ge=1, le=168, description="Session timeout in hours")
    cookie_secure: bool = Field(default=True, description="Secure cookie flag")
    cookie_httponly: bool = Field(default=True, description="HttpOnly cookie flag")
    cookie_samesite: Literal["lax", "strict", "none"] = Field(
        default="lax",
        description="SameSite cookie attribute"
    )


class Config(BaseModel):
    """Main configuration model."""
    
    explorers: List[ExplorerConfig] = Field(..., description="List of blockchain explorers")
    stablecoins: Dict[str, StablecoinConfig] = Field(..., description="Stablecoin contract addresses")
    auth0: Auth0Config = Field(..., description="Auth0 configuration")
    database: DatabaseConfig = Field(..., description="Database configuration")
    app: AppConfig = Field(..., description="Application configuration")
    output: OutputConfig = Field(default_factory=OutputConfig, description="Output configuration")
    retry: RetryConfig = Field(default_factory=RetryConfig, description="Retry configuration")
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig, description="Rate limit configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    cors: CORSConfig = Field(default_factory=CORSConfig, description="CORS configuration")
    session: SessionConfig = Field(default_factory=SessionConfig, description="Session configuration")
    
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
