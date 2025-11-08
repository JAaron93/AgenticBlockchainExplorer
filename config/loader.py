"""Configuration loader that reads from JSON file and environment variables."""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from pydantic import ValidationError

from config.models import (
    Config,
    ExplorerConfig,
    StablecoinConfig,
    Auth0Config,
    DatabaseConfig,
    AppConfig,
    OutputConfig,
    RetryConfig,
    RateLimitConfig,
    LoggingConfig,
    CORSConfig,
    SessionConfig,
)


class ConfigurationManager:
    """Manages loading and validation of configuration from multiple sources."""
    
    def __init__(self, config_path: Optional[str] = None, env_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to JSON configuration file (default: ./config.json)
            env_file: Path to .env file (default: ./.env)
        """
        self.config_path = config_path or "./config.json"
        self.env_file = env_file or "./.env"
        self._config: Optional[Config] = None
    
    def load_config(self) -> Config:
        """
        Load configuration from JSON file and environment variables.
        
        Environment variables take precedence over JSON file values.
        
        Returns:
            Validated Config object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If configuration is invalid
            json.JSONDecodeError: If JSON is malformed
        """
        # Load environment variables
        if Path(self.env_file).exists():
            load_dotenv(self.env_file)
        
        # Load JSON configuration
        config_data = self._load_json_config()
        
        # Override with environment variables
        config_data = self._override_with_env(config_data)
        
        # Validate and create Config object
        try:
            self._config = Config(**config_data)
            return self._config
        except ValidationError:
            raise
    
    def _load_json_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        config_path = Path(self.config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(config_path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON in configuration file: {e.msg}",
                    e.doc,
                    e.pos
                )
    
    def _override_with_env(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override configuration values with environment variables.
        
        Environment variables follow the pattern:
        - AUTH0_DOMAIN -> auth0.domain
        - DATABASE_URL -> database.url
        - ETHERSCAN_API_KEY -> explorers[name=etherscan].api_key
        """
        # Auth0 configuration
        if "auth0" not in config_data:
            config_data["auth0"] = {}
        
        auth0_env_mapping = {
            "AUTH0_DOMAIN": "domain",
            "AUTH0_CLIENT_ID": "client_id",
            "AUTH0_CLIENT_SECRET": "client_secret",
            "AUTH0_AUDIENCE": "audience",
            "AUTH0_CALLBACK_URL": "callback_url",
            "AUTH0_LOGOUT_URL": "logout_url",
        }
        
        for env_var, config_key in auth0_env_mapping.items():
            value = os.getenv(env_var)
            if value:
                config_data["auth0"][config_key] = value
        
        # Database configuration
        if "database" not in config_data:
            config_data["database"] = {}
        
        if os.getenv("DATABASE_URL"):
            config_data["database"]["url"] = os.getenv("DATABASE_URL")
        if os.getenv("DATABASE_POOL_SIZE"):
            try:
                config_data["database"]["pool_size"] = int(os.getenv("DATABASE_POOL_SIZE"))
            except ValueError as e:
                raise ValueError(f"Invalid DATABASE_POOL_SIZE: must be an integer") from e
        if os.getenv("DATABASE_MAX_OVERFLOW"):
            try:
                config_data["database"]["max_overflow"] = int(os.getenv("DATABASE_MAX_OVERFLOW"))
            except ValueError as e:
                raise ValueError(f"Invalid DATABASE_MAX_OVERFLOW: must be an integer") from e
        
        # Application configuration
        if "app" not in config_data:
            config_data["app"] = {}
        
        app_env_mapping = {
            "APP_ENV": "env",
            "APP_HOST": "host",
            "APP_PORT": ("port", int),
            "APP_DEBUG": ("debug", lambda x: x.lower() in ["true", "1", "yes"]),
            "SECRET_KEY": "secret_key",
        }
        
        for env_var, config_key in app_env_mapping.items():
            value = os.getenv(env_var)
            if value:
                if isinstance(config_key, tuple):
                    key, converter = config_key
                    config_data["app"][key] = converter(value)
                else:
                    config_data["app"][config_key] = value
        
        # Output configuration
        if "output" not in config_data:
            config_data["output"] = {}
        
        if os.getenv("OUTPUT_DIRECTORY"):
            config_data["output"]["directory"] = os.getenv("OUTPUT_DIRECTORY")
        if os.getenv("MAX_RECORDS_PER_EXPLORER"):
            config_data["output"]["max_records_per_explorer"] = int(os.getenv("MAX_RECORDS_PER_EXPLORER"))
        
        # Retry configuration
        if "retry" not in config_data:
            config_data["retry"] = {}
        
        if os.getenv("RETRY_MAX_ATTEMPTS"):
            config_data["retry"]["max_attempts"] = int(os.getenv("RETRY_MAX_ATTEMPTS"))
        if os.getenv("RETRY_BACKOFF_SECONDS"):
            config_data["retry"]["backoff_seconds"] = int(os.getenv("RETRY_BACKOFF_SECONDS"))
        if os.getenv("REQUEST_TIMEOUT_SECONDS"):
            config_data["retry"]["request_timeout_seconds"] = int(os.getenv("REQUEST_TIMEOUT_SECONDS"))
        if os.getenv("MAX_CONCURRENT_REQUESTS"):
            config_data["retry"]["max_concurrent_requests"] = int(os.getenv("MAX_CONCURRENT_REQUESTS"))
        
        # Rate limit configuration
        if "rate_limit" not in config_data:
            config_data["rate_limit"] = {}
        
        if os.getenv("RATE_LIMIT_PER_MINUTE"):
            config_data["rate_limit"]["per_minute"] = int(os.getenv("RATE_LIMIT_PER_MINUTE"))
        
        # Logging configuration
        if "logging" not in config_data:
            config_data["logging"] = {}
        
        if os.getenv("LOG_LEVEL"):
            config_data["logging"]["level"] = os.getenv("LOG_LEVEL")
        if os.getenv("LOG_FORMAT"):
            config_data["logging"]["format"] = os.getenv("LOG_FORMAT")
        
        # CORS configuration
        if "cors" not in config_data:
            config_data["cors"] = {}
        
        if os.getenv("CORS_ALLOWED_ORIGINS"):
            origins = os.getenv("CORS_ALLOWED_ORIGINS").split(",")
            config_data["cors"]["allowed_origins"] = [o.strip() for o in origins]
        if os.getenv("CORS_ALLOW_CREDENTIALS"):
            config_data["cors"]["allow_credentials"] = os.getenv("CORS_ALLOW_CREDENTIALS").lower() in ["true", "1", "yes"]
        
        # Session configuration
        if "session" not in config_data:
            config_data["session"] = {}
        
        if os.getenv("SESSION_TIMEOUT_HOURS"):
            config_data["session"]["timeout_hours"] = int(os.getenv("SESSION_TIMEOUT_HOURS"))
        if os.getenv("SESSION_COOKIE_SECURE"):
            config_data["session"]["cookie_secure"] = os.getenv("SESSION_COOKIE_SECURE").lower() in ["true", "1", "yes"]
        if os.getenv("SESSION_COOKIE_HTTPONLY"):
            config_data["session"]["cookie_httponly"] = os.getenv("SESSION_COOKIE_HTTPONLY").lower() in ["true", "1", "yes"]
        if os.getenv("SESSION_COOKIE_SAMESITE"):
            config_data["session"]["cookie_samesite"] = os.getenv("SESSION_COOKIE_SAMESITE")
        
        # Explorer API keys from environment
        if "explorers" in config_data:
            for explorer in config_data["explorers"]:
                explorer_name = explorer.get("name", "").upper()
                api_key_env = f"{explorer_name}_API_KEY"
                if os.getenv(api_key_env):
                    explorer["api_key"] = os.getenv(api_key_env)
        
        # Stablecoin addresses from environment
        if "stablecoins" in config_data:
            for coin_name, coin_config in config_data["stablecoins"].items():
                for chain in ["ethereum", "bsc", "polygon"]:
                    env_var = f"{coin_name}_{chain.upper()}"
                    if os.getenv(env_var):
                        coin_config[chain] = os.getenv(env_var)
        
        return config_data
    
    def validate_config(self, config: Config) -> bool:
        """
        Validate configuration object.
        
        Args:
            config: Config object to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If configuration is invalid
        """
        # Pydantic already validates on creation, but we can add additional checks
        
        # Ensure output directory is writable
        output_dir = Path(config.output.directory)
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValueError(f"Cannot create output directory: {e}")
        
        if not os.access(output_dir, os.W_OK):
            raise ValueError(f"Output directory is not writable: {output_dir}")
        
        # Validate explorer chains match stablecoin chains
        explorer_chains = {explorer.chain for explorer in config.explorers}
        
        # Collect all chains required by stablecoins
        required_chains = set()
        for coin_name, coin_config in config.stablecoins.items():
            # Get chain names from the StablecoinConfig fields
            # StablecoinConfig has ethereum, bsc, and polygon fields
            coin_dict = coin_config.model_dump()
            for chain_name in coin_dict.keys():
                required_chains.add(chain_name)
        
        # Check if all required chains have explorers configured
        missing_chains = required_chains - explorer_chains
        if missing_chains:
            raise ValueError(
                f"Missing explorer configurations for chains: {', '.join(sorted(missing_chains))}. "
                f"Stablecoins require explorers for: {', '.join(sorted(required_chains))}, "
                f"but only found explorers for: {', '.join(sorted(explorer_chains))}"
            )
        
        return True
    
    def get_explorer_configs(self) -> list[ExplorerConfig]:
        """Get list of explorer configurations."""
        if not self._config:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._config.explorers
    
    def get_stablecoin_addresses(self) -> Dict[str, StablecoinConfig]:
        """Get stablecoin contract addresses."""
        if not self._config:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._config.stablecoins
    
    def get_explorer_by_name(self, name: str) -> Optional[ExplorerConfig]:
        """Get explorer configuration by name."""
        if not self._config:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        
        for explorer in self._config.explorers:
            if explorer.name.lower() == name.lower():
                return explorer
        
        return None
    
    def get_explorer_by_chain(self, chain: str) -> Optional[ExplorerConfig]:
        """Get explorer configuration by chain."""
        if not self._config:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        
        for explorer in self._config.explorers:
            if explorer.chain.lower() == chain.lower():
                return explorer
        
        return None
    
    @property
    def config(self) -> Config:
        """Get the loaded configuration."""
        if not self._config:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._config
