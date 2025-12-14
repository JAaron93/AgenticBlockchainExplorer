"""Configuration loader that reads from JSON file and environment variables."""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from config.models import Config, ExplorerConfig, StablecoinConfig


class ConfigurationManager:
    """Manages loading and validation of configuration from multiple sources."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        env_file: Optional[str] = None
    ):
        """Initialize configuration manager.

        Args:
            config_path: Path to JSON configuration file (default: ./config.json)
            env_file: Path to .env file (default: ./.env)
        """
        self.config_path = config_path or "./config.json"
        self.env_file = env_file or "./.env"
        self._config: Optional[Config] = None

    def load_config(self) -> Config:
        """Load configuration from JSON file and environment variables.

        Environment variables take precedence over JSON file values.

        Returns:
            Validated Config object

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If configuration is invalid
            json.JSONDecodeError: If JSON is malformed
        """
        if Path(self.env_file).exists():
            load_dotenv(self.env_file)

        config_data = self._load_json_config()
        config_data = self._override_with_env(config_data)
        self._config = Config(**config_data)
        return self._config

    def _load_json_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        config_path = Path(self.config_path)

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

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
        """Override configuration values with environment variables."""
        # Auth0 configuration - all fields required via environment variables
        if "auth0" not in config_data:
            config_data["auth0"] = {}

        auth0_mapping = {
            "AUTH0_DOMAIN": "domain",
            "AUTH0_CLIENT_ID": "client_id",
            "AUTH0_CLIENT_SECRET": "client_secret",
            "AUTH0_AUDIENCE": "audience",
            "AUTH0_CALLBACK_URL": "callback_url",
            "AUTH0_LOGOUT_URL": "logout_url",
        }

        for env_var, key in auth0_mapping.items():
            env_value = os.getenv(env_var)
            config_value = config_data.get("auth0", {}).get(key, "")

            # Check if config uses placeholder pattern ${ENV_VAR}
            if config_value == f"${{{env_var}}}" or not config_value:
                if not env_value:
                    raise ValueError(
                        f"{env_var} environment variable is required. "
                        f"Set it in your .env file or environment."
                    )
                config_data["auth0"][key] = env_value
            elif env_value:
                # Environment variable overrides config file
                config_data["auth0"][key] = env_value

        # Database configuration
        if "database" not in config_data:
            config_data["database"] = {}

        db_url_env = os.getenv("DATABASE_URL")
        db_url_config = config_data.get("database", {}).get("url", "")

        if db_url_config == "${DATABASE_URL}" or not db_url_config:
            if not db_url_env:
                raise ValueError(
                    "DATABASE_URL environment variable is required. "
                    "Set it in your .env file or environment. "
                    "Expected format: postgresql://user:password@host:port/dbname"
                )
            config_data["database"]["url"] = db_url_env
        elif db_url_env:
            config_data["database"]["url"] = db_url_env

        pool_size = os.getenv("DATABASE_POOL_SIZE")
        if pool_size:
            try:
                config_data["database"]["pool_size"] = int(pool_size)
            except ValueError:
                raise ValueError("Invalid DATABASE_POOL_SIZE: must be an integer")

        max_overflow = os.getenv("DATABASE_MAX_OVERFLOW")
        if max_overflow:
            try:
                config_data["database"]["max_overflow"] = int(max_overflow)
            except ValueError:
                raise ValueError("Invalid DATABASE_MAX_OVERFLOW: must be integer")

        # Application configuration
        if "app" not in config_data:
            config_data["app"] = {}

        app_mapping = {
            "APP_ENV": ("env", str),
            "APP_HOST": ("host", str),
            "APP_PORT": ("port", int),
            "APP_DEBUG": ("debug", lambda x: x.lower() in ["true", "1", "yes"]),
        }

        for env_var, (key, converter) in app_mapping.items():
            value = os.getenv(env_var)
            if value:
                config_data["app"][key] = converter(value)

        # SECRET_KEY - required via environment variable when placeholder used
        secret_key_env = os.getenv("SECRET_KEY")
        secret_key_config = config_data.get("app", {}).get("secret_key", "")

        if secret_key_config == "${SECRET_KEY}":
            if not secret_key_env:
                raise ValueError(
                    "SECRET_KEY environment variable is required. "
                    "Generate with: python -c \"import secrets; "
                    "print(secrets.token_urlsafe(32))\""
                )
            config_data["app"]["secret_key"] = secret_key_env
        elif secret_key_env:
            config_data["app"]["secret_key"] = secret_key_env

        # Output configuration
        if "output" not in config_data:
            config_data["output"] = {}

        output_dir = os.getenv("OUTPUT_DIRECTORY")
        if output_dir:
            config_data["output"]["directory"] = output_dir

        max_records = os.getenv("MAX_RECORDS_PER_EXPLORER")
        if max_records:
            try:
                config_data["output"]["max_records_per_explorer"] = int(max_records)
            except ValueError:
                raise ValueError("Invalid MAX_RECORDS_PER_EXPLORER: must be integer")

        # Retry configuration
        if "retry" not in config_data:
            config_data["retry"] = {}

        retry_mapping = {
            "RETRY_MAX_ATTEMPTS": "max_attempts",
            "RETRY_BACKOFF_SECONDS": "backoff_seconds",
            "REQUEST_TIMEOUT_SECONDS": "request_timeout_seconds",
            "MAX_CONCURRENT_REQUESTS": "max_concurrent_requests",
        }

        for env_var, key in retry_mapping.items():
            value = os.getenv(env_var)
            if value:
                try:
                    config_data["retry"][key] = int(value)
                except ValueError:
                    raise ValueError(f"Invalid {env_var}: must be an integer")

        # Rate limit configuration
        if "rate_limit" not in config_data:
            config_data["rate_limit"] = {}

        rate_limit = os.getenv("RATE_LIMIT_PER_MINUTE")
        if rate_limit:
            try:
                config_data["rate_limit"]["per_minute"] = int(rate_limit)
            except ValueError:
                raise ValueError("Invalid RATE_LIMIT_PER_MINUTE: must be integer")

        # Logging configuration
        if "logging" not in config_data:
            config_data["logging"] = {}

        log_level = os.getenv("LOG_LEVEL")
        if log_level:
            config_data["logging"]["level"] = log_level

        log_format = os.getenv("LOG_FORMAT")
        if log_format:
            config_data["logging"]["format"] = log_format

        # CORS configuration
        if "cors" not in config_data:
            config_data["cors"] = {}

        cors_origins = os.getenv("CORS_ALLOWED_ORIGINS")
        if cors_origins:
            config_data["cors"]["allowed_origins"] = [
                o.strip() for o in cors_origins.split(",")
            ]

        cors_creds = os.getenv("CORS_ALLOW_CREDENTIALS")
        if cors_creds:
            config_data["cors"]["allow_credentials"] = (
                cors_creds.lower() in ["true", "1", "yes"]
            )

        # Session configuration
        if "session" not in config_data:
            config_data["session"] = {}

        session_timeout = os.getenv("SESSION_TIMEOUT_HOURS")
        if session_timeout:
            try:
                config_data["session"]["timeout_hours"] = int(session_timeout)
            except ValueError:
                raise ValueError("Invalid SESSION_TIMEOUT_HOURS: must be integer")

        cookie_secure = os.getenv("SESSION_COOKIE_SECURE")
        if cookie_secure:
            config_data["session"]["cookie_secure"] = (
                cookie_secure.lower() in ["true", "1", "yes"]
            )

        cookie_httponly = os.getenv("SESSION_COOKIE_HTTPONLY")
        if cookie_httponly:
            config_data["session"]["cookie_httponly"] = (
                cookie_httponly.lower() in ["true", "1", "yes"]
            )

        cookie_samesite = os.getenv("SESSION_COOKIE_SAMESITE")
        if cookie_samesite:
            config_data["session"]["cookie_samesite"] = cookie_samesite

        # Security configuration
        config_data = self._override_security_config(config_data)

        # Explorer API keys from environment
        if "explorers" in config_data:
            for explorer in config_data["explorers"]:
                name = explorer.get("name", "").upper()
                env_var = f"{name}_API_KEY"
                env_value = os.getenv(env_var)
                config_value = explorer.get("api_key", "")

                # Check if config uses placeholder pattern ${ENV_VAR}
                if config_value == f"${{{env_var}}}":
                    if not env_value:
                        raise ValueError(
                            f"{env_var} environment variable is required. "
                            f"Set it in your .env file or environment."
                        )
                    explorer["api_key"] = env_value
                elif env_value:
                    # Environment variable overrides config file
                    explorer["api_key"] = env_value

        # Stablecoin addresses from environment
        if "stablecoins" in config_data:
            chains = list(StablecoinConfig.model_fields.keys())
            for coin_name, coin_config in config_data["stablecoins"].items():
                for chain in chains:
                    env_var = f"{coin_name}_{chain.upper()}"
                    value = os.getenv(env_var)
                    if value:
                        coin_config[chain] = value

        # Security configuration overrides
        config_data = self._override_security_config(config_data)

        return config_data

    def validate_config(self, config: Config) -> bool:
        """Validate configuration object.

        Args:
            config: Config object to validate

        Returns:
            bool: True if configuration is valid
        """
        output_dir = Path(config.output.directory)
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValueError(f"Cannot create output directory: {e}")

        if not os.access(output_dir, os.W_OK):
            raise ValueError(f"Output directory is not writable: {output_dir}")

        explorer_chains = {explorer.chain for explorer in config.explorers}

        required_chains = set()
        for coin_config in config.stablecoins.values():
            coin_dict = coin_config.model_dump()
            for chain_name, address in coin_dict.items():
                if address is not None:
                    required_chains.add(chain_name)

        missing_chains = required_chains - explorer_chains
        if missing_chains:
            raise ValueError(
                f"Missing explorer configurations for chains: "
                f"{', '.join(sorted(missing_chains))}. "
                f"Stablecoins require explorers for: "
                f"{', '.join(sorted(required_chains))}, "
                f"but only found explorers for: "
                f"{', '.join(sorted(explorer_chains))}"
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

    def _override_security_config(
        self, config_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Override security configuration with environment variables.

        Supports the following environment variables:
        - ALLOWED_EXPLORER_DOMAINS: Comma-separated list of allowed domains
        - MAX_RESPONSE_SIZE_BYTES: Maximum API response size in bytes
        - OVERALL_RUN_TIMEOUT_SECONDS: Overall agent run timeout

        Requirements: 2.2, 3.1, 6.4
        """
        if "security" not in config_data:
            config_data["security"] = {}

        security = config_data["security"]

        # SSRF Protection - allowed domains
        if "ssrf_protection" not in security:
            security["ssrf_protection"] = {}

        allowed_domains = os.getenv("ALLOWED_EXPLORER_DOMAINS")
        if allowed_domains:
            domains = [d.strip() for d in allowed_domains.split(",") if d.strip()]
            if not domains:
                raise ValueError(
                    "ALLOWED_EXPLORER_DOMAINS cannot be empty when set. "
                    "Provide comma-separated domain patterns."
                )
            security["ssrf_protection"]["allowed_domains"] = domains

        require_https = os.getenv("SSRF_REQUIRE_HTTPS")
        if require_https:
            security["ssrf_protection"]["require_https"] = (
                require_https.lower() in ["true", "1", "yes"]
            )

        block_private_ips = os.getenv("SSRF_BLOCK_PRIVATE_IPS")
        if block_private_ips:
            security["ssrf_protection"]["block_private_ips"] = (
                block_private_ips.lower() in ["true", "1", "yes"]
            )

        # Resource Limits
        if "resource_limits" not in security:
            security["resource_limits"] = {}

        max_response_size = os.getenv("MAX_RESPONSE_SIZE_BYTES")
        if max_response_size:
            try:
                security["resource_limits"]["max_response_size_bytes"] = int(
                    max_response_size
                )
            except ValueError:
                raise ValueError(
                    "Invalid MAX_RESPONSE_SIZE_BYTES: must be an integer"
                )

        max_output_size = os.getenv("MAX_OUTPUT_FILE_SIZE_BYTES")
        if max_output_size:
            try:
                security["resource_limits"]["max_output_file_size_bytes"] = int(
                    max_output_size
                )
            except ValueError:
                raise ValueError(
                    "Invalid MAX_OUTPUT_FILE_SIZE_BYTES: must be an integer"
                )

        max_memory = os.getenv("MAX_MEMORY_USAGE_MB")
        if max_memory:
            try:
                security["resource_limits"]["max_memory_usage_mb"] = int(max_memory)
            except ValueError:
                raise ValueError(
                    "Invalid MAX_MEMORY_USAGE_MB: must be an integer"
                )

        max_cpu = os.getenv("MAX_CPU_TIME_SECONDS")
        if max_cpu:
            try:
                security["resource_limits"]["max_cpu_time_seconds"] = int(max_cpu)
            except ValueError:
                raise ValueError(
                    "Invalid MAX_CPU_TIME_SECONDS: must be an integer"
                )

        # Timeouts
        if "timeouts" not in security:
            security["timeouts"] = {}

        overall_timeout = os.getenv("OVERALL_RUN_TIMEOUT_SECONDS")
        if overall_timeout:
            try:
                security["timeouts"]["overall_run_timeout_seconds"] = int(
                    overall_timeout
                )
            except ValueError:
                raise ValueError(
                    "Invalid OVERALL_RUN_TIMEOUT_SECONDS: must be an integer"
                )

        per_collection_timeout = os.getenv("PER_COLLECTION_TIMEOUT_SECONDS")
        if per_collection_timeout:
            try:
                security["timeouts"]["per_collection_timeout_seconds"] = int(
                    per_collection_timeout
                )
            except ValueError:
                raise ValueError(
                    "Invalid PER_COLLECTION_TIMEOUT_SECONDS: must be an integer"
                )

        shutdown_timeout = os.getenv("SHUTDOWN_TIMEOUT_SECONDS")
        if shutdown_timeout:
            try:
                security["timeouts"]["shutdown_timeout_seconds"] = int(
                    shutdown_timeout
                )
            except ValueError:
                raise ValueError(
                    "Invalid SHUTDOWN_TIMEOUT_SECONDS: must be an integer"
                )

        # Credential Sanitizer
        if "credential_sanitizer" not in security:
            security["credential_sanitizer"] = {}

        redaction_placeholder = os.getenv("CREDENTIAL_REDACTION_PLACEHOLDER")
        if redaction_placeholder:
            security["credential_sanitizer"]["redaction_placeholder"] = (
                redaction_placeholder
            )

        return config_data

    @property
    def config(self) -> Config:
        """Get the loaded configuration."""
        if not self._config:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._config
