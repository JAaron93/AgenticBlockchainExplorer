# Configuration Management

This module provides configuration management for the blockchain stablecoin explorer agent.

## Features

- **Pydantic-based validation**: All configuration is validated using Pydantic models
- **Multiple sources**: Configuration can be loaded from JSON files and environment variables
- **Environment variable override**: Environment variables take precedence over JSON configuration
- **Type safety**: Full type checking and validation for all configuration fields

## Configuration Files

### config.json

The main configuration file. Copy `config.example.json` to `config.json` and update with your values:

```bash
cp config.example.json config.json
```

### .env

Environment variables file. Copy `.env.example` to `.env` and update with your values:

```bash
cp .env.example .env
```

## ⚠️ Production Deployment

**CRITICAL**: The default `config.example.json` contains development settings that are **INSECURE for production use**. Before deploying to production, you MUST review and update the following security-sensitive settings.

### Production Configuration Checklist

Use `config.example.production.json` as a starting point for production deployments:

```bash
cp config.example.production.json config.json
```

#### Required Changes for Production

1. **App Environment & Debug Mode**
   ```jsonc
   "app": {
     "env": "production",        // MUST be "production"
     "debug": false              // MUST be false in production
   }
   ```
   - **Risk**: Debug mode exposes sensitive error details and stack traces
   - **Action**: Set `debug: false` and `env: "production"`

2. **Secret Key**
   ```json
   "app": {
     "secret_key": "${SECRET_KEY}"
   }
   ```
   - **Risk**: Weak or default secret keys compromise session security
   - **Action**: Always use the `SECRET_KEY` environment variable in production
   - **Required**: The application will fail to start if `SECRET_KEY` is not set when using the placeholder
   - **Requirements**: Must be cryptographically generated with at least 32 bytes (43+ characters when base64-encoded)
   - **How to generate**:
     ```bash
     # Generate a secure 32-byte (256-bit) secret key
     python -c "import secrets; print(secrets.token_urlsafe(32))"
     
     # Alternative using OpenSSL
     openssl rand -base64 32
     ```
   - **Never** commit the production secret key to version control
   - **Secure provisioning**: Use environment variables, AWS Secrets Manager, HashiCorp Vault, or similar

3. **Cookie Security**
   ```jsonc
   "session": {
     "cookie_secure": true,      // MUST be true (requires HTTPS)
     "cookie_httponly": true,    // Keep as true
     "cookie_samesite": "strict" // Use "strict" for production
   }
   ```
   - **Risk**: Insecure cookies can be intercepted or used in CSRF attacks
   - **Action**: Set `cookie_secure: true` (requires HTTPS), use `cookie_samesite: "strict"`

4. **CORS Origins**
   ```jsonc
   "cors": {
     "allowed_origins": [
       "https://your-production-domain.com"  // Only your actual domain(s)
     ]
   }
   ```
   - **Risk**: Permissive CORS allows unauthorized domains to access your API
   - **Action**: Replace localhost URLs with your actual production domain(s)
   - **Never** use `"*"` or include development URLs in production

5. **Auth0 Configuration**
   ```json
   "auth0": {
     "domain": "${AUTH0_DOMAIN}",
     "client_id": "${AUTH0_CLIENT_ID}",
     "client_secret": "${AUTH0_CLIENT_SECRET}",
     "audience": "${AUTH0_AUDIENCE}",
     "callback_url": "${AUTH0_CALLBACK_URL}",
     "logout_url": "${AUTH0_LOGOUT_URL}"
   }
   ```
   - **Risk**: Storing Auth0 credentials in config files can lead to accidental exposure
   - **Action**: Always use environment variables for all Auth0 settings
   - **Required**: The application will fail to start if any Auth0 environment variable is missing
   - **Never** commit Auth0 credentials (especially `client_secret`) to version control

6. **Database Credentials**
   ```json
   "database": {
     "url": "${DATABASE_URL}"
   }
   ```
   - **Risk**: Storing credentials in config files can lead to accidental exposure
   - **Action**: Always use the `DATABASE_URL` environment variable
   - **Required**: The application will fail to start if `DATABASE_URL` is not set
   - **Format**: `postgresql://username:password@hostname:port/database_name`
   - **Example**: `postgresql://app_user:secure_password@db.example.com:5432/stablecoin_explorer`
   - **Never** commit database credentials to version control

7. **Logging Level**
   ```json
   "logging": {
     "level": "WARNING"  // Use "WARNING" or "ERROR" for production
   }
   ```
   - **Risk**: Verbose logging can expose sensitive information and impact performance
   - **Action**: Set to "WARNING" or "ERROR" in production

### Production Deployment Best Practices

#### Use Environment Variables for Secrets

Instead of storing secrets in `config.json`, use environment variables:

```bash
# .env (DO NOT commit to version control)

# Auth0 Configuration (ALL REQUIRED)
AUTH0_DOMAIN=your-tenant.auth0.com
AUTH0_CLIENT_ID=your_auth0_client_id
AUTH0_CLIENT_SECRET=your_auth0_client_secret
AUTH0_AUDIENCE=https://your-api-identifier
AUTH0_CALLBACK_URL=https://your-production-domain.com/callback
AUTH0_LOGOUT_URL=https://your-production-domain.com

# Database (REQUIRED)
DATABASE_URL=postgresql://prod_user:strong_password@prod-db:5432/stablecoin_explorer

# Application Secret (REQUIRED)
SECRET_KEY=your_generated_secret_key_here

# API Keys
ETHERSCAN_API_KEY=your_etherscan_api_key
BSCSCAN_API_KEY=your_bscscan_api_key
POLYGONSCAN_API_KEY=your_polygonscan_api_key
```

Environment variables automatically override config file values.

#### Required Auth0 Environment Variables

All Auth0 configuration **must** be provided via environment variables. The application will fail to start if any are missing.

| Variable | Description | Example |
|----------|-------------|---------|
| `AUTH0_DOMAIN` | Your Auth0 tenant domain | `your-tenant.auth0.com` |
| `AUTH0_CLIENT_ID` | Application client ID from Auth0 | `abc123def456...` |
| `AUTH0_CLIENT_SECRET` | Application client secret from Auth0 | `xyz789...` |
| `AUTH0_AUDIENCE` | API identifier/audience | `https://api.example.com` |
| `AUTH0_CALLBACK_URL` | OAuth callback URL (must match Auth0 config) | `https://app.example.com/callback` |
| `AUTH0_LOGOUT_URL` | Post-logout redirect URL | `https://app.example.com` |

**Important:**
- Get these values from your Auth0 Dashboard → Applications → Your App
- `AUTH0_CLIENT_SECRET` is highly sensitive - never commit it to version control
- Callback and logout URLs must use HTTPS in production
- URLs must be registered in your Auth0 application settings

#### DATABASE_URL Format

The `DATABASE_URL` environment variable is **required** and must follow this format:

```
postgresql://username:password@hostname:port/database_name
```

**Components:**
- `username`: PostgreSQL user with appropriate permissions
- `password`: User's password (URL-encode special characters)
- `hostname`: Database server hostname or IP address
- `port`: PostgreSQL port (default: 5432)
- `database_name`: Name of the database

**Examples:**
```bash
# Local development
DATABASE_URL=postgresql://dev_user:dev_pass@localhost:5432/stablecoin_dev

# Production with standard port
DATABASE_URL=postgresql://app_user:secure_password@db.example.com:5432/stablecoin_explorer

# With special characters in password (URL-encoded)
DATABASE_URL=postgresql://user:p%40ssw%23rd@db.example.com:5432/mydb

# AWS RDS example
DATABASE_URL=postgresql://admin:password@mydb.abc123.us-east-1.rds.amazonaws.com:5432/stablecoin
```

**Special Character Encoding:**
If your password contains special characters, URL-encode them:
- `@` → `%40`
- `#` → `%23`
- `/` → `%2F`
- `:` → `%3A`
- `%` → `%25`

#### Security Checklist Before Deployment

- [ ] `app.env` is set to `"production"`
- [ ] `app.debug` is set to `false`
- [ ] `SECRET_KEY` environment variable is set with a cryptographically secure value (32+ bytes)
- [ ] `session.cookie_secure` is set to `true`
- [ ] `session.cookie_samesite` is set to `"strict"`
- [ ] `cors.allowed_origins` contains only your production domain(s)
- [ ] `AUTH0_DOMAIN` environment variable is set
- [ ] `AUTH0_CLIENT_ID` environment variable is set
- [ ] `AUTH0_CLIENT_SECRET` environment variable is set (never in config files!)
- [ ] `AUTH0_AUDIENCE` environment variable is set
- [ ] `AUTH0_CALLBACK_URL` environment variable uses HTTPS and your production domain
- [ ] `AUTH0_LOGOUT_URL` environment variable uses HTTPS and your production domain
- [ ] `DATABASE_URL` environment variable is set with strong credentials
- [ ] `logging.level` is set to `"WARNING"` or `"ERROR"`
- [ ] All API keys are valid and not placeholder values
- [ ] `.env` file is added to `.gitignore`
- [ ] Production secrets are stored securely (e.g., AWS Secrets Manager, HashiCorp Vault)

#### Configuration Validation

Use the provided validation script to check your production configuration:

```bash
python config/validate_production.py ./config.json
```

This script checks for common security issues including:
- Debug mode enabled
- Weak or placeholder secret keys
- Insecure cookie settings
- Localhost URLs in CORS or Auth0 settings
- Placeholder API keys or database credentials
- Inappropriate logging levels

You can also validate programmatically:

```python
from config import ConfigurationManager

# Load and validate production config
manager = ConfigurationManager(config_path="./config.json")
config = manager.load_config()

# Verify production settings
assert config.app.env == "production", "Environment must be 'production'"
assert config.app.debug is False, "Debug mode must be disabled"
assert config.session.cookie_secure is True, "Secure cookies must be enabled"
assert "localhost" not in str(config.cors.allowed_origins), "Remove localhost from CORS"

print("✓ Production configuration validated")
```

## Usage

### Basic Usage

```python
from config import ConfigurationManager

# Load configuration
manager = ConfigurationManager(config_path="./config.json")
config = manager.load_config()

# Access configuration
print(config.app.env)  # development
print(config.database.url)  # postgresql://...

# Get explorer configurations
# Returns: List[ExplorerConfig]
explorers = manager.get_explorer_configs()
for explorer in explorers:  # Each explorer is an ExplorerConfig object
    print(f"{explorer.name}: {explorer.chain}")

# Get specific explorer
# Returns: Optional[ExplorerConfig] (None if not found)
etherscan = manager.get_explorer_by_name("etherscan")
if etherscan:
    print(f"Etherscan API: {etherscan.base_url}")

# Get stablecoin addresses
# Returns: Dict[str, StablecoinConfig]
# Structure: {"TICKER": StablecoinConfig(ethereum="0x...", bsc="0x...", polygon="0x...")}
stablecoins = manager.get_stablecoin_addresses()
usdc = stablecoins["USDC"]  # usdc is a StablecoinConfig object
print(f"USDC on Ethereum: {usdc.ethereum}")  # Access via attribute
print(f"USDC on BSC: {usdc.bsc}")
print(f"USDC on Polygon: {usdc.polygon}")

# Example structure returned:
# {
#     "USDC": StablecoinConfig(
#         ethereum="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
#         bsc="0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d",
#         polygon="0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
#     ),
#     "USDT": StablecoinConfig(...)
# }
```

### Environment Variable Override

Environment variables will override values from the JSON file:

```bash
# Set environment variables
export AUTH0_DOMAIN="production.auth0.com"
export DATABASE_URL="postgresql://prod:pass@prod-db:5432/prod"
export APP_ENV="production"

# Load configuration (environment variables take precedence)
python your_script.py
```

## Configuration Schema

### Main Configuration

- `explorers`: List of blockchain explorer configurations
- `stablecoins`: Stablecoin contract addresses for each chain
- `auth0`: Auth0 authentication configuration
- `database`: Database connection configuration
- `app`: Application settings
- `output`: Output directory and limits
- `retry`: Retry and timeout settings
- `rate_limit`: API rate limiting configuration
- `logging`: Logging configuration
- `cors`: CORS settings
- `session`: Session management settings

### Explorer Configuration

Each explorer requires:
- `name`: Explorer name (e.g., "etherscan")
- `base_url`: API base URL
- `api_key`: API key for authentication
- `type`: Explorer type ("api" or "scraper")
- `chain`: Blockchain network ("ethereum", "bsc", or "polygon")

### Stablecoin Configuration

For each stablecoin (USDC, USDT), provide contract addresses for:
- `ethereum`: Ethereum mainnet address
- `bsc`: Binance Smart Chain address
- `polygon`: Polygon network address

**Return Type**: `get_stablecoin_addresses()` returns `Dict[str, StablecoinConfig]` where each `StablecoinConfig` is a Pydantic model with attributes (not dict keys):

```python
# Access via object attributes (not dictionary keys)
stablecoins = manager.get_stablecoin_addresses()
usdc = stablecoins["USDC"]  # StablecoinConfig object
address = usdc.ethereum  # Use dot notation, not usdc["ethereum"]
```

## Validation

The configuration manager validates:

- Required fields are present
- Field types are correct
- Ethereum addresses are valid (42 characters, starts with 0x)
- URLs are valid
- Numeric values are within acceptable ranges
- Explorer names and chains are unique
- Required stablecoins (USDC, USDT) are configured

## Error Handling

The configuration manager will raise exceptions for:

- Missing configuration file: `FileNotFoundError`
- Invalid JSON: `json.JSONDecodeError`
- Validation errors: `pydantic.ValidationError`

Example:

```python
from pydantic import ValidationError

try:
    manager = ConfigurationManager()
    config = manager.load_config()
except FileNotFoundError as e:
    print(f"Configuration file not found: {e}")
except ValidationError as e:
    print(f"Configuration validation failed: {e}")
```

## Environment Variable Mapping

| Environment Variable | Configuration Path | Required |
|---------------------|-------------------|----------|
| `AUTH0_DOMAIN` | `auth0.domain` | **Yes** |
| `AUTH0_CLIENT_ID` | `auth0.client_id` | **Yes** |
| `AUTH0_CLIENT_SECRET` | `auth0.client_secret` | **Yes** |
| `AUTH0_AUDIENCE` | `auth0.audience` | **Yes** |
| `AUTH0_CALLBACK_URL` | `auth0.callback_url` | **Yes** |
| `AUTH0_LOGOUT_URL` | `auth0.logout_url` | **Yes** |
| `DATABASE_URL` | `database.url` | **Yes** |
| `SECRET_KEY` | `app.secret_key` | **Yes** (production) |
| `APP_ENV` | `app.env` | No |
| `APP_PORT` | `app.port` | No |
| `ETHERSCAN_API_KEY` | `explorers[name=etherscan].api_key` | No |
| `BSCSCAN_API_KEY` | `explorers[name=bscscan].api_key` | No |
| `POLYGONSCAN_API_KEY` | `explorers[name=polygonscan].api_key` | No |
| `OUTPUT_DIRECTORY` | `output.directory` | No |
| `MAX_RECORDS_PER_EXPLORER` | `output.max_records_per_explorer` | No |

See `.env.example` for a complete list of environment variables.

## Quick Start Guide

### Development Setup

```bash
# 1. Copy development config
cp config.example.json config.json

# 2. Update with your API keys and settings
# Edit config.json and replace placeholder values

# 3. (Optional) Use environment variables
cp .env.example .env
# Edit .env with your secrets
```

### Production Setup

```bash
# 1. Copy production config template
cp config/config.example.production.json config.json

# 2. Update all security-sensitive settings (see checklist above)
# Edit config.json

# 3. Validate configuration
python config/validate_production.py

# 4. Use environment variables for secrets (recommended)
# Set DATABASE_URL, SECRET_KEY, API keys via environment
```

**Remember**: Never commit `config.json` or `.env` files containing real credentials to version control!
