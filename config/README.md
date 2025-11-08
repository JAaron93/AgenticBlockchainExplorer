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
   ```json
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
     "secret_key": "<GENERATE_STRONG_SECRET_KEY>"
   }
   ```
   - **Risk**: Weak or default secret keys compromise session security
   - **Action**: Generate a cryptographically secure random key (minimum 32 characters)
   - **How to generate**:
     ```bash
     python -c "import secrets; print(secrets.token_urlsafe(32))"
     ```
   - **Never** commit the production secret key to version control

3. **Cookie Security**
   ```json
   "session": {
     "cookie_secure": true,      // MUST be true (requires HTTPS)
     "cookie_httponly": true,    // Keep as true
     "cookie_samesite": "strict" // Use "strict" for production
   }
   ```
   - **Risk**: Insecure cookies can be intercepted or used in CSRF attacks
   - **Action**: Set `cookie_secure: true` (requires HTTPS), use `cookie_samesite: "strict"`

4. **CORS Origins**
   ```json
   "cors": {
     "allowed_origins": [
       "https://your-production-domain.com"  // Only your actual domain(s)
     ]
   }
   ```
   - **Risk**: Permissive CORS allows unauthorized domains to access your API
   - **Action**: Replace localhost URLs with your actual production domain(s)
   - **Never** use `"*"` or include development URLs in production

5. **Auth0 Callback URLs**
   ```json
   "auth0": {
     "callback_url": "https://your-production-domain.com/callback",
     "logout_url": "https://your-production-domain.com"
   }
   ```
   - **Risk**: Incorrect callback URLs can break authentication or create security vulnerabilities
   - **Action**: Use HTTPS URLs matching your production domain

6. **Database Credentials**
   ```json
   "database": {
     "url": "postgresql://<USER>:<PASSWORD>@<DB_HOST>:5432/stablecoin_explorer"
   }
   ```
   - **Risk**: Weak or default credentials compromise database security
   - **Action**: Use strong, unique credentials; consider using environment variables
   - **Best Practice**: Store `DATABASE_URL` in environment variables, not in config files

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
DATABASE_URL=postgresql://prod_user:strong_password@prod-db:5432/stablecoin_explorer
SECRET_KEY=your_generated_secret_key_here
AUTH0_CLIENT_SECRET=your_auth0_client_secret
ETHERSCAN_API_KEY=your_etherscan_api_key
BSCSCAN_API_KEY=your_bscscan_api_key
POLYGONSCAN_API_KEY=your_polygonscan_api_key
```

Environment variables automatically override config file values.

#### Security Checklist Before Deployment

- [ ] `app.env` is set to `"production"`
- [ ] `app.debug` is set to `false`
- [ ] `app.secret_key` is a strong, randomly generated value (not the example)
- [ ] `session.cookie_secure` is set to `true`
- [ ] `session.cookie_samesite` is set to `"strict"`
- [ ] `cors.allowed_origins` contains only your production domain(s)
- [ ] `auth0.callback_url` and `auth0.logout_url` use HTTPS and your production domain
- [ ] `database.url` uses strong credentials (preferably from environment variables)
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

| Environment Variable | Configuration Path |
|---------------------|-------------------|
| `AUTH0_DOMAIN` | `auth0.domain` |
| `AUTH0_CLIENT_ID` | `auth0.client_id` |
| `AUTH0_CLIENT_SECRET` | `auth0.client_secret` |
| `DATABASE_URL` | `database.url` |
| `APP_ENV` | `app.env` |
| `APP_PORT` | `app.port` |
| `SECRET_KEY` | `app.secret_key` |
| `ETHERSCAN_API_KEY` | `explorers[name=etherscan].api_key` |
| `BSCSCAN_API_KEY` | `explorers[name=bscscan].api_key` |
| `POLYGONSCAN_API_KEY` | `explorers[name=polygonscan].api_key` |
| `OUTPUT_DIRECTORY` | `output.directory` |
| `MAX_RECORDS_PER_EXPLORER` | `output.max_records_per_explorer` |

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
