# Setup Guide

This guide walks you through setting up the Blockchain Stablecoin Explorer from scratch.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Auth0 Setup](#auth0-setup)
3. [Database Setup](#database-setup)
4. [Blockchain Explorer API Keys](#blockchain-explorer-api-keys)
5. [Environment Configuration](#environment-configuration)
6. [Application Setup](#application-setup)
7. [Running the Application](#running-the-application)
8. [Verification](#verification)

---

## Prerequisites

Before starting, ensure you have the following installed:

- **Python 3.9+** - [Download Python](https://www.python.org/downloads/)
- **PostgreSQL 12+** - [Download PostgreSQL](https://www.postgresql.org/download/)
- **Git** - [Download Git](https://git-scm.com/downloads)

Verify installations:

```bash
python --version   # Should be 3.9 or higher
psql --version     # Should be 12 or higher
git --version
```

---

## Auth0 Setup

Auth0 provides authentication and authorization for the application. Follow these steps to configure it.

### Step 1: Create an Auth0 Account

1. Go to [Auth0](https://auth0.com/) and sign up for a free account
2. After signing up, you'll be prompted to create a tenant (e.g., `your-company.auth0.com`)

### Step 2: Create an Application

1. In the Auth0 Dashboard, go to **Applications** → **Applications**
2. Click **+ Create Application**
3. Enter a name (e.g., "Blockchain Stablecoin Explorer")
4. Select **Regular Web Applications**
5. Click **Create**

### Step 3: Configure Application Settings

In your application settings, configure the following:

**Basic Information:**
- Note down the **Domain**, **Client ID**, and **Client Secret**

**Application URIs:**

For development:
```
Allowed Callback URLs: http://localhost:8000/callback
Allowed Logout URLs: http://localhost:8000
Allowed Web Origins: http://localhost:8000
```

For production (replace with your domain):
```
Allowed Callback URLs: https://your-domain.com/callback
Allowed Logout URLs: https://your-domain.com
Allowed Web Origins: https://your-domain.com
```

Click **Save Changes**.

### Step 4: Create an API

1. Go to **Applications** → **APIs**
2. Click **+ Create API**
3. Enter:
   - **Name**: Stablecoin Explorer API
   - **Identifier**: `https://stablecoin-explorer-api` (this becomes your `AUTH0_AUDIENCE`)
   - **Signing Algorithm**: RS256
4. Click **Create**

### Step 5: Define Permissions

In your API settings, go to the **Permissions** tab and add:

| Permission | Description |
|------------|-------------|
| `run:agent` | Trigger data collection runs |
| `view:results` | View collection results |
| `download:data` | Download JSON outputs |
| `admin:config` | Modify agent configuration (admin only) |

### Step 6: Assign Permissions to Users

1. Go to **User Management** → **Roles**
2. Create roles (e.g., "User", "Admin") and assign permissions
3. Assign roles to users in **User Management** → **Users**

Alternatively, use Auth0 Rules or Actions to assign default permissions to new users.

### Auth0 Configuration Summary

You'll need these values for your `.env` file:

```bash
AUTH0_DOMAIN=your-tenant.auth0.com
AUTH0_CLIENT_ID=your_client_id_here
AUTH0_CLIENT_SECRET=your_client_secret_here
AUTH0_AUDIENCE=https://stablecoin-explorer-api
AUTH0_CALLBACK_URL=http://localhost:8000/callback
AUTH0_LOGOUT_URL=http://localhost:8000
```

---

## Database Setup

### Step 1: Install PostgreSQL

**macOS (using Homebrew):**
```bash
brew install postgresql@12
brew services start postgresql@12
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

**Windows:**
Download and run the installer from [PostgreSQL Downloads](https://www.postgresql.org/download/windows/)

### Step 2: Create Database and User

Connect to PostgreSQL:
```bash
# macOS/Linux
sudo -u postgres psql

# Windows (use pgAdmin or psql from Start Menu)
psql -U postgres
```

Create the database and user:

> **Security Note:** Use a strong, randomly generated password (not the placeholder below). Never commit database credentials to source control. Store credentials securely using a password manager or secret management tool (e.g., AWS Secrets Manager, HashiCorp Vault).

```sql
-- Create a dedicated user (replace 'your_secure_password' with a strong password)
CREATE USER stablecoin_user WITH PASSWORD 'your_secure_password';

-- Create the database
CREATE DATABASE stablecoin_explorer OWNER stablecoin_user;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE stablecoin_explorer TO stablecoin_user;

-- Exit
\q
```

**Note:** The password `'your_secure_password'` above is illustrative only. Replace it with a secure, randomly generated password for production use.

### Step 3: Verify Connection

Test the connection:
```bash
psql -h localhost -U stablecoin_user -d stablecoin_explorer
```

Your `DATABASE_URL` will be:
```
postgresql://stablecoin_user:your_secure_password@localhost:5432/stablecoin_explorer
```

### Step 4: Run Database Migrations

After setting up the application (see below), run migrations:
```bash
alembic upgrade head
```

This creates the required tables:
- `users` - User accounts
- `agent_runs` - Data collection run history
- `run_results` - Results metadata
- `audit_logs` - User action audit trail

---

## Blockchain Explorer API Keys

The application collects data from three blockchain explorers. You'll need API keys for each.

### Etherscan (Ethereum)

1. Go to [Etherscan](https://etherscan.io/)
2. Create an account or sign in
3. Go to **API Keys** in your account settings
4. Click **Add** to create a new API key
5. Copy the API key

**Free tier limits:** 5 calls/second, 100,000 calls/day

### BscScan (Binance Smart Chain)

1. Go to [BscScan](https://bscscan.com/)
2. Create an account or sign in
3. Go to **API Keys** in your account settings
4. Click **Add** to create a new API key
5. Copy the API key

**Free tier limits:** 5 calls/second, 100,000 calls/day

### Polygonscan (Polygon)

1. Go to [Polygonscan](https://polygonscan.com/)
2. Create an account or sign in
3. Go to **API Keys** in your account settings
4. Click **Add** to create a new API key
5. Copy the API key

**Free tier limits:** 5 calls/second, 100,000 calls/day

### API Keys Summary

Add these to your `.env` file:
```bash
ETHERSCAN_API_KEY=your_etherscan_api_key
BSCSCAN_API_KEY=your_bscscan_api_key
POLYGONSCAN_API_KEY=your_polygonscan_api_key
```

---

## Environment Configuration

### Understanding Configuration Files

The application uses two types of configuration files:

- **`.env.example`** - A template for sensitive, environment-specific values (secrets, API keys, database credentials). Copy this to `.env` and fill in your values. The `.env.example` file stays in the repository as a reference; only `.env` is used at runtime.

- **`config.example.json`** - Contains non-sensitive, default application settings (timeouts, limits, feature flags). Copy this to `config.json` for local overrides. The `config.example.json` file stays in the repository as a template.

**Required files for the application to run:** `.env` and `config.json`

**Configuration precedence:** Environment variables (from `.env`) take precedence over values in `config.json`. This allows you to override any setting via environment variables without modifying the JSON file.

### Step 1: Create Environment File

Copy the example environment file:
```bash
cp .env.example .env
```

> **Security Warning:**
> - Never commit `.env` to version control (ensure `.env` is in `.gitignore`)
> - Keep `.env.example` with placeholder values only—no real secrets
> - If secrets are accidentally exposed, rotate all affected API keys and credentials immediately

### Step 2: Configure Required Variables

Edit `.env` with your values:

```bash
# Database Configuration (REQUIRED)
DATABASE_URL=postgresql://stablecoin_user:your_secure_password@localhost:5432/stablecoin_explorer

# Auth0 Configuration (REQUIRED)
AUTH0_DOMAIN=your-tenant.auth0.com
AUTH0_CLIENT_ID=your_client_id
AUTH0_CLIENT_SECRET=your_client_secret
AUTH0_AUDIENCE=https://stablecoin-explorer-api
AUTH0_CALLBACK_URL=http://localhost:8000/callback
AUTH0_LOGOUT_URL=http://localhost:8000

# Blockchain Explorer API Keys (REQUIRED)
ETHERSCAN_API_KEY=your_etherscan_api_key
BSCSCAN_API_KEY=your_bscscan_api_key
POLYGONSCAN_API_KEY=your_polygonscan_api_key

# Application Secret (REQUIRED for production)
# Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
SECRET_KEY=your_generated_secret_key

# Application Settings
APP_ENV=development
APP_PORT=8000
APP_DEBUG=true
```

### Step 3: Create Configuration File

Copy the example configuration:
```bash
cp config.example.json config.json
```

The configuration file contains non-sensitive settings. API keys and secrets should be in `.env` (environment variables override config.json values).

### Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `AUTH0_DOMAIN` | Yes | Your Auth0 tenant domain |
| `AUTH0_CLIENT_ID` | Yes | Auth0 application client ID |
| `AUTH0_CLIENT_SECRET` | Yes | Auth0 application client secret |
| `AUTH0_AUDIENCE` | Yes | Auth0 API identifier |
| `AUTH0_CALLBACK_URL` | Yes | OAuth callback URL |
| `AUTH0_LOGOUT_URL` | Yes | Post-logout redirect URL |
| `ETHERSCAN_API_KEY` | Yes | Etherscan API key |
| `BSCSCAN_API_KEY` | Yes | BscScan API key |
| `POLYGONSCAN_API_KEY` | Yes | Polygonscan API key |
| `SECRET_KEY` | Production | Session encryption key |
| `APP_ENV` | No | Environment (development/production) |
| `APP_PORT` | No | Server port (default: 8000) |
| `APP_DEBUG` | No | Enable debug mode (default: false) |

---

## Application Setup

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd blockchain-stablecoin-explorer
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run Database Migrations

```bash
alembic upgrade head
```

### Step 5: Create Output Directory

```bash
mkdir -p output
```

---

## Running the Application

### Development Mode

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at `http://localhost:8000`

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Or use the provided entry point:
```bash
python main.py
```

### Using the CLI (Optional)

For standalone agent execution without the web interface:
The CLI mode allows you to run data collection tasks independently. It reads configuration from `config.json` and credentials from `.env`.


---

## Verification

### Check Health Endpoints

```bash
# Basic health check
curl http://localhost:8000/health

# Readiness check (verifies database and Auth0)
curl http://localhost:8000/health/ready

# Liveness check
curl http://localhost:8000/health/live
```

### Access API Documentation

Open your browser and navigate to:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Test Authentication Flow

Before testing, ensure you have a user account in Auth0. You can create one in the Auth0 Dashboard or sign up through the login flow.

1. Navigate to http://localhost:8000/login
2. You should be redirected to Auth0
3. Log in with your Auth0 credentials
4. You should be redirected back with an access token
### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_config.py -v
```

---

## Troubleshooting

### Database Connection Issues

**Error:** `connection refused`
- Ensure PostgreSQL is running: `sudo systemctl status postgresql`
- Check the connection string in `DATABASE_URL`

**Error:** `authentication failed`
- Verify username and password in `DATABASE_URL`
- Check PostgreSQL user permissions

### Auth0 Issues

**Error:** `Invalid callback URL`
- Ensure `AUTH0_CALLBACK_URL` matches exactly what's configured in Auth0 Dashboard
- Check for trailing slashes

**Error:** `Invalid audience`
- Verify `AUTH0_AUDIENCE` matches the API identifier in Auth0

### API Key Issues

**Error:** `Invalid API Key` from blockchain explorers
- Verify API keys are correct and active
- Check if you've exceeded rate limits
- Ensure API keys have the required permissions

### Migration Issues

**Error:** `relation does not exist`
- Run migrations: `alembic upgrade head`
- Check database connection

---

## Next Steps

- Read the [API Documentation](./API.md) to understand available endpoints
- Review [config/README.md](../config/README.md) for advanced configuration options
- Check the production deployment checklist in config/README.md before deploying
