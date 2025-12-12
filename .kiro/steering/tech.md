# Tech Stack

## Language & Runtime

- Python 3.9+
- Async/await patterns throughout

## Framework & Libraries

- FastAPI - Web framework with async support
- Pydantic v2 - Configuration and request/response validation
- SQLAlchemy 2.x - Async ORM with PostgreSQL
- asyncpg - Async PostgreSQL driver
- aiohttp - Async HTTP client for API calls
- Alembic - Database migrations
- python-jose - JWT token handling
- Authlib - OAuth/Auth0 integration
- eth-utils - Ethereum address validation

## Database

- PostgreSQL 12+
- Connection pooling via SQLAlchemy async engine

## Authentication

- Auth0 for OAuth 2.0 / JWT authentication
- Permission-based authorization (run:agent, view:results, download:data, admin:config)

## Configuration

- JSON config file (`config.json`) with Pydantic validation
- Environment variables override JSON values (via `.env` file)
- Sensitive values (API keys, secrets) must come from environment variables

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Start development server
uvicorn main:app --reload

# Run tests
pytest

# Type checking
mypy .

# Code formatting
black .

# Linting
flake8 .
```

## Package Management

- Poetry (pyproject.toml) or pip (requirements.txt)
- Both are supported and kept in sync
