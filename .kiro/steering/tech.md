# Tech Stack

## Language & Runtime

- Python 3.9+
- Async/await patterns throughout

## Framework & Libraries

### Core
- FastAPI - Web framework with async support
- Pydantic v2 - Configuration and request/response validation
- SQLAlchemy 2.x - Async ORM with PostgreSQL
- asyncpg - Async PostgreSQL driver
- aiohttp - Async HTTP client for API calls
- Alembic - Database migrations

### Authentication & Security
- python-jose - JWT token handling
- Authlib - OAuth/Auth0 integration
- eth-utils - Ethereum address validation

### Orchestration & ML
- ZenML - Pipeline orchestration and scheduling
- XGBoost / Scikit-Learn - ML classification models
- Pandas & NumPy - Data processing

### Visualization
- Marimo - Interactive notebook framework
- Altair - Declarative statistical visualizations

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
- Pipeline scheduling via `config/scheduling.json`

## Security Components

- `CredentialSanitizer` - Scrubs secrets from logs
- `SSRFProtector` - Validates external API requests
- `ResourceLimiter` - Prevents resource exhaustion
- `GracefulTerminator` - Handles shutdown signals safely
- `CircuitBreaker` - API resilience with exponential backoff
- `SecureLogger` - Logger wrapper with automatic sanitization

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt
# or with Poetry
poetry install

# Run database migrations
alembic upgrade head

# Start development server
uvicorn main:app --reload

# Run CLI agent (standalone)
python cli.py --config config.json

# Run ZenML master pipeline
python -c "from pipelines.master_pipeline import run_master_pipeline; run_master_pipeline()"

# Open visualization notebook
marimo edit notebooks/stablecoin_analysis.py

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

- Poetry (pyproject.toml) - Primary
- pip (requirements.txt) - Alternative
- Both are supported and kept in sync
