# AgenticBlockchainExplorer Rules

## Project Context
This project is an autonomous agent that explores blockchain explorers to collect and analyze USDC and USDT stablecoin usage data.
It is built with Python, FastAPI, and SQLAlchemy.

## Tech Stack & Libraries
- **Framework**: FastAPI (Async)
- **Database**: PostgreSQL with SQLAlchemy 2.0 (ORM) & Alembic for migrations
- **Authentication**: Auth0 (OAuth2 / OIDC)
- **Dependency Management**: Poetry
- **Testing**: pytest (asyncio)

## Coding Standards
- **Formatting**: Follow `black` style.
- **Linting**: Ensure code passes `flake8` and `mypy` type checking.
- **Type Hints**: All functions and methods must have type hints.
- **Async/Await**: Use asynchronous database sessions and HTTP requests (aiohttp/httpx) where possible.
- **Pydantic**: Use Pydantic V2 models for schemas and validation.

## Development Workflow
- **Running the App**: `uvicorn main:app --reload`
- **Database Migrations**:
    - Create: `alembic revision --autogenerate -m "message"`
    - Apply: `alembic upgrade head`
- **Testing**: Run `pytest` for the test suite.

## Security & Configuration
- **Secrets**: NEVER hardcode API keys or secrets. Use environment variables and `pydantic-settings`.
- **API Keys**: Required keys include `ETHERSCAN_API_KEY`, `BSCSCAN_API_KEY`, `POLYGONSCAN_API_KEY`, and Auth0 credentials.
- **Environment**: Use `.env` for local development configurations (see `.env.example`).
