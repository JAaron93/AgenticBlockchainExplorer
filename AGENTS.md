# AgenticBlockchainExplorer Rules

## Project Context
This project is an autonomous Stablecoin Data Analysis System that explores blockchain explorers to collect, analyze, and predict usage patterns of USDC and USDT stablecoins.

Unlike a simple API, this system is an orchestrated data platform that combines:
1.  **Data Collection**: Automated gathering from Etherscan, BscScan, and Polygonscan.
2.  **Machine Learning**: SoV (Store of Value) prediction and wallet classification.
3.  **Visualization**: Interactive dashboards using Marimo and Altair.

## Tech Stack & Libraries
- **Orchestration**: ZenML (Pipelines, Steps, Scheduling)
- **Framework**: FastAPI (Async) for API layer
- **Database**: PostgreSQL with SQLAlchemy 2.0 (ORM) & Alembic
- **Machine Learning**: 
    - XGBoost / Scikit-Learn (Classification)
    - Pandas & NumPy (Data Processing)
- **Visualization**: Marimo (Interactive Notebooks), Altair (Charts)
- **Authentication**: Auth0 (OAuth2 / OIDC)
- **Dependency Management**: Poetry
- **Testing**: pytest (asyncio)

## Architecture & Pipelines
The system is built around ZenML pipelines that orchestrate the data flow:

### Master Pipeline (`master_pipeline.py`)
The primary workflow running on a weekly schedule:
1.  **Collection Phase**: Parallel data fetching from multiple explorers.
2.  **Aggregation Phase**: Deduplication and merging of transaction data.
3.  **Analysis Phase**: 
    - Activity Analysis (Transaction types)
    - Holder Analysis (Behavior metrics)
    - Time Series (Volume/Count over time)
    - Chain Comparison (Cross-chain metrics)
4.  **ML Inference Phase**:
    - Feature Engineering
    - SoV Prediction
    - Wallet Classification

### Other Pipelines
- **Collection Pipeline**: Specialized for raw data gathering.
- **Analysis Pipeline**: Focused on re-running metrics on existing data.

## ML & Analysis
The system performs advanced analytics beyond simple aggregation:
- **Feature Engineering**: Extracts transaction frequency, volatility, recency, and holding periods.
- **SoV Prediction**: Uses XGBoost/RandomForest to predict if a wallet is a "Store of Value" user.
- **Wallet Classification**: Classifies wallets based on behavioral patterns.

## Development Workflow

### Running the System
- **CLI Agent (Standalone)**:
  ```bash
  python cli.py --config config.json
  ```
- **ZenML Pipeline**:
  ```bash
  python -c "from pipelines.master_pipeline import run_master_pipeline; run_master_pipeline()"
  ```
- **API Server**:
  ```bash
  uvicorn main:app --reload
  ```
- **Visualization**:
  ```bash
  marimo edit notebooks/stablecoin_analysis.py
  ```

### Database Migrations
- Create: `alembic revision --autogenerate -m "message"`
- Apply: `alembic upgrade head`

### Testing
- Run suite: `pytest`

## Security & Configuration
- **Core Security Modules**:
    - `credential_sanitizer`: Scrubs secrets from logs.
    - `ssrf_protector`: Validates external API requests.
    - `resource_limiter`: Prevents resource exhaustion.
    - `graceful_terminator`: Handles shutdown signals safely.
- **Configuration**:
    - `.env`: Secrets (API Keys, DB Credentials). **NEVER commit this.**
    - `config.json`: General agent configuration.
    - `config/scheduling.json`: Pipeline schedules.
- **API Keys**: Required for `ETHERSCAN`, `BSCSCAN`, `POLYGONSCAN`, and Auth0.

## Coding Standards
- **Formatting**: Follow `black` style.
- **Linting**: Ensure code passes `flake8` and `mypy` type checking.
- **Type Hints**: All functions and methods must have type hints.
- **Async/Await**: Use asynchronous database sessions and HTTP requests where possible.
- **Pydantic**: Use Pydantic V2 models for schemas and validation.
