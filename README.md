# Blockchain Stablecoin Explorer

An autonomous agent that explores blockchain explorers to collect and analyze USDC and USDT stablecoin usage data.

## Features

- Multi-chain support (Ethereum, BSC, Polygon)
- Auth0 authentication and authorization
- RESTful API for triggering data collection and retrieving results
- Activity classification (transactions, store of value, other)
- Structured JSON output for analysis
- Robust error handling and retry logic

## Project Structure

```
.
├── api/              # FastAPI endpoints and middleware
├── core/             # Core business logic (orchestrator, classifier, aggregator)
├── collectors/       # Blockchain explorer data collectors
├── models/           # Data models and database schemas
├── config/           # Configuration management
├── tests/            # Test suite
├── output/           # Generated JSON output files (created at runtime)
└── main.py           # Application entry point
```

## Setup

### Prerequisites

- Python 3.9+
- PostgreSQL 12+
- Auth0 account
- API keys for Etherscan, BscScan, and Polygonscan

### Installation

1. Clone the repository
2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Copy `.env.example` to `.env` and fill in your configuration
4. Install dependencies:

Using Poetry:
```bash
poetry install
```

Using pip:
```bash
pip install -r requirements.txt
```

5. Run database migrations:
```bash
alembic upgrade head
```

6. Start the application:
```bash
uvicorn main:app --reload
```

## Configuration

See `.env.example` for all available configuration options.

## API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation.

## Development

Run tests:
```bash
pytest
```

Format code:
```bash
black .
```

Lint code:
```bash
flake8 .
```

Type check:
```bash
mypy .
```

## License

MIT
