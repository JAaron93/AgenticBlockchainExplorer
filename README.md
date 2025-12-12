# Blockchain Stablecoin Explorer

An autonomous agent that explores blockchain explorers to collect and analyze USDC and USDT stablecoin usage data.

## Features

- Multi-chain support (Ethereum, BSC, Polygon)
- Auth0 authentication and authorization
- RESTful API for triggering data collection and retrieving results
- Activity classification (transactions, store of value, other)
- Structured JSON output for analysis
- Robust error handling and retry logic

## Documentation

- **[Setup Guide](docs/SETUP.md)** - Complete setup instructions including Auth0, database, and API keys
- **[API Documentation](docs/API.md)** - REST API reference with examples
- **[Configuration Guide](config/README.md)** - Detailed configuration options and production deployment

## Quick Start

```bash
# 1. Clone and setup
git clone <repository-url>
cd blockchain-stablecoin-explorer
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
cp config.example.json config.json
# Edit .env with your Auth0, database, and API key settings

# 4. Run migrations
alembic upgrade head

# 5. Start the server
uvicorn main:app --reload
```

For detailed setup instructions, see the [Setup Guide](docs/SETUP.md).

## Project Structure

```
.
├── api/              # FastAPI endpoints and middleware
├── core/             # Core business logic (orchestrator, auth, database)
├── collectors/       # Blockchain explorer data collectors
├── models/           # SQLAlchemy database models
├── config/           # Configuration management
├── docs/             # Documentation
├── tests/            # Test suite
├── output/           # Generated JSON output files (created at runtime)
└── main.py           # Application entry point
```

## API Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/login` | GET | Initiate Auth0 login |
| `/callback` | GET | Auth0 OAuth callback |
| `/logout` | GET | Log out user |
| `/api/agent/run` | POST | Start data collection |
| `/api/agent/status/{run_id}` | GET | Check run status |
| `/api/results` | GET | List all runs |
| `/api/results/{run_id}` | GET | Get run details |
| `/api/results/{run_id}/download` | GET | Download JSON output |

For complete API documentation, see [docs/API.md](docs/API.md) or visit `/docs` when running.

## Configuration

### Required Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/stablecoin_explorer

# Auth0
AUTH0_DOMAIN=your-tenant.auth0.com
AUTH0_CLIENT_ID=your_client_id
AUTH0_CLIENT_SECRET=your_client_secret
AUTH0_AUDIENCE=https://your-api-identifier
AUTH0_CALLBACK_URL=http://localhost:8000/callback
AUTH0_LOGOUT_URL=http://localhost:8000

# Blockchain Explorer API Keys
ETHERSCAN_API_KEY=your_key
BSCSCAN_API_KEY=your_key
POLYGONSCAN_API_KEY=your_key
```

See [.env.example](.env.example) for all configuration options.

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
