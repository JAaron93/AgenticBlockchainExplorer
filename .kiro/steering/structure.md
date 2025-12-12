# Project Structure

```
.
├── main.py              # FastAPI app entry point, lifespan management
├── config.json          # Runtime configuration (gitignored)
├── config.example.json  # Configuration template
├── .env                 # Environment variables (gitignored)
├── .env.example         # Environment template
│
├── api/                 # API layer
│   ├── routes.py        # FastAPI routers (auth, agent, results)
│   └── auth_middleware.py  # JWT validation, permission checking
│
├── config/              # Configuration management
│   ├── models.py        # Pydantic config schemas
│   └── loader.py        # ConfigurationManager - loads JSON + env vars
│
├── core/                # Core business logic
│   ├── auth0_manager.py # Auth0 client, token verification
│   ├── database.py      # Async database connection management
│   └── db_manager.py    # CRUD operations for all entities
│
├── collectors/          # Blockchain data collectors
│   ├── base.py          # ExplorerCollector ABC with retry/rate limiting
│   ├── models.py        # Transaction, Holder, ExplorerData dataclasses
│   ├── etherscan.py     # Etherscan API collector
│   ├── bscscan.py       # BscScan API collector
│   ├── polygonscan.py   # Polygonscan API collector
│   ├── classifier.py    # ActivityClassifier for transaction types
│   ├── aggregator.py    # DataAggregator for deduplication/merging
│   └── exporter.py      # JSON export functionality
│
├── models/              # Database models
│   └── database.py      # SQLAlchemy models (User, AgentRun, RunResult, AuditLog)
│
├── alembic/             # Database migrations
│   ├── env.py           # Migration environment config
│   └── versions/        # Migration scripts
│
├── tests/               # Test suite
│   └── test_*.py        # Pytest test files
│
└── output/              # Generated JSON output (created at runtime)
```

## Key Patterns

- **Async everywhere**: All database and HTTP operations are async
- **Dependency injection**: FastAPI Depends() for auth, database, config
- **Background tasks**: Long-running collection jobs run via FastAPI BackgroundTasks
- **Layered architecture**: API → Core → Collectors → Models
- **Configuration precedence**: Environment variables override JSON config values
