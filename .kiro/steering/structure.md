# Project Structure

```
.
├── main.py              # FastAPI app entry point, lifespan management
├── cli.py               # Standalone CLI agent runner
├── config.json          # Runtime configuration (gitignored)
├── config.example.json  # Configuration template
├── .env                 # Environment variables (gitignored)
├── .env.example         # Environment template
│
├── api/                 # API layer
│   ├── routes.py        # FastAPI routers (auth, agent, results)
│   ├── auth_middleware.py  # JWT validation, permission checking
│   ├── rate_limiter.py  # Request rate limiting
│   └── security.py      # Security headers, CORS, CSRF protection
│
├── config/              # Configuration management
│   ├── models.py        # Pydantic config schemas (including security configs)
│   ├── loader.py        # ConfigurationManager - loads JSON + env vars
│   ├── scheduling.json  # Pipeline scheduling configuration
│   └── ml_thresholds.json  # ML model thresholds
│
├── core/                # Core business logic
│   ├── auth0_manager.py # Auth0 client, token verification
│   ├── database.py      # Async database connection management
│   ├── db_manager.py    # CRUD operations for all entities
│   ├── orchestrator.py  # Agent orchestration logic
│   ├── exceptions.py    # Custom exception classes
│   ├── logging.py       # Logging configuration
│   └── security/        # Security hardening components
│       ├── credential_sanitizer.py  # Credential detection and redaction
│       ├── ssrf_protector.py        # SSRF attack prevention
│       ├── resource_limiter.py      # Resource consumption limits
│       ├── graceful_terminator.py   # Graceful shutdown handling
│       ├── circuit_breaker.py       # Circuit breaker pattern
│       └── secure_logger.py         # Sanitized logging wrapper
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
├── pipelines/           # ZenML pipeline definitions
│   ├── master_pipeline.py     # Main weekly pipeline (collection + analysis + ML)
│   ├── collection_pipeline.py # Data collection only
│   ├── analysis_pipeline.py   # Analysis only (re-run on existing data)
│   ├── model_monitor.py       # ML model monitoring
│   └── steps/                 # Pipeline step implementations
│       ├── collectors.py      # Data collection steps
│       ├── analysis.py        # Analysis steps (activity, holder, time series)
│       ├── ml.py              # ML inference steps (SoV prediction)
│       └── wallet_classifier.py  # Wallet classification steps
│
├── notebooks/           # Marimo visualization notebooks
│   ├── stablecoin_analysis.py       # Main analysis notebook
│   ├── stablecoin_analysis_functions.py  # Analysis helper functions
│   ├── stablecoin_loader.py         # Data loading utilities
│   ├── stablecoin_validation.py     # Schema validation
│   ├── sample_data_generator.py     # Sample data for testing
│   ├── zenml_bridge.py              # ZenML integration for notebooks
│   └── web_exporter.py              # Web export utilities
│
├── models/              # Database models
│   └── database.py      # SQLAlchemy models (User, AgentRun, RunResult, AuditLog)
│
├── alembic/             # Database migrations
│   ├── env.py           # Migration environment config
│   └── versions/        # Migration scripts
│
├── tests/               # Test suite
│   ├── test_aggregator.py
│   ├── test_api_integration.py
│   ├── test_auth0.py
│   ├── test_classifier.py
│   ├── test_collectors_integration.py
│   ├── test_config.py
│   ├── test_config_validation.py
│   ├── test_e2e.py
│   ├── test_notebook_properties.py
│   ├── test_rate_limiter.py
│   ├── test_security.py
│   ├── test_setup.py
│   └── test_zenml_pipeline_properties.py
│
├── docs/                # Documentation
│   ├── API.md           # API documentation
│   ├── DEPLOYMENT.md    # Deployment guide
│   ├── SCHEDULING.md    # Pipeline scheduling guide
│   └── SETUP.md         # Setup instructions
│
└── output/              # Generated JSON output (created at runtime)
```

## Key Patterns

- **Async everywhere**: All database and HTTP operations are async
- **Dependency injection**: FastAPI Depends() for auth, database, config
- **Background tasks**: Long-running collection jobs run via FastAPI BackgroundTasks
- **Pipeline orchestration**: ZenML pipelines for scheduled data collection and analysis
- **Layered architecture**: API → Core → Collectors → Models
- **Configuration precedence**: Environment variables override JSON config values
- **Security by default**: Credential sanitization, SSRF protection, resource limits

## Pipeline Architecture

```
Master Pipeline (Weekly Cron)
├── Collection Phase
│   ├── etherscan_collector_step
│   ├── bscscan_collector_step
│   └── polygonscan_collector_step
├── Aggregation Phase
│   └── aggregate_data_step
├── Analysis Phase
│   ├── activity_analysis_step
│   ├── holder_analysis_step
│   ├── time_series_step
│   └── chain_analysis_step
└── ML Inference Phase
    ├── feature_engineering_step
    ├── train_sov_predictor_step
    ├── predict_sov_step
    ├── train_wallet_classifier_step
    └── classify_wallets_step
```
