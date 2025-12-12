# Implementation Plan

- [x] 1. Set up project structure and dependencies
  - Create Python project with poetry or pip requirements
  - Install core dependencies: FastAPI, SQLAlchemy, asyncio, aiohttp, python-jose, authlib
  - Set up directory structure: api/, core/, collectors/, models/, config/, tests/
  - Create .env.example file with required environment variables
  - _Requirements: 5.1, 5.2_

- [x] 2. Implement configuration management
  - [x] 2.1 Create configuration schema and validation
    - Define Pydantic models for Config, ExplorerConfig, StablecoinConfig, Auth0Config
    - Implement configuration loader that reads from JSON file and environment variables
    - Add validation for required fields and format checking
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  
  - [x] 2.2 Create configuration file templates
    - Write config.json template with all three explorers (Etherscan, BscScan, Polygonscan)
    - Include USDC and USDT contract addresses for each chain
    - Add Auth0 configuration section
    - _Requirements: 5.1, 5.2_

- [x] 3. Set up database layer
  - [x] 3.1 Create database models and schema
    - Define SQLAlchemy models for users, agent_runs, run_results, audit_logs tables
    - Write Alembic migration scripts for initial schema
    - Implement database connection management with connection pooling
    - _Requirements: 4.2, 4.4_
  
  - [x] 3.2 Implement database manager
    - Code DatabaseManager class with async methods for CRUD operations
    - Implement create_run, update_run_status, save_run_result methods
    - Add get_user_runs, get_run_details, log_user_action methods
    - _Requirements: 4.2, 4.4_

- [x] 4. Implement Auth0 integration
  - [x] 4.1 Create Auth0 manager
    - Implement Auth0Manager class with token verification using python-jose
    - Add methods for verify_token, get_user_permissions, check_permission
    - Handle token expiration and validation errors
    - _Requirements: 4.1, 4.3_
  
  - [x] 4.2 Create authentication middleware
    - Write FastAPI dependency for requires_auth decorator
    - Implement JWT token extraction from Authorization header
    - Add permission checking logic
    - Return appropriate HTTP errors (401, 403) for auth failures
    - _Requirements: 4.1, 4.3_

- [x] 5. Build FastAPI web API
  - [x] 5.1 Implement authentication endpoints
    - Create /login endpoint that redirects to Auth0
    - Implement /callback endpoint to handle Auth0 response and create session
    - Add /logout endpoint to clear session
    - _Requirements: 4.1_
  
  - [x] 5.2 Implement agent control endpoints
    - Create POST /api/agent/run endpoint to trigger data collection
    - Implement GET /api/agent/status/{run_id} to check run progress
    - Add background task execution for long-running agent operations
    - _Requirements: 1.4, 4.2_
  
  - [x] 5.3 Implement results endpoints
    - Create GET /api/results to list all runs for authenticated user
    - Implement GET /api/results/{run_id} to get detailed results
    - Add GET /api/results/{run_id}/download to serve JSON file
    - _Requirements: 3.5, 4.2_

- [x] 6. Implement blockchain explorer collectors
  - [x] 6.1 Create base collector class
    - Define ExplorerCollector abstract base class
    - Implement common methods: handle_rate_limit, validate_response
    - Add retry logic with exponential backoff
    - _Requirements: 1.4, 4.1, 4.5_
  
  - [x] 6.2 Implement Etherscan collector
    - Create EtherscanCollector class extending ExplorerCollector
    - Implement fetch_stablecoin_transactions using Etherscan API
    - Implement fetch_token_holders method
    - Parse API responses into Transaction and Holder models
    - _Requirements: 1.1, 1.2, 1.3, 2.2, 2.3, 2.4_
  
  - [x] 6.3 Implement BscScan collector
    - Create BscscanCollector class extending ExplorerCollector
    - Implement fetch_stablecoin_transactions using BscScan API
    - Implement fetch_token_holders method
    - Handle BSC-specific response formats
    - _Requirements: 1.1, 1.2, 1.3, 2.2, 2.3, 2.4_
  
  - [x] 6.4 Implement Polygonscan collector
    - Create PolygonscanCollector class extending ExplorerCollector
    - Implement fetch_stablecoin_transactions using Polygonscan API
    - Implement fetch_token_holders method
    - Handle Polygon-specific response formats
    - _Requirements: 1.1, 1.2, 1.3, 2.2, 2.3, 2.4_

- [x] 7. Implement activity classification
  - [x] 7.1 Create activity classifier
    - Implement ActivityClassifier class with classify_transaction method
    - Add logic to identify transaction type based on from/to addresses and amount
    - Implement identify_store_of_value method checking 30-day holding period
    - Add calculate_holding_period helper method
    - _Requirements: 2.1, 2.5_

- [x] 8. Implement data aggregation
  - [x] 8.1 Create data aggregator
    - Implement DataAggregator class with aggregate method
    - Add deduplicate_transactions logic using transaction hash
    - Implement merge_holder_data to combine data from multiple sources
    - Generate summary statistics by stablecoin, chain, and activity type
    - _Requirements: 3.4, 4.4_

- [x] 9. Implement JSON export
  - [x] 9.1 Create JSON exporter
    - Implement JSONExporter class with export method
    - Generate output filename with run_id and timestamp
    - Validate JSON schema before writing
    - Write JSON file to configured output directory
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  
  - [x] 9.2 Add database persistence
    - Implement save_to_database method in JSONExporter
    - Store run results metadata in run_results table
    - Save summary statistics to database
    - _Requirements: 3.5, 4.2_

- [x] 10. Implement agent orchestrator
  - [x] 10.1 Create agent core orchestrator
    - Implement AgentOrchestrator class with run method
    - Initialize all three collector instances from configuration
    - Execute collectors in parallel using asyncio.gather
    - Handle partial failures and continue with available data
    - _Requirements: 1.1, 1.4, 1.5, 4.1, 4.5_
  
  - [x] 10.2 Add progress tracking
    - Implement update_progress method to update database
    - Track collection progress per explorer
    - Update run status in database (running, completed, failed)
    - _Requirements: 4.2_
  
  - [x] 10.3 Wire orchestrator with other components
    - Connect orchestrator to collectors, classifier, aggregator, and exporter
    - Pass collected data through classification pipeline
    - Aggregate results and export to JSON and database
    - Generate final report with record counts per source
    - _Requirements: 1.1, 1.4, 4.2_

- [x] 11. Implement error handling and logging
  - [x] 11.1 Set up structured logging
    - Configure Python logging with JSON formatter
    - Add log levels: INFO for progress, WARNING for retries, ERROR for failures
    - Include correlation IDs (run_id) in all log messages
    - _Requirements: 4.1, 4.3_
  
  - [x] 11.2 Add error handling
    - Implement try-catch blocks in all collectors for network errors
    - Handle rate limiting with 60-second wait and retry
    - Validate data and skip invalid records with logging
    - Return partial results when some explorers fail
    - _Requirements: 1.5, 4.1, 4.3, 4.5_

- [x] 12. Create data models
  - [x] 12.1 Define core data models
    - Create Transaction dataclass with all required fields
    - Create Holder dataclass for token holder information
    - Define ActivityType enum
    - Add validation methods to ensure data integrity
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.2, 3.3, 3.4_

- [x] 13. Add API rate limiting and security
  - [x] 13.1 Implement rate limiting
    - Add rate limiting middleware to FastAPI (100 requests/minute per user)
    - Implement request throttling for blockchain explorer APIs
    - _Requirements: 4.5_
  
  - [x] 13.2 Add security headers and CORS
    - Configure CORS for trusted domains only
    - Add security headers (HSTS, X-Frame-Options, CSP)
    - Implement CSRF protection for state-changing endpoints
    - _Requirements: 4.1_

- [x] 14. Create startup script and entry point
  - [x] 14.1 Create main application entry point
    - Write main.py to initialize FastAPI app
    - Set up database connection on startup
    - Configure Auth0 integration
    - Add health check endpoint
    - _Requirements: 5.1, 5.2_
  
  - [x] 14.2 Add CLI for standalone agent execution
    - Create CLI script for running agent without web interface (optional for testing)
    - Accept configuration file path as argument
    - Output results to console and file
    - _Requirements: 5.1, 5.2_

- [ ]* 15. Write tests
  - [ ]* 15.1 Write unit tests for core components
    - Test configuration validation with valid and invalid configs
    - Test activity classification logic with sample transactions
    - Test data aggregation and deduplication
    - Test Auth0 token validation with mock tokens
    - _Requirements: All_
  
  - [ ]* 15.2 Write integration tests
    - Test API endpoints with mocked Auth0
    - Test collectors with mock API responses
    - Test database operations with test database
    - Test complete data flow from collection to export
    - _Requirements: All_
  
  - [ ]* 15.3 Write end-to-end tests
    - Test authentication flow with Auth0 test tenant
    - Test complete agent run from API trigger to result download
    - Test multi-user scenarios and permission enforcement
    - _Requirements: All_

- [ ] 16. Create documentation
  - [ ] 16.1 Write setup documentation
    - Document Auth0 setup steps (create tenant, configure application)
    - Document database setup (PostgreSQL installation, running migrations)
    - Document environment variable configuration
    - Document how to obtain blockchain explorer API keys
    - _Requirements: 5.1, 5.2_
  
  - [ ] 16.2 Write API documentation
    - Generate OpenAPI/Swagger documentation from FastAPI
    - Document authentication flow
    - Document all endpoints with request/response examples
    - _Requirements: 4.1_
