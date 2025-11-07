# Design Document

## Overview

The blockchain stablecoin explorer agent is a Python-based autonomous system that collects and analyzes USDC and USDT usage data from three blockchain explorer platforms. The agent uses a combination of API calls (where available) and web scraping to gather transaction data, classify usage patterns, and output structured JSON for downstream analysis.

The system includes a web-based interface with Auth0 authentication to provide multi-user access control. Users authenticate through Auth0 to trigger agent runs, view collection history, and download results. This enables secure collaboration between team members while maintaining audit trails of who initiated each data collection run.

The system prioritizes API-based data collection for reliability and falls back to web scraping when necessary. It implements retry logic, rate limiting, and error handling to ensure robust operation across different blockchain explorers.

## Architecture

### High-Level Architecture

```
                    ┌──────────────┐
                    │   Auth0      │
                    │   Identity   │
                    └──────┬───────┘
                           │
                           ▼
┌──────────────────────────────────────────────────┐
│              Web API Layer (FastAPI)             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│  │   Auth     │  │   Agent    │  │  Results   │ │
│  │ Middleware │  │  Endpoints │  │ Endpoints  │ │
│  └────────────┘  └────────────┘  └────────────┘ │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Configuration  │
                │     Manager     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   Agent Core    │
                │   Orchestrator  │
                └────────┬────────┘
                         │
         ├───────────────┼───────────────┬──────────────┐
         ▼               ▼               ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Explorer 1  │ │  Explorer 2  │ │  Explorer 3  │ │   Activity   │
│   Collector  │ │   Collector  │ │   Collector  │ │  Classifier  │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │                │
       └────────────────┴────────────────┴────────────────┘
                                │
                                ▼
                        ┌──────────────┐
                        │     Data     │
                        │  Aggregator  │
                        └──────┬───────┘
                               │
                               ▼
                        ┌──────────────┐
                        │    JSON      │
                        │   Exporter   │
                        └──────┬───────┘
                               │
                               ▼
                        ┌──────────────┐
                        │   Database   │
                        │  (PostgreSQL)│
                        └──────────────┘
```

### Component Interaction Flow

1. User authenticates via Auth0 through web interface
2. Web API validates JWT token and authorizes user
3. User triggers agent run through API endpoint
4. Configuration Manager loads settings (explorer URLs, API keys, stablecoin addresses)
5. Agent Core Orchestrator initializes collector instances for each explorer
6. Collectors fetch data in parallel using asyncio
7. Activity Classifier analyzes collected data to determine usage patterns
8. Data Aggregator combines results from all sources
9. JSON Exporter writes structured output to file and database
10. User can view results and download JSON through web interface

## Components and Interfaces

### 1. Auth0 Integration Layer

**Responsibility:** Handle user authentication and authorization

**Interface:**
```python
class Auth0Manager:
    def __init__(self, domain: str, client_id: str, client_secret: str)
    def verify_token(self, token: str) -> UserInfo
    def get_user_permissions(self, user_id: str) -> List[str]
    def check_permission(self, user_id: str, permission: str) -> bool
```

**Auth0 Configuration:**
- Application Type: Regular Web Application
- Allowed Callback URLs: `http://localhost:8000/callback`, `https://yourdomain.com/callback`
- Allowed Logout URLs: `http://localhost:8000`, `https://yourdomain.com`
- Permissions:
  - `run:agent` - Trigger data collection runs
  - `view:results` - View collection results
  - `download:data` - Download JSON outputs
  - `admin:config` - Modify agent configuration (optional, for admin users)

### 2. Web API Layer (FastAPI)

**Responsibility:** Provide HTTP endpoints for user interaction with the agent

**Interface:**
```python
class AgentAPI:
    # Authentication endpoints
    @app.get("/login")
    async def login() -> RedirectResponse
    
    @app.get("/callback")
    async def callback(code: str) -> dict
    
    @app.get("/logout")
    async def logout() -> RedirectResponse
    
    # Agent control endpoints
    @app.post("/api/agent/run")
    @requires_auth(permission="run:agent")
    async def trigger_agent_run(config: Optional[RunConfig]) -> RunResponse
    
    @app.get("/api/agent/status/{run_id}")
    @requires_auth(permission="view:results")
    async def get_run_status(run_id: str) -> StatusResponse
    
    # Results endpoints
    @app.get("/api/results")
    @requires_auth(permission="view:results")
    async def list_results(limit: int = 50) -> List[ResultSummary]
    
    @app.get("/api/results/{run_id}")
    @requires_auth(permission="view:results")
    async def get_result_details(run_id: str) -> ResultDetails
    
    @app.get("/api/results/{run_id}/download")
    @requires_auth(permission="download:data")
    async def download_result(run_id: str) -> FileResponse
```

**Authentication Middleware:**
```python
async def requires_auth(permission: Optional[str] = None):
    """Decorator to protect endpoints with Auth0 authentication"""
    token = extract_token_from_header(request)
    user_info = auth0_manager.verify_token(token)
    
    if permission:
        if not auth0_manager.check_permission(user_info.user_id, permission):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    return user_info
```

### 3. Database Layer

**Responsibility:** Store run history, results metadata, and user activity logs

**Schema:**
```sql
CREATE TABLE users (
    user_id VARCHAR(255) PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE agent_runs (
    run_id UUID PRIMARY KEY,
    user_id VARCHAR(255) REFERENCES users(user_id),
    status VARCHAR(50) NOT NULL, -- 'running', 'completed', 'failed'
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    config JSONB,
    error_message TEXT
);

CREATE TABLE run_results (
    result_id UUID PRIMARY KEY,
    run_id UUID REFERENCES agent_runs(run_id),
    total_records INTEGER,
    explorers_queried TEXT[],
    output_file_path TEXT,
    summary JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE audit_logs (
    log_id UUID PRIMARY KEY,
    user_id VARCHAR(255) REFERENCES users(user_id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(255),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT
);
```

**Interface:**
```python
class DatabaseManager:
    async def create_run(self, user_id: str, config: dict) -> str
    async def update_run_status(self, run_id: str, status: str) -> None
    async def save_run_result(self, run_id: str, result: AggregatedData) -> None
    async def get_user_runs(self, user_id: str, limit: int) -> List[AgentRun]
    async def get_run_details(self, run_id: str) -> RunDetails
    async def log_user_action(self, user_id: str, action: str, resource: dict) -> None
```

### 4. Configuration Manager

**Responsibility:** Load and validate configuration from file

**Interface:**
```python
class ConfigurationManager:
    def load_config(self, config_path: str) -> Config
    def validate_config(self, config: Config) -> bool
    def get_explorer_configs(self) -> List[ExplorerConfig]
    def get_stablecoin_addresses(self) -> Dict[str, str]
```

**Configuration Schema:**
```json
{
  "explorers": [
    {
      "name": "etherscan",
      "base_url": "https://api.etherscan.io/api",
      "api_key": "YOUR_API_KEY",
      "type": "api"
    },
    {
      "name": "bscscan",
      "base_url": "https://api.bscscan.com/api",
      "api_key": "YOUR_API_KEY",
      "type": "api"
    },
    {
      "name": "polygonscan",
      "base_url": "https://api.polygonscan.com/api",
      "api_key": "YOUR_API_KEY",
      "type": "api"
    }
  ],
  "stablecoins": {
    "USDC": {
      "ethereum": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
      "bsc": "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d",
      "polygon": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    },
    "USDT": {
      "ethereum": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
      "bsc": "0x55d398326f99059fF775485246999027B3197955",
      "polygon": "0xc2132D05D31c914a87C6611C10748AEb04B58e8F"
    }
  },
  "output": {
    "directory": "./output",
    "max_records_per_explorer": 1000
  },
  "retry": {
    "max_attempts": 3,
    "backoff_seconds": 60
  }
}
```

### 5. Agent Core Orchestrator

**Responsibility:** Coordinate data collection across all explorers

**Interface:**
```python
class AgentOrchestrator:
    def __init__(self, config: Config, run_id: str, db_manager: DatabaseManager)
    async def run(self) -> CollectionResult
    async def collect_from_all_explorers(self) -> List[ExplorerData]
    def generate_report(self, results: List[ExplorerData]) -> Report
    async def update_progress(self, progress: float, message: str) -> None
```

### 6. Explorer Collector (Base Class)

**Responsibility:** Abstract interface for data collection from blockchain explorers

**Interface:**
```python
class ExplorerCollector(ABC):
    def __init__(self, config: ExplorerConfig)
    
    @abstractmethod
    async def fetch_stablecoin_transactions(
        self, 
        stablecoin: str, 
        contract_address: str,
        limit: int
    ) -> List[Transaction]
    
    @abstractmethod
    async def fetch_token_holders(
        self, 
        contract_address: str,
        limit: int
    ) -> List[Holder]
    
    def handle_rate_limit(self) -> None
    def validate_response(self, response: dict) -> bool
```

**Concrete Implementations:**
- `EtherscanCollector`: Uses Etherscan API for Ethereum data
- `BscscanCollector`: Uses BscScan API for BSC data
- `PolygonscanCollector`: Uses Polygonscan API for Polygon data

### 7. Activity Classifier

**Responsibility:** Analyze transaction patterns to classify usage types

**Interface:**
```python
class ActivityClassifier:
    def classify_transaction(self, tx: Transaction) -> ActivityType
    def identify_store_of_value(self, holder: Holder, transactions: List[Transaction]) -> bool
    def calculate_holding_period(self, address: str, transactions: List[Transaction]) -> int
```

**Classification Logic:**
- **Transaction**: Transfer with both sender and receiver, amount > 0
- **Store of Value**: Address holds tokens for > 30 days without outgoing transfers
- **Other**: Minting, burning, contract interactions, or unclassifiable activities

### 8. Data Aggregator

**Responsibility:** Combine and deduplicate data from multiple sources

**Interface:**
```python
class DataAggregator:
    def aggregate(self, explorer_results: List[ExplorerData]) -> AggregatedData
    def deduplicate_transactions(self, transactions: List[Transaction]) -> List[Transaction]
    def merge_holder_data(self, holders: List[Holder]) -> List[Holder]
```

### 9. JSON Exporter

**Responsibility:** Format and write data to JSON file and database

**Interface:**
```python
class JSONExporter:
    def __init__(self, db_manager: DatabaseManager)
    async def export(self, data: AggregatedData, run_id: str, output_path: str) -> None
    def generate_filename(self, run_id: str) -> str
    def validate_json_schema(self, data: dict) -> bool
    async def save_to_database(self, run_id: str, data: AggregatedData) -> None
```

## Data Models

### Transaction
```python
@dataclass
class Transaction:
    transaction_hash: str
    block_number: int
    timestamp: datetime
    from_address: str
    to_address: str
    amount: Decimal
    stablecoin: str  # "USDC" or "USDT"
    chain: str  # "ethereum", "bsc", "polygon"
    activity_type: ActivityType
    source_explorer: str
    gas_used: Optional[int] = None
    gas_price: Optional[Decimal] = None
```

### Holder
```python
@dataclass
class Holder:
    address: str
    balance: Decimal
    stablecoin: str
    chain: str
    first_seen: datetime
    last_activity: datetime
    is_store_of_value: bool
    source_explorer: str
```

### ActivityType
```python
class ActivityType(Enum):
    TRANSACTION = "transaction"
    STORE_OF_VALUE = "store_of_value"
    OTHER = "other"
    UNKNOWN = "unknown"
```

### Output JSON Schema
```json
{
  "metadata": {
    "run_id": "550e8400-e29b-41d4-a716-446655440000",
    "user_id": "auth0|123456789",
    "collection_timestamp": "2025-10-30T10:30:00Z",
    "agent_version": "1.0.0",
    "explorers_queried": ["etherscan", "bscscan", "polygonscan"],
    "total_records": 2847
  },
  "summary": {
    "by_stablecoin": {
      "USDC": {
        "total_transactions": 1523,
        "total_volume": "45678901.23",
        "unique_addresses": 892
      },
      "USDT": {
        "total_transactions": 1324,
        "total_volume": "38901234.56",
        "unique_addresses": 756
      }
    },
    "by_activity_type": {
      "transaction": 2401,
      "store_of_value": 389,
      "other": 57
    },
    "by_chain": {
      "ethereum": 1203,
      "bsc": 891,
      "polygon": 753
    }
  },
  "transactions": [
    {
      "transaction_hash": "0x...",
      "block_number": 18500000,
      "timestamp": "2025-10-30T09:15:23Z",
      "from_address": "0x...",
      "to_address": "0x...",
      "amount": "1000.50",
      "stablecoin": "USDC",
      "chain": "ethereum",
      "activity_type": "transaction",
      "source_explorer": "etherscan"
    }
  ],
  "holders": [
    {
      "address": "0x...",
      "balance": "50000.00",
      "stablecoin": "USDT",
      "chain": "bsc",
      "first_seen": "2025-08-15T12:00:00Z",
      "last_activity": "2025-08-20T14:30:00Z",
      "is_store_of_value": true,
      "source_explorer": "bscscan"
    }
  ]
}
```

## Error Handling

### Error Categories and Responses

1. **Network Errors**
   - Retry up to 3 times with exponential backoff
   - Log error details and continue with next explorer
   - Include partial results in final output

2. **Rate Limiting**
   - Detect 429 status codes or rate limit messages
   - Wait 60 seconds before retry
   - Implement request throttling (max 5 requests/second per explorer)

3. **Invalid Data**
   - Validate all required fields before adding to dataset
   - Log validation failures with details
   - Skip invalid records and continue processing

4. **API Key Issues**
   - Detect authentication errors (401, 403)
   - Log clear error message indicating API key problem
   - Skip that explorer and continue with others

5. **Configuration Errors**
   - Validate configuration on startup
   - Fail fast with clear error message if config is invalid
   - Provide examples of correct configuration format

### Logging Strategy

- Use Python `logging` module with structured logging
- Log levels:
  - INFO: Collection progress, records collected
  - WARNING: Retries, partial failures, data quality issues
  - ERROR: Explorer failures, API errors
  - DEBUG: Detailed request/response data
- Include correlation IDs for tracking requests across components

## Testing Strategy

### Unit Tests

- Test each component in isolation with mocked dependencies
- Focus on:
  - Auth0 token validation and permission checking
  - Configuration validation logic
  - Activity classification algorithms
  - Data aggregation and deduplication
  - JSON schema validation
  - Error handling paths

### Integration Tests

- Test API endpoints with mocked Auth0 authentication
- Test collector implementations against mock API responses
- Verify data flow from collectors through aggregator to exporter
- Test retry and rate limiting behavior
- Validate output JSON structure and content
- Test database operations with test database

### End-to-End Tests

- Test complete authentication flow with Auth0 test tenant
- Use test configuration with limited data collection
- Verify complete workflow from user login to result download
- Test with simulated API failures and rate limiting
- Validate final output against expected schema
- Test multi-user scenarios and permission enforcement

### Security Tests

- Test unauthorized access attempts to protected endpoints
- Verify JWT token validation and expiration handling
- Test SQL injection prevention
- Verify CORS and security headers
- Test rate limiting enforcement

### Test Data

- Create fixtures with sample API responses from each explorer
- Include edge cases: empty results, malformed data, rate limit responses
- Maintain separate test configuration file
- Use Auth0 test tenant for authentication testing

## Performance Considerations

- Use `asyncio` for concurrent API requests across explorers
- Implement connection pooling with `aiohttp`
- Set reasonable timeouts (30 seconds per request)
- Limit concurrent requests per explorer to avoid rate limiting
- Stream large result sets to avoid memory issues
- Consider pagination for explorers returning large datasets

## Security Considerations

### Authentication & Authorization
- Use Auth0 for centralized identity management
- Implement JWT token validation on all protected endpoints
- Use role-based access control (RBAC) for different permission levels
- Implement token refresh mechanism for long-running sessions
- Store Auth0 credentials in environment variables

### API Security
- Store blockchain explorer API keys in environment variables or secure vault
- Never log API keys, tokens, or sensitive addresses
- Implement rate limiting on API endpoints (e.g., 100 requests/minute per user)
- Use HTTPS for all external communications
- Validate and sanitize all user inputs

### Data Security
- Encrypt sensitive data at rest in PostgreSQL
- Use parameterized queries to prevent SQL injection
- Implement audit logging for all user actions
- Restrict database access to application service account only
- Regular backup of database with encrypted storage

### Session Management
- Use secure, httpOnly cookies for session tokens
- Implement CSRF protection on state-changing endpoints
- Set appropriate session timeout (e.g., 24 hours)
- Invalidate sessions on logout

### Infrastructure Security
- Use environment-specific configurations (dev, staging, prod)
- Implement secrets management (e.g., AWS Secrets Manager, HashiCorp Vault)
- Enable CORS only for trusted domains
- Use security headers (HSTS, X-Frame-Options, CSP)
