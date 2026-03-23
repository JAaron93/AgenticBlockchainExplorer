# AgenticBlockchainExplorer Helper Functions

This document outlines the shared helper functions and base classes created during the DRY refactoring efforts to improve maintainability and avoid code duplication.

## 1. Collector Layer Refactor (`collectors/base.py`)

To eliminate duplicate logic across `EtherscanCollector`, `BscscanCollector`, and `PolygonscanCollector`, core methods for data extraction, parsing, and aggregation were moved to the `ExplorerCollector` base class.

### Available Base Class Methods

- **`fetch_stablecoin_transactions(stablecoin, contract_address, limit, run_id)`**
  Coordinates fetching `tokenTx` API actions for ERC-20 compliant tokens. Handles pagination, mapping, and error checking natively.

- **`fetch_token_holders(stablecoin, contract_address, limit, run_id)`**
  Coordinates fetching the `tokenholderlist` endpoint. Handles missing token handlers (graceful degradation) natively.

- **`_parse_transaction(tx_data, stablecoin)`**
  Parses raw blockchain transaction data into the Pydantic `Transaction` model. Performs `BlockchainDataValidator` checks on the data when available.

- **`_parse_holder(holder_data, stablecoin, contract_address)`**
  Similar to `_parse_transaction`, validates raw token holder lists and parses valid entries into `Holder` records.

- **`_parse_amount(value_str, stablecoin)`**
  Uses the sub-class's native `TOKEN_DECIMALS` mapping to safely convert `value` strings to parsed `Decimal` units.

- **`_parse_timestamp(timestamp_str)`**
  Resolves UTC timezone-aware datetimes natively.

- **`_classify_activity(from_address, to_address, amount)`**
  Evaluates source/destination hashes (`0x000..000`) to infer Minting/Burning vs. Standard transfer behavior.

### Usage in Subclasses

When creating a new blockchain collector (e.g. `ArbitrumCollector`), inherit from `ExplorerCollector` and configure the `TOKEN_DECIMALS` class mapping:

```python
from collectors.base import ExplorerCollector

class ArbitrumCollector(ExplorerCollector):
    TOKEN_DECIMALS = {"USDC": 6, "USDT": 6}
    
    def __init__(self, config, retry_config=None):
        super().__init__(config, retry_config)
        if config.chain != "arbitrum": ...
```
No additional parsing overrides are needed unless the new chain relies on a fundamentally different schema for token transactions.

---

## 2. API Layer Authorization (`api/helpers.py`)

Instead of duplicating database and user ID assertions across every API endpoint, shared dependency helpers wrap native DatabaseManager interactions and authorization rules.

### Available Helper Functions

- **`get_authorized_run(run_id, user, db_manager, action="view") -> AgentRun`**
  Fetches a database `AgentRun` by its `run_id`. Handles `InvalidUUIDError` and asserts that `run.user_id == user.user_id` or `user.has_permission('admin:config')`. Returns the verified record or raises a fastAPI `HTTPException`.

- **`get_authorized_run_details(run_id, user, db_manager, action="view") -> Dict[str, Any]`**
  Similar to `get_authorized_run`, but queries the nested dictionary payload inside `db_manager.get_run_details()`.

### Usage in Routes

```python
from api.helpers import get_authorized_run

@router.get("/{run_id}/status")
async def get_run_status(
    run_id: str,
    user: UserInfo = Depends(require_auth),
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    # This 1-liner replaces 15 lines of repetitive checks mapping 400, 403, and 404 errors:
    run = await get_authorized_run(run_id, user, db_manager)
    return StatusResponse(run_id=run.run_id, status=run.status)
```

---

## 3. Collector Registry (`collectors/registry.py`)

A class-level registry that maps blockchain chain names to their `ExplorerCollector` subclasses, enabling a modular plugin-like architecture. Three collectors are registered by default: `EtherscanCollector` (`"ethereum"`), `BscscanCollector` (`"bsc"`), and `PolygonscanCollector` (`"polygon"`).

### Available Class Methods

- **`CollectorRegistry.register(chain, collector_class)`**
  Registers a new collector for the given chain name (case-insensitive). Logs a warning if an existing entry is overwritten.

- **`CollectorRegistry.get_collector_class(chain) → Optional[Type[ExplorerCollector]]`**
  Returns the collector class for the given chain, or `None` if not registered.

- **`CollectorRegistry.list_supported_chains() → List[str]`**
  Returns a list of all currently registered chain names.

### Adding a New Collector

Inherit from `ExplorerCollector`, then register the new class with the registry before running any collection:

```python
from collectors.base import ExplorerCollector
from collectors.registry import CollectorRegistry

class ArbitrumCollector(ExplorerCollector):
    TOKEN_DECIMALS = {"USDC": 6, "USDT": 6}

    def __init__(self, config, retry_config=None):
        super().__init__(config, retry_config)

# Register once at startup (e.g., in main.py or a config module)
CollectorRegistry.register("arbitrum", ArbitrumCollector)
```

No changes to `CollectionService` or `AgentOrchestrator` are needed — the service resolves the collector class from the registry at runtime via `get_collector_class`.

---

## 4. Collection Service (`core/services/collection.py`)

Encapsulates parallel data collection across all enabled blockchain explorers. Previously this logic lived inside `AgentOrchestrator`; it was extracted to `CollectionService` for modularity and testability.

### Constructor

```python
from core.services.collection import CollectionService
from config.models import Config

service = CollectionService(config)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `Config` | Full application configuration (explorers, stablecoins, retry settings) |

### Available Methods

- **`await service.collect_parallel(stablecoins, explorers=None, max_records=None, run_id=None, timeout_manager=None) → List[ExplorerData]`**
  Fans out collection to all enabled explorers concurrently using `asyncio.gather`. Failed collectors return partial `ExplorerData` objects with an `errors` list rather than raising, so one collector failure doesn't abort the others.

### How It Fits Together

```
AgentOrchestrator
    └── CollectionService.collect_parallel(...)
            └── CollectorRegistry.get_collector_class(chain)
                    └── ExplorerCollector.collect_all(...)
```

`AgentOrchestrator` instantiates `CollectionService` and calls `collect_parallel`. The service resolves each collector class from `CollectorRegistry`, instantiates it with the relevant `ExplorerConfig`, and runs all collectors concurrently. Results are always returned — errors are captured in the `ExplorerData.errors` field rather than propagated as exceptions.

