# Requirements Document

## Introduction

This document specifies the requirements for a marimo Python notebook integrated with ZenML that analyzes stablecoin transaction data collected by blockchain explorer agents. The system enables data scientists and analysts to determine whether users primarily use stablecoins (USDC, USDT) for everyday transactions or as a store of value, and provides predictive ML capabilities to forecast holder behavior.

The system is designed for deployment to a live website with scheduled weekly cron jobs. ZenML serves as the unified orchestration framework for:
- Blockchain data collection (wrapping existing collectors as ZenML steps)
- Data analysis pipelines
- Predictive ML models (SoV prediction and wallet behavior classification)
- Scheduled execution and experiment tracking

Marimo notebooks serve as both the interactive development environment and the visualization layer for ZenML pipeline outputs.

## Data Governance & Privacy

### Data Retention Policy
- **Raw transaction data**: Retain for 90 days in active storage, then archive to cold storage for 2 years
- **Aggregated analysis results**: Retain for 2 years in active storage
- **ML model artifacts**: Retain all production model versions indefinitely; non-production versions for 6 months
- **Audit logs**: Retain for 3 years (compliance requirement)
- **Deletion triggers**: Automated cleanup jobs run weekly; manual deletion requests processed within 30 days

### Sensitive Data Handling
- **Wallet addresses**: Store full addresses in database; mask to first 6 and last 4 characters (e.g., `0x1234...5678`) in logs, exports, and UI displays
- **Transaction hashes**: Store full hashes; mask in logs to first 10 characters
- **User identifiers**: Hash user_id values in logs using SHA-256; store plaintext only in authenticated session context
- **Encryption**: All data encrypted at rest (AES-256) and in transit (TLS 1.3)
- **API keys**: Never logged; stored only in environment variables or secrets manager

### Compliance Requirements
- **Scope**: System processes publicly available blockchain data; no direct PII collection
- **GDPR/CCPA**: Not directly applicable as blockchain addresses are pseudonymous; however, if user accounts are linked to addresses, data subject requests must be honored within 30 days
- **Data transfers**: Cross-border transfers permitted for blockchain data; user account data restricted to configured regions
- **Legal basis**: Legitimate interest for blockchain analytics; consent required for user account creation

### Audit Trail Requirements
- **Access logging**: All data access (read/write) logged with timestamp, user_id, resource, and action
- **RBAC enforcement**: Access controls enforced at API and database layers
- **Log retention**: Access logs retained for 3 years, immutable after creation
- **Responsibility**: Platform operator responsible for policy enforcement; automated compliance checks run daily

## Glossary

- **Marimo**: A reactive Python notebook framework that creates reproducible, interactive notebooks as pure Python scripts
- **ZenML**: An extensible, open-source MLOps framework for creating portable, production-ready machine learning pipelines with steps and artifacts
- **ZenML Step**: A decorated Python function that represents a single unit of work in a ZenML pipeline, with typed inputs/outputs
- **ZenML Pipeline**: A directed acyclic graph (DAG) of ZenML steps that defines an end-to-end ML workflow
- **ZenML Artifact**: A versioned output from a ZenML step that is automatically tracked and stored
- **Stablecoin**: A cryptocurrency designed to maintain a stable value relative to a reference asset (typically USD)
- **USDC**: USD Coin, a stablecoin issued by Circle
- **USDT**: Tether, a stablecoin issued by Tether Limited
- **Transaction Activity**: Regular transfers of stablecoins between addresses with non-zero amounts
- **Store of Value (SoV)**: Holding pattern where tokens remain in an address for extended periods (>30 days) without outgoing transfers
- **SoV Prediction**: Machine learning task to predict whether a holder will become a store-of-value user based on their transaction history
- **Wallet Behavior Classification**: Machine learning task to categorize wallets into behavioral patterns (e.g., trader, holder, whale, retail)
- **Activity Type**: Classification of stablecoin usage - "transaction", "store_of_value", or "other" (minting/burning)
- **Chain**: The blockchain network (Ethereum, BSC, Polygon)
- **Holding Period**: Number of days since last outgoing transfer from an address
- **JSON Export**: The structured output file from the data collection agent containing transactions, holders, and summary data
- **Gas Cost**: The transaction fee paid on a blockchain, computed as gas_used × gas_price; gas_price is stored in wei (10^-18 of native token)
- **Feature Engineering**: The process of extracting predictive features from raw transaction and holder data for ML models
- **Model Registry**: ZenML component for versioning and managing trained ML models

## Requirements

### Requirement 1

**User Story:** As a data analyst, I want to load and parse stablecoin data from JSON export files, so that I can analyze the collected blockchain data.

#### Acceptance Criteria

1. WHEN the notebook initializes THEN the Notebook SHALL provide a file selector UI element to choose JSON export files from the output directory
2. WHEN a user selects a JSON file THEN the Notebook SHALL parse and validate the JSON structure against the canonical schema defined in Appendix A: JSON Schema Specification
3. WHEN JSON parsing succeeds THEN the Notebook SHALL convert transaction and holder data into pandas DataFrames for analysis
4. IF the JSON file is malformed or missing required fields THEN the Notebook SHALL display a clear error message indicating the validation failure per the validation rules in Appendix A
5. WHEN data is loaded THEN the Notebook SHALL display metadata including run_id, collection timestamp, explorers queried, and total record count

### Requirement 2

**User Story:** As a data analyst, I want to see a breakdown of stablecoin usage by activity type, so that I can understand the primary use cases.

#### Acceptance Criteria

1. WHEN data is loaded THEN the Notebook SHALL calculate and display the count and percentage of transactions by activity type (transaction, store_of_value, other)
2. WHEN displaying activity breakdown THEN the Notebook SHALL render an interactive pie chart or bar chart showing the distribution
3. WHEN data is loaded THEN the Notebook SHALL calculate the total transaction volume (sum of amounts) for each activity type
4. WHEN displaying volume breakdown THEN the Notebook SHALL format amounts with appropriate decimal places and currency notation

### Requirement 3

**User Story:** As a data analyst, I want to compare stablecoin usage patterns across different tokens, so that I can identify differences between USDC and USDT behavior.

#### Acceptance Criteria

1. WHEN data is loaded THEN the Notebook SHALL group transactions by stablecoin type and calculate activity type distribution for each
2. WHEN displaying stablecoin comparison THEN the Notebook SHALL render a grouped bar chart comparing activity types across stablecoins
3. WHEN data is loaded THEN the Notebook SHALL calculate average transaction size per stablecoin
4. WHEN data is loaded THEN the Notebook SHALL calculate the ratio of store-of-value holders to active transactors per stablecoin globally (aggregated across all chains); for example, if there are 30 USDC holders with is_store_of_value=true and 70 USDC holders with is_store_of_value=false across all chains, the USDC SoV ratio is 30:70 or 0.43

### Requirement 4

**User Story:** As a data analyst, I want to analyze holder behavior patterns, so that I can determine what percentage of holders use stablecoins as store of value.

#### Acceptance Criteria

1. WHEN data is loaded THEN the Notebook SHALL calculate the percentage of holders classified as store_of_value versus active transactors
2. WHEN displaying holder analysis THEN the Notebook SHALL render a histogram of holder balances segmented by store_of_value status
3. WHEN data is loaded THEN the Notebook SHALL calculate the average and median holding period for store_of_value holders
4. WHEN data is loaded THEN the Notebook SHALL identify the top 10 holders by balance globally (across all stablecoins and chains combined), sorted by descending balance, and display each holder's address, balance, stablecoin, chain, and store_of_value classification

### Requirement 5

**User Story:** As a data analyst, I want to analyze transaction patterns over time, so that I can identify trends in stablecoin usage.

#### Acceptance Criteria

1. WHEN data is loaded THEN the Notebook SHALL parse transaction timestamps and create time-series aggregations
2. WHEN displaying time analysis THEN the Notebook SHALL render a line chart showing transaction count over time grouped by activity type
3. WHEN displaying time analysis THEN the Notebook SHALL render a line chart showing transaction volume over time grouped by stablecoin
4. WHEN data spans multiple days THEN the Notebook SHALL provide daily, weekly, and monthly aggregation options via UI controls

### Requirement 6

**User Story:** As a data analyst, I want to compare stablecoin usage across different blockchain networks, so that I can understand chain-specific patterns.

#### Acceptance Criteria

1. WHEN data is loaded THEN the Notebook SHALL group transactions by chain (Ethereum, BSC, Polygon) and calculate metrics for each
2. WHEN displaying chain comparison THEN the Notebook SHALL render a stacked bar chart showing activity type distribution per chain
3. WHEN data is loaded THEN the Notebook SHALL calculate average transaction size and average gas cost per chain, where gas cost is computed as: gas_cost = gas_used × gas_price (in wei), converted to native token units (ETH/BNB/MATIC) by dividing by 10^18; transactions with null gas_used or gas_price fields SHALL be excluded from gas cost calculations and the count of excluded transactions SHALL be noted
4. WHEN data is loaded THEN the Notebook SHALL calculate the store_of_value ratio per chain

### Requirement 7

**User Story:** As a data analyst, I want to generate a summary conclusion about stablecoin usage patterns, so that I can answer whether stablecoins are used primarily for transactions or store of value.

#### Acceptance Criteria

1. WHEN analysis is complete THEN the Notebook SHALL calculate an overall "transaction vs store_of_value" ratio based on both transaction counts and holder classifications
2. WHEN displaying conclusions THEN the Notebook SHALL render a summary panel with key findings including dominant usage pattern, chain with highest transaction activity, and stablecoin with highest store_of_value ratio
3. WHEN displaying conclusions THEN the Notebook SHALL calculate and display a confidence indicator using the following formula and thresholds:
   - **Field Completeness**: Percentage of non-null required fields (transaction_hash, timestamp, amount, stablecoin, chain, activity_type) across all transaction records
   - **Chain Coverage**: chains_with_data / 3 (where 3 is the fixed count of supported chains: ethereum, bsc, polygon)
   - **Combined Completeness**: completeness_percent = 0.7 × field_completeness + 0.3 × chain_coverage (field completeness weighted 70%, chain coverage weighted 30%)
   - **Normalized Sample Size**: normalized_sample_size = min(sample_size / 1000, 1.0)
   - **Combined Confidence Formula**: confidence_score = 0.6 × normalized_sample_size + 0.4 × completeness_percent
   - **Final Confidence Mapping**: High (score ≥ 0.85), Medium (0.50 ≤ score < 0.85), Low (score < 0.50)
   - **Example - High Confidence**: 1500 transactions, 98% field completeness, 3 chains → chain_coverage=1.0, completeness=0.7×0.98+0.3×1.0=0.986, score=0.6×1.0+0.4×0.986=0.994 → "High"
   - **Example - Medium Confidence**: 500 transactions, 95% field completeness, 2 chains → chain_coverage=0.667, completeness=0.7×0.95+0.3×0.667=0.865, score=0.6×0.5+0.4×0.865=0.646 → "Medium"
   - **Example - Low Confidence**: 50 transactions, 80% field completeness, 1 chain → chain_coverage=0.333, completeness=0.7×0.80+0.3×0.333=0.66, score=0.6×0.05+0.4×0.66=0.294 → "Low"
4. WHEN data contains errors from collection THEN the Notebook SHALL display warnings about data quality issues that may affect conclusions

### Requirement 8

**User Story:** As a data analyst, I want the notebook to work with sample data when no real dataset is available, so that I can validate the analysis logic.

#### Acceptance Criteria

1. WHEN no JSON file is selected THEN the Notebook SHALL offer an option to generate synthetic sample data
2. WHEN generating sample data THEN the Notebook SHALL create realistic transaction and holder records following the same schema as real exports
3. WHEN using sample data THEN the Notebook SHALL clearly indicate that analysis is based on synthetic data, not real blockchain data
4. WHEN generating sample data THEN the Notebook SHALL allow configuration of sample size and distribution parameters via UI controls

### Requirement 9

**User Story:** As a platform operator, I want the blockchain data collectors wrapped as ZenML steps, so that data collection can be orchestrated alongside analysis and ML pipelines.

#### Acceptance Criteria

1. WHEN defining the data collection pipeline THEN the System SHALL wrap each blockchain explorer collector (Etherscan, BscScan, Polygonscan) as a ZenML step with typed inputs and outputs
2. WHEN a collector step executes THEN the System SHALL return a ZenML artifact containing the collected transaction and holder data
3. WHEN multiple collector steps complete THEN the System SHALL provide an aggregation step that merges and deduplicates data from all sources
4. IF a collector step fails THEN the System SHALL:
   - Log the error with collector name, error type, and timestamp
   - Continue execution with remaining collectors
   - Include collector-run metadata in the output artifact specifying: successful_sources (list), failed_sources (list), and completeness_ratio (successful_count / 3)
   - By default, require at least 2 of 3 collectors to succeed (configurable via `min_successful_collectors` parameter, range 1-3)
   - Fail the aggregation step if fewer than `min_successful_collectors` succeed
5. WHEN the collection pipeline completes THEN the System SHALL export results as a versioned ZenML artifact that can be consumed by downstream analysis pipelines
6. WHEN downstream analysis steps receive data with completeness_ratio < 1.0 THEN the System SHALL:
   - Log a warning indicating which sources are missing
   - Include a `data_completeness` field in analysis outputs with the missing sources list
   - Continue processing unless completeness_ratio falls below a configurable `min_completeness_threshold` (default: 0.67, i.e., 2/3 sources)

### Requirement 10

**User Story:** As a platform operator, I want a unified ZenML pipeline that orchestrates data collection, analysis, and ML inference, so that the entire workflow runs as a single scheduled job.

#### Acceptance Criteria

1. WHEN defining the master pipeline THEN the System SHALL chain data collection, analysis, and ML inference steps in a single ZenML pipeline DAG
2. WHEN the pipeline executes THEN the System SHALL track all artifacts (raw data, analysis results, predictions) with ZenML's artifact versioning
3. WHEN configuring the pipeline THEN the System SHALL support parameterization for stablecoin types, chains, and date ranges
4. WHEN the pipeline completes THEN the System SHALL store results in a format consumable by the marimo visualization layer
5. WHEN scheduling the pipeline THEN the System SHALL support weekly cron execution via ZenML's scheduling capabilities or external scheduler integration

### Requirement 11

**User Story:** As a data scientist, I want to predict whether a holder will become a store-of-value user, so that I can identify potential long-term holders early.

#### Acceptance Criteria

1. WHEN preparing training data THEN the System SHALL extract features from holder history including: transaction frequency, average transaction size, holding period trends, balance volatility, and time since first activity
2. WHEN defining the prediction target THEN the System SHALL use a temporal prediction window where:
   - Target label: Whether holder transitions to is_store_of_value=True within the next 30 days (configurable: 30/60/90 days)
   - Features computed from data up to time T; label computed from data at T + prediction_window
   - Holders already classified as SoV at time T are excluded from training
3. WHEN splitting data THEN the System SHALL use time-based splits to prevent data leakage:
   - Training set: Data from oldest 70% of time range
   - Validation set: Next 15% of time range
   - Test set: Most recent 15% of time range
   - Rolling time-series cross-validation with 5 folds for hyperparameter tuning
4. WHEN handling class imbalance THEN the System SHALL:
   - Apply class weights inversely proportional to class frequency in the loss function (default)
   - Optionally apply SMOTE oversampling on training set only (configurable)
   - Tune classification threshold on validation set to optimize F1-score
5. WHEN training the SoV prediction model THEN the System SHALL:
   - Use XGBoost as the default algorithm (alternatives: RandomForest, LightGBM)
   - Constrain hyperparameters: max_depth ≤ 10, learning_rate ∈ [0.01, 0.3], n_estimators ∈ [50, 500]
   - Apply early stopping on validation loss with patience=10 rounds
6. WHEN tuning hyperparameters THEN the System SHALL use Bayesian optimization (Optuna) with:
   - Search space: n_estimators [50, 500], max_depth [3, 10], learning_rate [0.01, 0.3], subsample [0.6, 1.0], colsample_bytree [0.6, 1.0]
   - Optimization target: Validation F1-score
   - Maximum 50 trials with early pruning
7. WHEN evaluating the model THEN the System SHALL calculate and display precision, recall, F1-score, and AUC-ROC metrics on the held-out test set
8. WHEN the model is trained THEN the System SHALL register it in ZenML's model registry with version metadata including hyperparameters, metrics, and data split timestamps
9. WHEN running inference THEN the System SHALL predict SoV probability for each holder and store predictions as a ZenML artifact

### Requirement 12

**User Story:** As a data scientist, I want to classify wallet behavior patterns, so that I can segment users into meaningful categories for analysis.

#### Acceptance Criteria

1. WHEN defining wallet behavior classes THEN the System SHALL use Taxonomy v1.0 with exactly four mutually exclusive classes:
   - **trader**: transaction_frequency > 1.0 tx/day AND holding_period_days < 7
   - **holder**: transaction_frequency < 0.1 tx/day AND holding_period_days > 30
   - **whale**: balance_percentile >= 99 (top 1% by balance, regardless of activity pattern)
   - **retail**: All wallets not matching above criteria
   - Taxonomy version stored in model metadata; future versions require explicit migration
2. WHEN preparing classification features THEN the System SHALL extract: transaction_count, avg_transaction_size, balance_percentile, holding_period_days, activity_recency_days, transaction_frequency, and cross_chain_flag
3. WHEN handling multi-class imbalance THEN the System SHALL:
   - Apply class weights inversely proportional to class frequency in the loss function
   - Use stratified sampling to preserve class distribution in train/val/test splits
   - Optionally apply SMOTE-NC (for mixed numeric/categorical features) on training set only
4. WHEN training the classifier THEN the System SHALL use XGBoost multi-class classifier (alternatives: RandomForest, LightGBM) with the same hyperparameter constraints as Requirement 11
5. WHEN computing confidence scores THEN the System SHALL:
   - Use softmax probabilities from the classifier as raw confidence
   - Apply Platt scaling calibration on validation set to produce calibrated probabilities
   - Report both raw and calibrated confidence in outputs
   - Flag predictions with calibrated confidence < 0.6 as "low confidence"
6. WHEN the classifier runs THEN the System SHALL assign each wallet to exactly one behavior class with confidence score
7. WHEN generating model outputs THEN the System SHALL include interpretability artifacts:
   - Global feature importances (gain-based for tree models)
   - Per-wallet explanations via SHAP values for top 100 highest-value wallets
   - Feature importance visualization in the Notebook
8. WHEN displaying classification results THEN the Notebook SHALL show the distribution of wallet types, confidence distribution, and allow filtering analysis by behavior class

### Requirement 13

**User Story:** As a data analyst, I want to trigger ZenML pipelines from within the marimo notebook, so that I can interactively run data collection and analysis workflows.

#### Acceptance Criteria

1. WHEN the notebook loads THEN the System SHALL display available ZenML pipelines and their last run status
2. WHEN a user clicks "Run Pipeline" THEN the System SHALL trigger the selected ZenML pipeline with user-specified parameters
3. WHILE a pipeline is running THEN the Notebook SHALL display progress indicators and step completion status
4. WHEN a pipeline completes THEN the Notebook SHALL automatically load the output artifacts for visualization
5. IF a pipeline fails THEN the Notebook SHALL display the error message and failed step information
6. WHEN a user attempts to trigger a pipeline THEN the System SHALL verify the user is authenticated via a valid session token or service account credential before allowing execution
7. WHEN a user triggers a pipeline THEN the System SHALL enforce role-based authorization (RBAC) where:
   - `viewer` role: Can view pipeline status and results only
   - `analyst` role: Can trigger analysis and inference pipelines
   - `admin` role: Can trigger all pipelines including training and collection
   - Parameter scopes are restricted by role (e.g., only admin can set `min_successful_collectors` below default)
8. WHEN any pipeline trigger occurs THEN the System SHALL create an audit log entry containing: user_id, pipeline_name, parameters, timestamp, and trigger_source (notebook/cron/api)
9. WHEN pipeline triggers exceed rate limits THEN the System SHALL:
   - Enforce per-user limit of 10 pipeline triggers per hour
   - Enforce global limit of 50 pipeline triggers per hour
   - Return HTTP 429 with retry-after header when limits are exceeded
   - Log rate limit violations to the audit log

### Requirement 14

**User Story:** As a platform operator, I want the marimo notebook to serve as a visualization layer for ZenML pipeline outputs, so that results are displayed on the live website.

#### Acceptance Criteria

1. WHEN the notebook initializes THEN the System SHALL query ZenML for the latest successful pipeline run artifacts
2. WHEN displaying pipeline results THEN the Notebook SHALL render all analysis visualizations (activity breakdown, stablecoin comparison, holder analysis, time series, chain comparison) from the pipeline artifacts
3. WHEN displaying ML predictions THEN the Notebook SHALL show SoV prediction distributions and wallet behavior classification breakdowns
4. WHEN new pipeline results are available THEN the Notebook SHALL provide a refresh mechanism to load updated data
5. WHEN exporting for web display THEN the Notebook SHALL generate static HTML/JSON outputs suitable for embedding in a live website

### Requirement 15

**User Story:** As a platform operator, I want experiment tracking and model versioning, so that I can compare ML model performance over time.

#### Acceptance Criteria

1. WHEN a training pipeline runs THEN the System SHALL log hyperparameters, metrics, and model artifacts to ZenML's experiment tracker
2. WHEN comparing model versions THEN the Notebook SHALL display a table of model runs with their metrics (precision, recall, F1, AUC)
3. WHEN selecting a model for production THEN the System SHALL support promoting a specific model version to "production" status in the registry
4. WHEN running inference THEN the System SHALL use the production model version by default, with option to specify a different version
5. WHEN detecting model performance degradation THEN the System SHALL:
   - Apply default absolute thresholds: precision >= 0.70, recall >= 0.65, F1 >= 0.67, AUC >= 0.75
   - Apply relative degradation threshold: Flag if any metric drops > 10% from the current production model
   - Thresholds configurable via `config/ml_thresholds.json` with schema: `{"precision_min": 0.70, "recall_min": 0.65, "f1_min": 0.67, "auc_min": 0.75, "relative_drop_max": 0.10}`
   - Thresholds editable via admin UI in the Notebook (requires admin role)
6. WHEN a model is flagged for degradation THEN the System SHALL:
   - Create an audit log entry with model_name, version, metrics, thresholds violated, and timestamp
   - Send a system alert (configurable: email, Slack webhook, or log-only)
   - Block automatic promotion to production (manual override requires admin approval)
   - Display a warning banner in the Notebook model comparison UI
7. WHEN reviewing flagged models THEN the Notebook SHALL display:
   - Side-by-side comparison with production model metrics
   - Metrics trend chart showing performance over last 10 versions
   - Option for admin to acknowledge and override the flag with justification


---

## Appendix A: JSON Schema Specification

This appendix defines the canonical JSON schema for stablecoin export data. All validation in Requirement 1 Acceptance Criteria 2 and 4 SHALL use this schema.

### Top-Level Structure

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| metadata | object | Yes | Collection run metadata |
| summary | object | Yes | Aggregated statistics |
| transactions | array | Yes | List of transaction records |
| holders | array | Yes | List of holder records |
| errors | array | No | Collection errors (if any) |

### Metadata Object

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| run_id | string | Yes | Non-empty, unique identifier | UUID or unique run identifier |
| collection_timestamp | string | Yes | ISO-8601 format | When data was collected |
| agent_version | string | Yes | Semantic version format | Version of collection agent |
| explorers_queried | array[string] | Yes | Non-empty array | List of explorer sources |
| total_records | integer | Yes | >= 0 | Total transaction + holder count |
| user_id | string | No | - | User who initiated collection |

### Transaction Object

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| transaction_hash | string | Yes | Non-empty, unique per record | Blockchain transaction hash |
| block_number | integer | Yes | >= 0 | Block containing transaction |
| timestamp | string | Yes | ISO-8601 format | Transaction timestamp |
| from_address | string | Yes | Non-empty, valid address format | Sender wallet address |
| to_address | string | Yes | Non-empty, valid address format | Receiver wallet address |
| amount | string | Yes | Decimal string, >= 0 | Transaction amount (as string for precision) |
| stablecoin | string | Yes | Enum: "USDC", "USDT" | Token type |
| chain | string | Yes | Enum: "ethereum", "bsc", "polygon" | Blockchain network |
| activity_type | string | Yes | Enum: "transaction", "store_of_value", "other" | Activity classification |
| source_explorer | string | Yes | Non-empty | Data source explorer name |
| gas_used | integer | No | >= 0 if present | Gas units consumed by transaction |
| gas_price | string | No | Decimal string, >= 0 if present | Gas price in wei (10^-18 of native token: ETH/BNB/MATIC) |

### Holder Object

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| address | string | Yes | Non-empty, valid address format | Wallet address |
| balance | string | Yes | Decimal string, >= 0 | Current token balance (as string) |
| stablecoin | string | Yes | Enum: "USDC", "USDT" | Token type |
| chain | string | Yes | Enum: "ethereum", "bsc", "polygon" | Blockchain network |
| first_seen | string | Yes | ISO-8601 format | First activity timestamp |
| last_activity | string | Yes | ISO-8601 format, >= first_seen | Last activity timestamp |
| is_store_of_value | boolean | Yes | true or false | SoV classification |
| source_explorer | string | Yes | Non-empty | Data source explorer name |

### Summary Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| by_stablecoin | object | Yes | Metrics grouped by stablecoin type |
| by_activity_type | object | Yes | Counts grouped by activity type |
| by_chain | object | Yes | Counts grouped by blockchain |

### Validation Failure Rules

The Notebook SHALL reject JSON data and display an error message when:

1. **Missing Required Field**: Any required field listed above is absent
2. **Wrong Type**: Field value does not match expected type (e.g., string where integer expected)
3. **Invalid Enum Value**: Field with enum constraint contains unlisted value
4. **Out-of-Range Value**: Numeric field is negative when >= 0 required
5. **Invalid Timestamp**: Timestamp field is not valid ISO-8601 format
6. **Invalid Decimal String**: Amount/balance field cannot be parsed as decimal
7. **Temporal Constraint Violation**: last_activity < first_seen for holder records
8. **Empty Required Array**: transactions or holders array is missing (empty arrays are valid)

### Example Valid JSON Document

```json
{
  "metadata": {
    "run_id": "550e8400-e29b-41d4-a716-446655440000",
    "collection_timestamp": "2024-12-12T10:30:00Z",
    "agent_version": "1.0.0",
    "explorers_queried": ["etherscan", "bscscan"],
    "total_records": 3
  },
  "summary": {
    "by_stablecoin": {
      "USDC": {"transaction_count": 1, "total_volume": "1000.50"},
      "USDT": {"transaction_count": 1, "total_volume": "500.00"}
    },
    "by_activity_type": {"transaction": 2, "store_of_value": 0, "other": 0},
    "by_chain": {"ethereum": 1, "bsc": 1, "polygon": 0}
  },
  "transactions": [
    {
      "transaction_hash": "0xabc123def456789...",
      "block_number": 18500000,
      "timestamp": "2024-12-10T14:22:33Z",
      "from_address": "0x1234567890abcdef1234567890abcdef12345678",
      "to_address": "0xabcdef1234567890abcdef1234567890abcdef12",
      "amount": "1000.50",
      "stablecoin": "USDC",
      "chain": "ethereum",
      "activity_type": "transaction",
      "source_explorer": "etherscan",
      "gas_used": 65000,
      "gas_price": "30000000000"
    },
    {
      "transaction_hash": "0xdef789abc123456...",
      "block_number": 35000000,
      "timestamp": "2024-12-11T09:15:00Z",
      "from_address": "0x9876543210fedcba9876543210fedcba98765432",
      "to_address": "0xfedcba9876543210fedcba9876543210fedcba98",
      "amount": "500.00",
      "stablecoin": "USDT",
      "chain": "bsc",
      "activity_type": "transaction",
      "source_explorer": "bscscan",
      "gas_used": 45000,
      "gas_price": "5000000000"
    }
  ],
  "holders": [
    {
      "address": "0x1234567890abcdef1234567890abcdef12345678",
      "balance": "5000.00",
      "stablecoin": "USDC",
      "chain": "ethereum",
      "first_seen": "2024-01-15T00:00:00Z",
      "last_activity": "2024-12-10T14:22:33Z",
      "is_store_of_value": false,
      "source_explorer": "etherscan"
    }
  ]
}
```


---

## Appendix B: Web Export Specification

This appendix defines the output schema and format for web exports generated by Requirement 14 AC5.

### Export Directory Structure

```
exports/
├── {run_id}/
│   ├── manifest.json           # Export metadata and file listing
│   ├── analysis/
│   │   ├── activity_breakdown.json
│   │   ├── stablecoin_comparison.json
│   │   ├── holder_metrics.json
│   │   ├── time_series.json
│   │   └── chain_metrics.json
│   ├── predictions/
│   │   ├── sov_predictions.json
│   │   └── wallet_classifications.json
│   ├── visualizations/
│   │   ├── activity_pie.html      # Standalone Altair chart
│   │   ├── stablecoin_bar.html
│   │   ├── holder_histogram.html
│   │   ├── time_series_line.html
│   │   ├── chain_stacked_bar.html
│   │   ├── sov_distribution.html
│   │   └── wallet_class_breakdown.html
│   └── summary/
│       ├── conclusions.json
│       └── dashboard.html         # Combined summary page
└── latest -> {most_recent_run_id}/  # Symlink to latest export
```

### Manifest Schema

```json
{
  "export_version": "1.0",
  "run_id": "string (UUID)",
  "pipeline_name": "master_pipeline",
  "export_timestamp": "ISO8601 datetime",
  "data_timestamp": "ISO8601 datetime (when data was collected)",
  "files": [
    {
      "path": "analysis/activity_breakdown.json",
      "type": "json",
      "size_bytes": 1234,
      "checksum_sha256": "abc123..."
    }
  ],
  "data_completeness": {
    "sources_successful": ["etherscan", "bscscan"],
    "sources_failed": ["polygonscan"],
    "completeness_ratio": 0.67
  },
  "model_versions": {
    "sov_predictor": "v1.2.3",
    "wallet_classifier": "v1.1.0"
  }
}
```

### Analysis JSON Schema

Each analysis JSON file follows this structure:

```json
{
  "schema_version": "1.0",
  "generated_at": "ISO8601 datetime",
  "run_id": "string",
  "data": { /* analysis-specific data */ },
  "metadata": {
    "record_count": 1000,
    "time_range": {
      "start": "ISO8601 datetime",
      "end": "ISO8601 datetime"
    }
  }
}
```

### Predictions JSON Schema

```json
{
  "schema_version": "1.0",
  "model_name": "sov_predictor",
  "model_version": "v1.2.3",
  "generated_at": "ISO8601 datetime",
  "predictions": [
    {
      "address": "0x1234...5678",  // Masked format
      "prediction": "true|false|class_name",
      "probability": 0.85,
      "confidence": "high|medium|low",
      "calibrated_probability": 0.82
    }
  ],
  "summary": {
    "total_predictions": 1000,
    "class_distribution": { "true": 300, "false": 700 },
    "low_confidence_count": 50
  }
}
```

### HTML Visualization Requirements

| Requirement | Specification |
|-------------|---------------|
| Standalone | Each HTML file is self-contained with embedded CSS/JS |
| Responsive | Charts resize to container width (min: 320px, max: 1200px) |
| Accessibility | WCAG 2.1 AA compliant: alt text, keyboard navigation, color contrast |
| Framework | Altair/Vega-Lite charts exported as HTML with Vega-Embed |
| Size limit | Individual HTML files < 500KB; total export < 10MB |
| Browser support | Chrome 90+, Firefox 88+, Safari 14+, Edge 90+ |

### Dashboard HTML Structure

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Stablecoin Analysis Dashboard - {run_id}</title>
  <style>/* Embedded responsive CSS */</style>
</head>
<body>
  <header>
    <h1>Stablecoin Usage Analysis</h1>
    <p class="timestamp">Data as of: {data_timestamp}</p>
    <p class="completeness" role="status">Data completeness: {completeness_ratio}%</p>
  </header>
  <main>
    <section id="summary" aria-label="Key Findings"><!-- Conclusions --></section>
    <section id="activity" aria-label="Activity Analysis"><!-- Charts --></section>
    <section id="predictions" aria-label="ML Predictions"><!-- Charts --></section>
  </main>
  <footer>
    <p>Generated by Stablecoin Analysis Pipeline v{version}</p>
    <p>Run ID: {run_id}</p>
  </footer>
</body>
</html>
```

### Versioned URLs

Exports are accessible via versioned URLs:
- Latest: `/exports/latest/manifest.json`
- Specific run: `/exports/{run_id}/manifest.json`
- Historical: `/exports/archive/{YYYY-MM}/` (monthly archives after 90 days)
