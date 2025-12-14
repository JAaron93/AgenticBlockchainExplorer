# Blockchain Stablecoin Explorer

An autonomous data analysis platform that collects, analyzes, and predicts usage patterns of USDC and USDT stablecoins across multiple blockchain networks.

## Purpose

- Collect stablecoin transaction data from blockchain explorers (Etherscan, BscScan, Polygonscan)
- Classify activity types: transactions, store of value, and other (minting/burning)
- Aggregate and deduplicate data from multiple sources
- Run ML inference for SoV prediction and wallet classification
- Provide interactive visualizations via Marimo notebooks
- Export structured JSON output for analysis

## Key Features

- Multi-chain support: Ethereum, BSC, Polygon
- Auth0-based authentication and authorization
- RESTful API for triggering data collection and retrieving results
- ZenML pipeline orchestration for scheduled data collection and analysis
- Machine learning: SoV (Store of Value) prediction and wallet classification
- Interactive Marimo dashboards with Altair visualizations
- Background job processing for long-running collection tasks
- Audit logging for user actions
- Rate limiting and retry logic for API calls
- Comprehensive security hardening (SSRF protection, credential sanitization, resource limits)

## User Workflow

### API-Based Workflow
1. User authenticates via Auth0 OAuth flow
2. User triggers an agent run via POST `/api/agent/run`
3. Agent collects data from configured explorers in background
4. User checks status via GET `/api/agent/status/{run_id}`
5. User retrieves results via GET `/api/results/{run_id}`
6. User downloads JSON output via GET `/api/results/{run_id}/download`

### Pipeline-Based Workflow
1. ZenML master pipeline runs on weekly cron schedule (Sundays at midnight UTC)
2. Pipeline collects data from all explorers in parallel
3. Data is aggregated, deduplicated, and analyzed
4. ML inference runs for SoV prediction and wallet classification
5. Results are versioned as ZenML artifacts
6. Marimo notebook consumes artifacts for visualization

### Visualization Workflow
1. Open Marimo notebook: `marimo edit notebooks/stablecoin_analysis.py`
2. Load JSON export files or trigger ZenML pipeline from notebook
3. Explore interactive charts: activity breakdown, stablecoin comparison, holder analysis
4. View ML predictions and wallet classifications
5. Generate summary conclusions with confidence indicators

## Data Flow

```
Blockchain Explorers → Collectors → Aggregator → Analysis → ML Inference → Visualization
     ↓                    ↓            ↓           ↓            ↓              ↓
  Etherscan          Rate-limited   Dedup &    Activity,    SoV Pred,     Marimo +
  BscScan            API calls      Merge      Holder,      Wallet        Altair
  Polygonscan                                  TimeSeries   Classify      Charts
```
