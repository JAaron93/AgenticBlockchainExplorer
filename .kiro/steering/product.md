# Blockchain Stablecoin Explorer

An autonomous agent that collects and analyzes USDC and USDT stablecoin usage data across multiple blockchain networks.

## Purpose

- Collect stablecoin transaction data from blockchain explorers (Etherscan, BscScan, Polygonscan)
- Classify activity types: transactions, store of value, and other (minting/burning)
- Aggregate and deduplicate data from multiple sources
- Export structured JSON output for analysis

## Key Features

- Multi-chain support: Ethereum, BSC, Polygon
- Auth0-based authentication and authorization
- RESTful API for triggering data collection and retrieving results
- Background job processing for long-running collection tasks
- Audit logging for user actions
- Rate limiting and retry logic for API calls

## User Workflow

1. User authenticates via Auth0 OAuth flow
2. User triggers an agent run via POST `/api/agent/run`
3. Agent collects data from configured explorers in background
4. User checks status via GET `/api/agent/status/{run_id}`
5. User retrieves results via GET `/api/results/{run_id}`
6. User downloads JSON output via GET `/api/results/{run_id}/download`
