# API Documentation

This document describes the REST API for the Blockchain Stablecoin Explorer.

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Endpoints](#endpoints)
   - [Health Checks](#health-checks)
   - [Authentication Endpoints](#authentication-endpoints)
   - [Agent Control Endpoints](#agent-control-endpoints)
   - [Results Endpoints](#results-endpoints)
4. [Error Handling](#error-handling)
5. [Rate Limiting](#rate-limiting)

---

## Overview

**Base URL:** `http://localhost:8000` (development) or `https://your-domain.com` (production)

**API Version:** 1.0.0

**Content Type:** All requests and responses use `application/json` unless otherwise specified.

### Interactive Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI:** `GET /docs` - Interactive API explorer
- **ReDoc:** `GET /redoc` - Alternative documentation format
- **OpenAPI Schema:** `GET /openapi.json` - Raw OpenAPI 3.0 specification

---

## Authentication

The API uses **Auth0** for OAuth 2.0 authentication with JWT tokens.

### Authentication Flow

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Client  │────▶│  /login  │────▶│  Auth0   │────▶│ /callback│
└──────────┘     └──────────┘     └──────────┘     └──────────┘
                                                         │
                                                         ▼
                                                  Access Token
```

1. **Initiate Login:** Redirect user to `GET /login`
2. **Auth0 Authentication:** User authenticates with Auth0
3. **Callback:** Auth0 redirects to `/callback` with authorization code
4. **Token Exchange:** Application exchanges code for access token
5. **Use Token:** Include token in `Authorization` header for API requests

### Using Access Tokens

Include the access token in the `Authorization` header:

```http
Authorization: Bearer <access_token>
```

### Permissions

| Permission | Description |
|------------|-------------|
| `run:agent` | Trigger data collection runs |
| `view:results` | View collection results and run status |
| `download:data` | Download JSON output files |
| `admin:config` | Administrative access (view all users' runs) |

---

## Endpoints

### Health Checks

#### GET /

Root endpoint - basic service information.

**Authentication:** None required

**Response:**
```json
{
  "status": "ok",
  "service": "Blockchain Stablecoin Explorer",
  "version": "1.0.0"
}
```

---

#### GET /health

Basic health check for load balancers.

**Authentication:** None required

**Response:**
```json
{
  "status": "healthy"
}
```

---

#### GET /health/ready

Readiness check - verifies all dependencies are available.

**Authentication:** None required

**Response (200 OK):**
```json
{
  "status": "healthy",
  "checks": {
    "database": "healthy",
    "auth0": "healthy",
    "config": "healthy"
  },
  "version": "1.0.0"
}
```

**Response (503 Service Unavailable):**
```json
{
  "status": "unhealthy",
  "checks": {
    "database": "unhealthy: connection refused",
    "auth0": "healthy",
    "config": "healthy"
  },
  "version": "1.0.0"
}
```

---

#### GET /health/live

Liveness check - verifies the application is running.

**Authentication:** None required

**Response:**
```json
{
  "status": "alive"
}
```

---

### Authentication Endpoints

#### GET /login

Initiates the OAuth 2.0 authorization flow by redirecting to Auth0.

**Authentication:** None required

**Response:** `302 Found` - Redirects to Auth0 login page

**Example:**
```bash
# Open in browser or follow redirect
curl -L http://localhost:8000/login
```

---

#### GET /callback

Handles the Auth0 callback after authentication.

**Authentication:** None required (receives authorization code from Auth0)

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `code` | string | Yes | Authorization code from Auth0 |
| `state` | string | Yes | State parameter for CSRF protection |
| `error` | string | No | Error code if authentication failed |
| `error_description` | string | No | Human-readable error description |

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 86400,
  "user_id": "auth0|123456789",
  "email": "[email]"
}
```

**Error Response (400 Bad Request):**
```json
{
  "detail": "Authentication failed: access_denied"
}
```

---

#### GET /logout

Logs out the user and redirects to Auth0 logout.

**Authentication:** None required

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `return_to` | string | No | URL to redirect after logout |

**Response:** `302 Found` - Redirects to Auth0 logout, then to `return_to` URL

**Example:**
```bash
curl -L "http://localhost:8000/logout?return_to=http://localhost:8000"
```

---

#### GET /csrf-token

Get a CSRF token for state-changing requests to non-API endpoints.

**Authentication:** None required

**Response:**
```json
{
  "csrf_token": "abc123..."
}
```

**Note:** API endpoints (`/api/*`) use JWT authentication instead of CSRF tokens.

---

### Agent Control Endpoints

#### POST /api/agent/run

Trigger a new data collection run.

**Authentication:** Required - `run:agent` permission

**Request Headers:**
```http
Authorization: Bearer <access_token>
Content-Type: application/json
```

**Request Body (optional):**
```json
{
  "max_records_per_explorer": 500,
  "explorers": ["etherscan", "bscscan"],
  "stablecoins": ["USDC"]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `max_records_per_explorer` | integer | No | Max records per explorer (1-100000) |
| `explorers` | array | No | Explorer names to query (default: all) |
| `stablecoins` | array | No | Stablecoins to collect (default: all) |

**Response (200 OK):**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Agent run started successfully",
  "started_at": "2025-12-12T10:30:00Z"
}
```

**Example:**
```bash
# Start a run with default settings
curl -X POST http://localhost:8000/api/agent/run \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json"

# Start a run with custom settings
curl -X POST http://localhost:8000/api/agent/run \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "max_records_per_explorer": 100,
    "explorers": ["etherscan"],
    "stablecoins": ["USDC", "USDT"]
  }'
```

---

#### GET /api/agent/status/{run_id}

Get the status of an agent run.

**Authentication:** Required - `view:results` permission

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `run_id` | string (UUID) | The run identifier |

**Response (200 OK):**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "progress": 0.45,
  "progress_message": "Collecting from bscscan...",
  "started_at": "2025-12-12T10:30:00Z",
  "completed_at": null,
  "error_message": null
}
```

**Status Values:**

| Status | Description |
|--------|-------------|
| `pending` | Run created, waiting to start |
| `running` | Data collection in progress |
| `completed` | Run finished successfully |
| `failed` | Run failed with error |

**Example:**
```bash
curl http://localhost:8000/api/agent/status/550e8400-e29b-41d4-a716-446655440000 \
  -H "Authorization: Bearer <token>"
```

---

### Results Endpoints

#### GET /api/results

List all runs for the authenticated user.

**Authentication:** Required - `view:results` permission

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 50 | Max results to return (1-100) |
| `offset` | integer | 0 | Number of results to skip |

**Response (200 OK):**
```json
[
  {
    "run_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "started_at": "2025-12-12T10:30:00Z",
    "completed_at": "2025-12-12T10:32:15Z",
    "total_records": 2847,
    "explorers_queried": ["etherscan", "bscscan", "polygonscan"]
  },
  {
    "run_id": "660e8400-e29b-41d4-a716-446655440001",
    "status": "running",
    "started_at": "2025-12-12T11:00:00Z",
    "completed_at": null,
    "total_records": null,
    "explorers_queried": null
  }
]
```

**Example:**
```bash
# Get first 10 results
curl "http://localhost:8000/api/results?limit=10" \
  -H "Authorization: Bearer <token>"

# Get results with pagination
curl "http://localhost:8000/api/results?limit=10&offset=20" \
  -H "Authorization: Bearer <token>"
```

---

#### GET /api/results/{run_id}

Get detailed results for a specific run.

**Authentication:** Required - `view:results` permission

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `run_id` | string (UUID) | The run identifier |

**Response (200 OK):**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "auth0|123456789",
  "status": "completed",
  "started_at": "2025-12-12T10:30:00Z",
  "completed_at": "2025-12-12T10:32:15Z",
  "config": {
    "max_records_per_explorer": 1000,
    "explorers": ["etherscan", "bscscan", "polygonscan"],
    "stablecoins": ["USDC", "USDT"]
  },
  "error_message": null,
  "progress": 1.0,
  "progress_message": "Completed",
  "result": {
    "total_records": 2847,
    "explorers_queried": ["etherscan", "bscscan", "polygonscan"],
    "output_file_path": "./output/run_550e8400_20251212_103000.json",
    "summary": {
      "by_stablecoin": {
        "USDC": {"total_transactions": 1523, "total_volume": "45678901.23"},
        "USDT": {"total_transactions": 1324, "total_volume": "38901234.56"}
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
    }
  }
}
```

**Example:**
```bash
curl http://localhost:8000/api/results/550e8400-e29b-41d4-a716-446655440000 \
  -H "Authorization: Bearer <token>"
```

---

#### GET /api/results/{run_id}/download

Download the JSON output file for a run.

**Authentication:** Required - `download:data` permission

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `run_id` | string (UUID) | The run identifier |

**Response (200 OK):**
- Content-Type: `application/json`
- Content-Disposition: `attachment; filename="run_550e8400_20251212_103000.json"`

**Response Body:** JSON file containing all collected data

```json
{
  "metadata": {
    "run_id": "550e8400-e29b-41d4-a716-446655440000",
    "user_id": "auth0|123456789",
    "collection_timestamp": "2025-12-12T10:30:00Z",
    "agent_version": "1.0.0",
    "explorers_queried": ["etherscan", "bscscan", "polygonscan"],
    "total_records": 2847
  },
  "summary": {
    "by_stablecoin": {...},
    "by_activity_type": {...},
    "by_chain": {...}
  },
  "transactions": [
    {
      "transaction_hash": "0x...",
      "block_number": 18500000,
      "timestamp": "2025-12-12T09:15:23Z",
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

**Example:**
```bash
# Download to file
curl http://localhost:8000/api/results/550e8400-e29b-41d4-a716-446655440000/download \
  -H "Authorization: Bearer <token>" \
  -o results.json
```

---

## Error Handling

### Error Response Format

All errors return a JSON response with a `detail` field:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| `200` | Success |
| `302` | Redirect (for OAuth flows) |
| `400` | Bad Request - Invalid input or parameters |
| `401` | Unauthorized - Missing or invalid authentication |
| `403` | Forbidden - Insufficient permissions |
| `404` | Not Found - Resource doesn't exist |
| `429` | Too Many Requests - Rate limit exceeded |
| `500` | Internal Server Error |
| `503` | Service Unavailable - Dependency failure |

### Common Error Responses

**401 Unauthorized:**
```json
{
  "detail": "Not authenticated"
}
```

**403 Forbidden:**
```json
{
  "detail": "Insufficient permissions"
}
```

**404 Not Found:**
```json
{
  "detail": "Run not found"
}
```

**429 Too Many Requests:**
```json
{
  "detail": "Rate limit exceeded. Please wait before making more requests."
}
```

---

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Default limit:** 100 requests per minute per user
- **Scope:** Applied per authenticated user (by user ID)
- **Headers:** Rate limit info included in response headers

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1702383600
```

### Handling Rate Limits

When rate limited, the API returns `429 Too Many Requests`. Implement exponential backoff:

```python
import time
import requests

def make_request_with_retry(url, headers, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)
        
        if response.status_code == 429:
            wait_time = 2 ** attempt  # Exponential backoff
            time.sleep(wait_time)
            continue
            
        return response
    
    raise Exception("Max retries exceeded")
```

---

## SDK Examples

### Python

```python
import requests

class StablecoinExplorerClient:
    def __init__(self, base_url: str, access_token: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {access_token}"}
    
    def trigger_run(self, config: dict = None) -> dict:
        response = requests.post(
            f"{self.base_url}/api/agent/run",
            headers=self.headers,
            json=config or {}
        )
        response.raise_for_status()
        return response.json()
    
    def get_status(self, run_id: str) -> dict:
        response = requests.get(
            f"{self.base_url}/api/agent/status/{run_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def list_results(self, limit: int = 50) -> list:
        response = requests.get(
            f"{self.base_url}/api/results",
            headers=self.headers,
            params={"limit": limit}
        )
        response.raise_for_status()
        return response.json()
    
    def download_result(self, run_id: str, output_path: str):
        response = requests.get(
            f"{self.base_url}/api/results/{run_id}/download",
            headers=self.headers
        )
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(response.content)

# Usage
client = StablecoinExplorerClient(
    base_url="http://localhost:8000",
    access_token="your_access_token"
)

# Start a run
run = client.trigger_run({"max_records_per_explorer": 100})
print(f"Started run: {run['run_id']}")

# Check status
status = client.get_status(run['run_id'])
print(f"Status: {status['status']}")

# Download results when complete
if status['status'] == 'completed':
    client.download_result(run['run_id'], 'results.json')
```

### cURL

```bash
# Set your access token
TOKEN="your_access_token"

# Start a run
curl -X POST http://localhost:8000/api/agent/run \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"max_records_per_explorer": 100}'

# Check status
curl http://localhost:8000/api/agent/status/<run_id> \
  -H "Authorization: Bearer $TOKEN"

# List results
curl http://localhost:8000/api/results \
  -H "Authorization: Bearer $TOKEN"

# Download result
curl http://localhost:8000/api/results/<run_id>/download \
  -H "Authorization: Bearer $TOKEN" \
  -o results.json
```
