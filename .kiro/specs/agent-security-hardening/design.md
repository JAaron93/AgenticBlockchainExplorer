# Design Document: Agent Security Hardening

## Overview

This design document specifies the security hardening implementation for the blockchain explorer agents. The system collects stablecoin transaction data from external APIs (Etherscan, BscScan, Polygonscan) and must protect against credential leakage, SSRF attacks, resource exhaustion, and malicious input data.

The implementation adds security layers to the existing collector infrastructure without breaking current functionality. All security components are designed to be configurable and fail-safe (deny by default).

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Agent Orchestrator                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │ Timeout Manager │  │ Resource Monitor│  │ Graceful Terminator │ │
│  └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘ │
└───────────┼────────────────────┼─────────────────────┼─────────────┘
            │                    │                     │
┌───────────▼────────────────────▼─────────────────────▼─────────────┐
│                      Security Middleware Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  Credential  │  │    SSRF      │  │    Input Validator       │  │
│  │  Sanitizer   │  │  Protector   │  │  (Address/Hash/Amount)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  Safe Path   │  │  Response    │  │    Secure Logger         │  │
│  │  Handler     │  │  Size Limiter│  │  (Redaction Filter)      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
            │
┌───────────▼─────────────────────────────────────────────────────────┐
│                    Secure HTTP Client Wrapper                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │   Domain     │  │   Protocol   │  │    Redirect              │  │
│  │   Allowlist  │  │   Enforcer   │  │    Validator             │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
            │
            ▼
    External Explorer APIs (HTTPS only)
```

## Components and Interfaces

### 1. CredentialSanitizer

Responsible for detecting and redacting sensitive values from strings.

```python
class CredentialSanitizer:
    """Detects and redacts credentials from strings."""
    
    def __init__(self, config: CredentialSanitizerConfig):
        """Initialize with configurable patterns."""
        
    def sanitize(self, text: str) -> str:
        """Remove all detected credentials from text."""
        
    def sanitize_dict(self, data: dict) -> dict:
        """Recursively sanitize all string values in a dictionary."""
        
    def sanitize_url(self, url: str) -> str:
        """Sanitize URL query parameters containing credentials."""
        
    def is_credential(self, key: str, value: str) -> bool:
        """Check if a key-value pair appears to be a credential."""
```

### 2. SecureLogger

Logging wrapper that automatically sanitizes all output.

```python
class SecureLogger:
    """Logger wrapper that sanitizes credentials from all output."""
    
    def __init__(self, logger: logging.Logger, sanitizer: CredentialSanitizer):
        """Wrap an existing logger with sanitization."""
        
    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info with sanitization."""
        
    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning with sanitization."""
        
    def error(self, msg: str, *args, **kwargs) -> None:
        """Log error with sanitization."""
        
    def exception(self, msg: str, *args, **kwargs) -> None:
        """Log exception with stack trace sanitization."""
```

### 3. DomainAllowlist

Validates outbound request domains against configured allowlist.

```python
class DomainAllowlist:
    """Validates domains against an allowlist with pattern support."""
    
    def __init__(self, patterns: List[str]):
        """Initialize with domain patterns (exact or wildcard)."""
        
    def is_allowed(self, domain: str) -> bool:
        """Check if domain matches any allowlist pattern."""
        
    def validate_url(self, url: str) -> bool:
        """Validate full URL including protocol (HTTPS required)."""
        
    @classmethod
    def from_config(cls, config: Config) -> "DomainAllowlist":
        """Create from application configuration."""
```

### 4. SSRFProtector

Prevents SSRF attacks by validating all outbound requests.

```python
class SSRFProtector:
    """Protects against SSRF by validating outbound requests."""
    
    PRIVATE_IP_RANGES = [
        # IPv4 Private/Reserved
        ipaddress.ip_network("0.0.0.0/8"),       # "This" network
        ipaddress.ip_network("10.0.0.0/8"),      # Private (RFC 1918)
        ipaddress.ip_network("100.64.0.0/10"),   # Carrier-grade NAT
        ipaddress.ip_network("127.0.0.0/8"),     # Loopback
        ipaddress.ip_network("169.254.0.0/16"),  # Link-local
        ipaddress.ip_network("172.16.0.0/12"),   # Private (RFC 1918)
        ipaddress.ip_network("192.0.0.0/24"),    # IETF Protocol Assignments
        ipaddress.ip_network("192.0.2.0/24"),    # TEST-NET-1
        ipaddress.ip_network("192.168.0.0/16"),  # Private (RFC 1918)
        ipaddress.ip_network("198.18.0.0/15"),   # Benchmarking
        ipaddress.ip_network("198.51.100.0/24"), # TEST-NET-2
        ipaddress.ip_network("203.0.113.0/24"),  # TEST-NET-3
        ipaddress.ip_network("224.0.0.0/4"),     # Multicast
        ipaddress.ip_network("240.0.0.0/4"),     # Reserved for future use
        # IPv6 Private/Reserved
        ipaddress.ip_network("::/128"),          # Unspecified
        ipaddress.ip_network("::1/128"),         # Loopback
        ipaddress.ip_network("::ffff:0:0/96"),   # IPv4-mapped IPv6
        ipaddress.ip_network("64:ff9b::/96"),    # IPv4/IPv6 translation
        ipaddress.ip_network("100::/64"),        # Discard prefix
        ipaddress.ip_network("fc00::/7"),        # Unique local (ULA)
        ipaddress.ip_network("fe80::/10"),       # Link-local
        ipaddress.ip_network("ff00::/8"),        # Multicast
    ]
    
    # DNS resolution cache TTL for pinning (seconds)
    DNS_PIN_TTL = 60
    
    def __init__(self, allowlist: DomainAllowlist):
        """Initialize with domain allowlist."""
        self._dns_cache: Dict[str, Tuple[str, float]] = {}  # domain -> (ip, timestamp)
        
    async def validate_request(self, url: str) -> None:
        """Validate URL before making request. Raises SSRFError if invalid."""
        
    async def validate_redirect(
        self, 
        original_url: str, 
        redirect_url: str,
        original_resolved_ip: Optional[str] = None,
    ) -> None:
        """Validate redirect target including DNS rebinding check.
        
        DNS Rebinding Protection:
        1. Always perform fresh DNS resolution at validation time
        2. Pin the resolved IP for DNS_PIN_TTL seconds per domain
        3. Both original_url and redirect_url are resolved and checked
        4. Reject if either resolves to private/internal IP
        5. Detect rebinding: if same domain resolves to different IP within
           pin window, and new IP is private while old was public, reject
        6. On resolution mismatch (public→private), fail immediately
        7. On transient DNS failure, retry once after 1 second, then fail
        
        Args:
            original_url: The URL that returned the redirect
            redirect_url: The redirect target URL
            original_resolved_ip: Optional cached IP from original request
            
        Raises:
            SSRFError: If redirect target fails validation
            PrivateIPError: If redirect resolves to private IP
            DomainNotAllowedError: If redirect domain not in allowlist
        """
        
    async def _resolve_and_validate(self, hostname: str) -> str:
        """Resolve hostname and validate IP is not private.
        
        Uses pinned resolution if within TTL, otherwise fresh resolution.
        Returns the resolved IP address.
        Raises PrivateIPError if IP is in private ranges.
        """
        
    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP is in private/internal ranges."""
        
    def _is_rebinding_attempt(
        self, 
        domain: str, 
        old_ip: str, 
        new_ip: str
    ) -> bool:
        """Detect DNS rebinding: public IP changed to private IP."""
```

### 5. BlockchainDataValidator

Validates blockchain-specific data formats.

```python
class BlockchainDataValidator:
    """Validates blockchain data formats (addresses, hashes, amounts)."""
    
    ADDRESS_PATTERN = re.compile(r"^0x[a-fA-F0-9]{40}$")
    TX_HASH_PATTERN = re.compile(r"^0x[a-fA-F0-9]{64}$")
    AMOUNT_PATTERN = re.compile(r"^[0-9]+(\.[0-9]{1,18})?$")
    MAX_AMOUNT = 2**256 - 1
    
    # Genesis timestamps for supported chains
    GENESIS_TIMESTAMPS = {
        "ethereum": 1438269973,  # July 30, 2015
        "bsc": 1598671449,       # August 29, 2020
        "polygon": 1590824836,   # May 30, 2020
    }
    
    def validate_address(self, address: str) -> bool:
        """Validate Ethereum-style address format."""
        
    def validate_tx_hash(self, tx_hash: str) -> bool:
        """Validate transaction hash format."""
        
    def validate_amount(self, amount: str) -> bool:
        """Validate numeric amount format and bounds."""
        
    def validate_timestamp(self, timestamp: int, chain: str) -> bool:
        """Validate timestamp is within reasonable bounds for chain."""
        
    def normalize_address(self, address: str) -> str:
        """Normalize address to lowercase."""
        
    def validate_block_number(self, block_number: int, max_known: int) -> bool:
        """Validate block number is positive and not too far in future."""
```

### 6. SafePathHandler

Ensures file paths stay within allowed directories.

```python
class SafePathHandler:
    """Handles file paths safely, preventing directory traversal."""
    
    UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
    
    def __init__(self, base_directory: Path):
        """Initialize with base directory constraint."""
        
    def safe_join(self, *parts: str) -> Path:
        """Safely join path parts, raising error if result escapes base."""
        
    def sanitize_filename(self, filename: str) -> str:
        """Remove unsafe characters from filename."""
        
    def validate_path(self, path: Path) -> bool:
        """Check if path is within base directory."""
        
    def atomic_write(self, path: Path, content: bytes) -> None:
        """Write content atomically using temp file and rename."""
```

### 7. ResourceLimiter

Monitors and enforces resource limits.

```python
class ResourceLimiter:
    """Monitors and enforces resource consumption limits."""
    
    def __init__(self, config: ResourceLimitConfig):
        """Initialize with configured limits."""
        
    def check_response_size(self, size: int) -> None:
        """Raise error if response size exceeds limit."""
        
    def check_memory_usage(self) -> None:
        """Raise error if memory usage approaches limit."""
        
    def check_file_size(self, size: int) -> None:
        """Raise error if file size exceeds limit."""
        
    def get_current_memory_mb(self) -> float:
        """Get current process memory usage in MB."""
```

### 8. TimeoutManager

Manages collection timeouts at multiple levels.

```python
class TimeoutManager:
    """Manages hierarchical timeouts for agent runs."""
    
    def __init__(self, config: TimeoutConfig, num_collections: int):
        """Initialize with timeout config and expected collection count."""
        
    @property
    def overall_timeout(self) -> float:
        """Get overall run timeout in seconds."""
        
    @property
    def per_collection_timeout(self) -> float:
        """Get per-stablecoin collection timeout in seconds."""
        
    def time_remaining(self) -> float:
        """Get remaining time for overall run."""
        
    def should_terminate(self) -> bool:
        """Check if overall timeout is approaching (within 60s)."""
        
    def start(self) -> None:
        """Start the timeout clock."""
        
    async def run_with_timeout(self, coro, timeout: float) -> Any:
        """Run coroutine with timeout, raising TimeoutError if exceeded."""
```

### 9. GracefulTerminator

Handles graceful shutdown on timeout or resource exhaustion.

```python
class GracefulTerminator:
    """Handles graceful termination of agent runs."""
    
    def __init__(self, shutdown_timeout: float = 30.0):
        """Initialize with shutdown timeout."""
        
    async def terminate(
        self,
        reason: str,
        pending_tasks: List[asyncio.Task],
        partial_results: ExplorerData,
        output_handler: SafePathHandler,
    ) -> TerminationReport:
        """Execute graceful termination sequence."""
        
    def _cancel_tasks(self, tasks: List[asyncio.Task]) -> None:
        """Cancel all pending tasks."""
        
    async def _flush_results(
        self,
        results: ExplorerData,
        output_handler: SafePathHandler,
    ) -> str:
        """Write partial results with 'partial' status flag."""
```

### 10. SecureHTTPClient

HTTP client wrapper with all security protections.

```python
class SecureHTTPClient:
    """HTTP client with integrated security protections."""
    
    # Allowed parameter keys for explorer APIs
    ALLOWED_PARAM_KEYS = {
        "module", "action", "contractaddress", "address", "page", 
        "offset", "sort", "startblock", "endblock", "apikey"
    }
    
    def __init__(
        self,
        ssrf_protector: SSRFProtector,
        resource_limiter: ResourceLimiter,
        sanitizer: CredentialSanitizer,
        logger: SecureLogger,
    ):
        """Initialize with security components."""
        
    async def get(
        self,
        url: str,
        params: Optional[Dict] = None,
        timeout: float = 30.0,
    ) -> Dict:
        """Make GET request with all security validations.
        
        Parameter handling:
        1. Validate all param keys are in ALLOWED_PARAM_KEYS allowlist
        2. Validate param value types (str, int, bool only)
        3. URL-encode all param values to prevent injection
        4. Run CredentialSanitizer on param values for logging (never log raw apikey)
        5. Check if any param value looks like a URL and validate against SSRF allowlist
        6. Log only redacted params at debug level
        
        Raises:
            ValueError: If param key not in allowlist or invalid type
            SSRFError: If URL or param value fails SSRF validation
            ResponseTooLargeError: If response exceeds size limit
            asyncio.TimeoutError: If request exceeds timeout
        """
        
    def _validate_params(self, params: Dict) -> Dict:
        """Validate and sanitize request parameters.
        
        Returns sanitized params dict.
        Raises ValueError if validation fails.
        """
        
    def _sanitize_params_for_logging(self, params: Dict) -> Dict:
        """Create a copy of params with credentials redacted for logging."""
        
    async def _handle_redirect(
        self,
        response: aiohttp.ClientResponse,
        original_url: str,
    ) -> str:
        """Handle redirect with SSRF validation.
        
        Re-validates redirect target against domain allowlist and
        checks resolved IP is not in private ranges.
        """
```

## Data Models

### Configuration Models

```python
class CredentialSanitizerConfig(BaseModel):
    """Configuration for credential sanitization."""
    
    sensitive_param_names: List[str] = Field(
        default=["apikey", "api_key", "API_KEY", "token", "auth_token", 
                 "secret", "password", "client_secret"]
    )
    sensitive_header_names: List[str] = Field(
        default=["Authorization", "X-API-Key", "X-Auth-Token"]
    )
    credential_patterns: List[str] = Field(
        default=[
            r"[a-zA-Z0-9]{32,}",  # 32+ char alphanumeric
            r"eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+",  # JWT
        ]
    )
    redaction_placeholder: str = "[REDACTED]"


class SSRFProtectionConfig(BaseModel):
    """Configuration for SSRF protection."""
    
    allowed_domains: List[str] = Field(
        default=[
            "api.etherscan.io",
            "*.etherscan.io",
            "api.bscscan.com",
            "*.bscscan.com",
            "api.polygonscan.com",
            "*.polygonscan.com",
        ]
    )
    require_https: bool = True
    block_private_ips: bool = True


class ResourceLimitConfig(BaseModel):
    """Configuration for resource limits."""
    
    max_response_size_bytes: int = Field(default=10 * 1024 * 1024)  # 10MB
    max_output_file_size_bytes: int = Field(default=100 * 1024 * 1024)  # 100MB
    max_memory_usage_mb: int = Field(default=512)


class TimeoutConfig(BaseModel):
    """Configuration for timeouts."""
    
    overall_run_timeout_seconds: int = Field(default=1800)  # 30 minutes
    per_collection_timeout_seconds: int = Field(default=180)  # 3 minutes
    shutdown_timeout_seconds: int = Field(default=30)
    
    @model_validator(mode="after")
    def validate_timeouts(self) -> "TimeoutConfig":
        """Validate timeout relationships."""
        # Will be validated against actual collection count at runtime
        return self


class SecurityConfig(BaseModel):
    """Combined security configuration."""
    
    credential_sanitizer: CredentialSanitizerConfig = Field(
        default_factory=CredentialSanitizerConfig
    )
    ssrf_protection: SSRFProtectionConfig = Field(
        default_factory=SSRFProtectionConfig
    )
    resource_limits: ResourceLimitConfig = Field(
        default_factory=ResourceLimitConfig
    )
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)
```

### Result Models

```python
class TerminationReport(BaseModel):
    """Report generated on graceful termination."""
    
    reason: str
    timestamp: datetime
    records_collected: int
    records_persisted: int
    output_file: Optional[str]
    partial: bool = True
    duration_seconds: float
```

### ExplorerData Reference

The `ExplorerData` dataclass from `collectors/models.py` is used by GracefulTerminator. For security hardening, we extend the serialization with a status field:

```python
@dataclass
class ExplorerData:
    """Data collected from a blockchain explorer (existing model)."""
    
    explorer_name: str
    chain: str
    transactions: List[Transaction] = field(default_factory=list)
    holders: List[Holder] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    collection_time_seconds: float = 0.0
    
    @property
    def total_records(self) -> int:
        return len(self.transactions) + len(self.holders)


class ExplorerDataOutput(BaseModel):
    """JSON-serializable output format for ExplorerData with status."""
    
    explorer_name: str
    chain: str
    transactions: List[Dict[str, Any]]
    holders: List[Dict[str, Any]]
    errors: List[str]
    collection_time_seconds: float
    total_records: int
    # Security hardening additions
    status: Literal["complete", "partial"] = "complete"
    termination_reason: Optional[str] = None
    timestamp: str  # ISO-8601 format
    run_id: str
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v),
        }
```

**Serialization Notes:**
- All output is JSON with UTF-8 encoding
- `_flush_results` sets `status='partial'` when writing during graceful termination
- `termination_reason` is populated only for partial outputs
- All fields must be JSON-serializable (Decimal → str, datetime → ISO-8601)

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

Based on the prework analysis, the following correctness properties must be verified:

### Property 1: Credential Sanitization Completeness
*For any* string containing a credential pattern (API key, token, secret), the sanitized output SHALL NOT contain that credential value.
**Validates: Requirements 1.1, 1.2, 1.3, 1.5**

### Property 2: Domain Allowlist Enforcement
*For any* URL, the SSRF protector SHALL allow the request if and only if the domain matches an allowlist pattern AND the protocol is HTTPS.
**Validates: Requirements 2.4, 2.5, 2.6**

### Property 3: Private IP Blocking
*For any* IP address in private ranges (10.x, 172.16-31.x, 192.168.x, 127.x, ::1), the SSRF protector SHALL block requests resolving to that IP.
**Validates: Requirements 2.9**

### Property 4: Response Size Enforcement
*For any* API response exceeding the configured size limit, the resource limiter SHALL abort the request before fully reading the response.
**Validates: Requirements 3.1, 3.2**

### Property 5: Address Validation Correctness
*For any* string, the address validator SHALL accept it if and only if it matches the pattern "^0x[a-fA-F0-9]{40}$".
**Validates: Requirements 4.1**

### Property 6: Transaction Hash Validation Correctness
*For any* string, the transaction hash validator SHALL accept it if and only if it matches the pattern "^0x[a-fA-F0-9]{64}$".
**Validates: Requirements 4.2**

### Property 7: Amount Validation Correctness
*For any* string, the amount validator SHALL accept it if and only if it matches "^[0-9]+(\.[0-9]{1,18})?$" AND the numeric value is ≤ 2^256-1.
**Validates: Requirements 4.3**

### Property 8: Address Normalization Idempotence
*For any* valid address, normalizing it twice SHALL produce the same result as normalizing once (lowercase).
**Validates: Requirements 4.5**

### Property 9: Path Containment
*For any* path constructed via SafePathHandler, the resolved absolute path SHALL be within the configured base directory.
**Validates: Requirements 5.1, 5.2**

### Property 10: Filename Sanitization Safety
*For any* filename after sanitization, it SHALL NOT contain path traversal sequences (../, ..\) or null bytes.
**Validates: Requirements 5.3**

## Error Handling

### Security Errors

All security violations raise specific exception types:

```python
class SecurityError(Exception):
    """Base class for security-related errors."""
    pass

class CredentialLeakageError(SecurityError):
    """Raised when credential leakage is detected."""
    pass

class SSRFError(SecurityError):
    """Raised when SSRF protection blocks a request."""
    pass

class DomainNotAllowedError(SSRFError):
    """Raised when domain is not in allowlist."""
    pass

class PrivateIPError(SSRFError):
    """Raised when request would resolve to private IP."""
    pass

class ResourceLimitError(SecurityError):
    """Raised when resource limits are exceeded."""
    pass

class ResponseTooLargeError(ResourceLimitError):
    """Raised when response exceeds size limit."""
    pass

class PathTraversalError(SecurityError):
    """Raised when path traversal is detected."""
    pass

class ValidationError(SecurityError):
    """Raised when blockchain data validation fails."""
    pass
```

### Error Handling Strategy

1. **Security errors are never swallowed** - All security violations are logged and propagated
2. **Fail-safe defaults** - On configuration errors, use most restrictive settings
3. **Graceful degradation** - On resource limits, save partial results before terminating
4. **No credential exposure in errors** - All error messages are sanitized before logging

## Testing Strategy

### Dual Testing Approach

The implementation uses both unit tests and property-based tests:

- **Unit tests**: Verify specific examples, edge cases, and integration points
- **Property-based tests**: Verify universal properties hold across all valid inputs

### Property-Based Testing Framework

Use **Hypothesis** for Python property-based testing. Configure minimum 100 iterations per property.

### Test Organization

```
tests/
├── security/
│   ├── test_credential_sanitizer.py      # Unit + Property tests
│   ├── test_ssrf_protector.py            # Unit + Property tests
│   ├── test_blockchain_validator.py      # Unit + Property tests
│   ├── test_safe_path_handler.py         # Unit + Property tests
│   ├── test_resource_limiter.py          # Unit tests
│   ├── test_timeout_manager.py           # Unit tests
│   └── test_secure_http_client.py        # Integration tests
```

### Property Test Annotations

Each property-based test must be annotated with the correctness property it validates:

```python
# **Feature: agent-security-hardening, Property 1: Credential Sanitization Completeness**
@given(text=st.text(), credential=st.from_regex(r"[a-zA-Z0-9]{32,64}"))
def test_credential_sanitization_completeness(text, credential):
    ...
```
