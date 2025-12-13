# Requirements Document

## Introduction

This specification defines security hardening requirements for the blockchain explorer agents in the stablecoin analysis system. The agents collect data from external blockchain explorer APIs (Etherscan, BscScan, Polygonscan) and process transaction/holder information. Given the system's deployment to production, these requirements address identified attack vectors including credential exposure, SSRF, resource exhaustion, and input validation gaps.

## Glossary

- **Agent**: The autonomous data collection component that queries blockchain explorer APIs
- **Explorer API**: External blockchain data services (Etherscan, BscScan, Polygonscan)
- **Credential**: Sensitive authentication data including API keys, secrets, and tokens
- **SSRF**: Server-Side Request Forgery - an attack where the server is tricked into making requests to unintended destinations
- **Allowlist**: A list of explicitly permitted values (domains, patterns, etc.)
- **Sanitization**: The process of cleaning or validating input data to remove potentially harmful content
- **Blockchain Address**: A 40-character hexadecimal string prefixed with "0x" (42 characters total)
- **Transaction Hash**: A 64-character hexadecimal string prefixed with "0x" (66 characters total)

## Requirements

### Requirement 1: Credential Protection

**User Story:** As a system operator, I want API keys and secrets to never appear in logs, error messages, or API responses, so that credentials cannot be leaked through operational channels.

#### Acceptance Criteria

1. WHEN the system logs any message containing an API key pattern THEN the Agent SHALL replace the API key with a redacted placeholder "[REDACTED]"
2. WHEN an error occurs during API communication THEN the Agent SHALL sanitize the error message to remove any credential values before logging or returning
3. WHEN the system constructs HTTP request URLs with API keys THEN the Agent SHALL exclude the full URL from debug logs, showing only the base URL and non-sensitive parameters
4. WHEN exception stack traces are logged THEN the Agent SHALL filter local variables that may contain credential values
5. WHEN API responses contain error messages that echo back request parameters THEN the Agent SHALL sanitize credentials from the response before processing
6. WHEN detecting credentials for redaction THEN the system SHALL use a configurable list of sensitive parameter names including: "apikey", "api_key", "API_KEY", "token", "auth_token", "secret", "password", "client_secret"
7. WHEN detecting credentials for redaction THEN the system SHALL use a configurable list of sensitive header names including: "Authorization", "X-API-Key", "X-Auth-Token"
8. WHEN detecting credentials for redaction THEN the system SHALL match values against configurable regex patterns for common key formats (e.g., 32+ character alphanumeric strings, JWT patterns "eyJ...")
9. WHEN the credential detection configuration is loaded THEN the system SHALL merge default patterns with any user-provided patterns from configuration

### Requirement 2: SSRF Protection via Domain Allowlisting

**User Story:** As a security engineer, I want outbound HTTP requests restricted to known blockchain explorer domains, so that attackers cannot use the agent to probe internal networks or arbitrary external services.

#### Acceptance Criteria

1. WHEN the Agent initializes THEN the system SHALL validate that all configured explorer base URLs match the domain allowlist
2. WHEN the system starts THEN the domain allowlist SHALL be loaded from configuration (config.json or environment variable ALLOWED_EXPLORER_DOMAINS) and validated for correct pattern syntax
3. IF the domain allowlist configuration is invalid or empty THEN the system SHALL fail startup with a clear error message
4. WHEN the Agent makes an outbound HTTP request THEN the system SHALL verify the target domain is in the allowlist before sending
5. WHEN the Agent makes an outbound HTTP request THEN the system SHALL require HTTPS protocol and reject plain HTTP requests
6. IF a request targets a domain not in the allowlist THEN the system SHALL reject the request and log a security warning
7. WHEN the domain allowlist is configured THEN the system SHALL support exact domain matching and subdomain patterns (e.g., "*.etherscan.io")
8. WHEN a redirect response is received THEN the Agent SHALL validate the redirect target hostname against the allowlist before following
9. WHEN following a redirect THEN the Agent SHALL re-resolve DNS and validate the resolved IP is not in private/internal ranges (10.x, 172.16-31.x, 192.168.x, 127.x, ::1) to mitigate DNS rebinding attacks
10. WHEN the allowlist is loaded THEN the system SHALL log the configured patterns at INFO level for operational visibility

### Requirement 3: Resource Exhaustion Protection

**User Story:** As a system operator, I want limits on data collection volume and duration, so that malicious or malformed API responses cannot exhaust system memory or disk space.

#### Acceptance Criteria

1. WHEN receiving an API response THEN the Agent SHALL enforce a maximum response body size limit (configurable, default 10MB)
2. WHEN an API response exceeds the size limit THEN the Agent SHALL abort the request and log a warning
3. WHEN an agent run starts THEN the system SHALL enforce a maximum total runtime for the entire run across all stablecoins and explorers (configurable, default 30 minutes)
4. IF an agent run exceeds the maximum total runtime THEN the system SHALL initiate graceful termination
5. WHEN writing output files THEN the Agent SHALL enforce a maximum file size limit and fail gracefully if exceeded
6. WHEN collecting transactions THEN the Agent SHALL track memory usage and abort if approaching configured limits
7. WHEN initiating graceful termination THEN the system SHALL cancel any pending API requests immediately to prevent further data collection
8. WHEN initiating graceful termination THEN the system SHALL flush and write any in-progress results atomically to output with a "partial" status flag in metadata
9. WHEN initiating graceful termination THEN the system SHALL wait for file writes to complete subject to a configurable shutdown timeout (default 30 seconds)
10. WHEN graceful termination completes THEN the system SHALL record a structured log entry with termination reason, timestamp, records collected, and summary of persisted data
11. WHEN retrying failed API requests THEN the Agent SHALL use exponential backoff with configurable base delay (default 1 second), multiplier (default 2), and maximum delay (default 60 seconds)
12. WHEN an API response includes rate-limit headers THEN the Agent SHALL honor those headers when scheduling retries with the following precedence: (1) Retry-After header takes highest precedence if present and valid, (2) X-RateLimit-Reset is used if Retry-After is absent, (3) exponential backoff is used as fallback if both headers are absent or malformed
13. WHEN parsing Retry-After header THEN the Agent SHALL support both delta-seconds (numeric) and HTTP-date formats per RFC 7231; for HTTP-date values, compute delay as (parsedDate - currentUTC)
14. WHEN parsing X-RateLimit-Reset header THEN the Agent SHALL treat it as Unix epoch timestamp and compute delay as (resetTime - currentUTC)
15. IF both Retry-After and X-RateLimit-Reset headers are present with different valid values THEN the Agent SHALL use the larger of the two computed delays to ensure compliance with the stricter limit
16. IF a computed delay from rate-limit headers is negative or exceeds 3600 seconds (1 hour) THEN the Agent SHALL treat it as invalid, log a WARNING with the raw header value, and fall back to exponential backoff
17. IF rate-limit headers are malformed (non-numeric when expected, unparsable HTTP-date, or other parse errors) THEN the Agent SHALL log a WARNING with the raw header value and fall back to exponential backoff calculation
18. WHEN an explorer API fails repeatedly THEN the system SHALL implement a circuit-breaker with configurable failure threshold (default 5 failures) and cool-down window (default 5 minutes) to stop retries when the service is degraded
19. WHEN the circuit-breaker is OPEN THEN incoming requests SHALL fail fast with a deterministic CircuitBreakerOpenError (error code "CIRCUIT_OPEN", message including explorer name and remaining cool-down time) rather than being queued or deferred
20. WHEN the circuit-breaker is OPEN THEN the system SHALL NOT wait indefinitely; requests SHALL immediately return the CircuitBreakerOpenError until the cool-down window elapses
21. WHEN retry attempts are made while circuit-breaker is OPEN THEN those attempts SHALL be suppressed and SHALL NOT count toward the retry budget or cumulative delay
22. WHEN the circuit-breaker is OPEN and the cool-down window expires THEN the system SHALL automatically transition to HALF-OPEN state to test service recovery
23. WHEN the circuit-breaker is HALF-OPEN THEN the system SHALL transition to CLOSED after a configurable number of consecutive successful requests (halfOpenSuccessThreshold, default 1)
24. WHEN the circuit-breaker is HALF-OPEN and a request fails THEN the system SHALL immediately transition back to OPEN state (halfOpenFailureThreshold, default 1)
25. WHEN circuit-breaker state changes (CLOSED→OPEN, OPEN→HALF-OPEN, HALF-OPEN→CLOSED, HALF-OPEN→OPEN) THEN the system SHALL log the transition with explorer name, previous state, new state, failure count, configured thresholds, and timestamp
26. WHEN calculating retry delays THEN the system SHALL enforce a maximum retry budget measured as total cumulative delay time (not attempt count) that does not exceed 50% of the remaining overall run timeout
27. WHEN the retry budget is checked THEN the system SHALL evaluate before each retry attempt: if (cumulative_delay_so_far + next_delay) exceeds the budget, the retry SHALL be skipped
28. WHEN the retry budget is exhausted THEN the system SHALL immediately stop retrying for that request, log a WARNING with "retry_budget_exhausted" reason including cumulative delay and remaining timeout, and return the last error to the caller
29. WHEN calculating retry budget THEN the system SHALL NOT count circuit-breaker cool-down periods against the retry budget; cool-down is a separate mechanism that blocks all requests regardless of budget

### Requirement 4: Blockchain Data Input Validation

**User Story:** As a developer, I want all blockchain data validated against expected patterns, so that malformed or malicious data cannot cause injection attacks or processing errors.

#### Acceptance Criteria

1. WHEN parsing a blockchain address THEN the Agent SHALL validate it matches the pattern "^0x[a-fA-F0-9]{40}$"
2. WHEN parsing a transaction hash THEN the Agent SHALL validate it matches the pattern "^0x[a-fA-F0-9]{64}$"
3. WHEN parsing a numeric amount THEN the Agent SHALL validate it matches an unsigned decimal format: digits with optional single decimal point and up to 18 fractional digits (pattern: "^[0-9]+(\.[0-9]{1,18})?$"), SHALL NOT accept scientific notation or negative values, and SHALL validate the value does not exceed 2^256-1 to prevent overflow
4. IF any blockchain data field fails validation THEN the Agent SHALL skip that record and log a warning with the field name (not the invalid value)
5. WHEN storing blockchain data THEN the Agent SHALL normalize addresses to lowercase to prevent case-based duplicates
6. WHEN parsing timestamp values THEN the Agent SHALL validate they fall within reasonable bounds (not before blockchain genesis, not in future)
7. WHEN parsing block numbers THEN the Agent SHALL validate they are positive integers not exceeding current known block height plus a reasonable buffer (1000 blocks)
8. WHEN receiving an explorer API response THEN the Agent SHALL validate the response structure against a JSON schema defining required fields, expected types, and maximum nesting depth (default 10 levels)
9. IF an API response fails schema validation THEN the Agent SHALL skip processing that response, log a warning with the schema violation type and field path (not raw values), and continue with remaining requests
10. WHEN schema validation is configured THEN the system SHALL load JSON schemas from a configurable location (default: schemas/ directory) with separate schemas per explorer API endpoint
11. WHEN validating API responses THEN the Agent SHALL detect schema version by checking: (1) response header "X-Schema-Version" first, (2) JSON body path "meta.schemaVersion" second; If header is present but empty or the body version is present, check the body path; treat as 'unknown' only if both are absent or unparseable."
12. IF no version indicator is present in response THEN the Agent SHALL treat version as "unknown" and apply the configured fallback strategy with a logged WARNING
13. WHEN a schema version mismatch is detected THEN the Agent SHALL classify it by semantic level: major (breaking changes, reject response), minor (non-breaking additions, allow with warning), or patch (backward-compatible, allow silently)
14. WHEN logging a schema version mismatch THEN the Agent SHALL include: detected version, expected version, classification level, schema file path, and actionable guidance (e.g., "Update local schema at schemas/etherscan/tokentx.json or run schema-sync command; see https://docs.etherscan.io/api for API contract changes")
15. IF a required schema file is missing at startup THEN the system SHALL fail startup with a clear error message identifying the missing schema path
16. IF a schema file contains malformed JSON or invalid JSON Schema syntax THEN the system SHALL fail startup with a clear error message including the file path and parse error details
17. WHEN schema validation cannot be performed THEN the system SHALL use a configurable fallback strategy: "fail-closed" (default, reject unvalidated responses), "skip-validation" (allow responses without validation, log warning), or "permissive-default" (use minimal built-in schema)
18. WHEN "permissive-default" fallback is used THEN the system SHALL apply a minimal built-in schema that enforces only top-level required fields ("status", "result") and basic types (no deep structural constraints), log a WARNING, and increment a permissive_fallback_used metric
19. WHEN schema loading or validation fails THEN the system SHALL log detailed error information including schema path, error type, and timestamp, and increment a schema_validation_failures metric
20. WHEN schema files are updated at runtime THEN the system SHALL support atomic hot-reload: replace the complete schema set in a single swap so all schemas are consistent
21. WHEN hot-reload occurs THEN in-flight validations SHALL complete against the schema snapshot active at validation start; no re-validation mid-flight
22. IF hot-reload fails THEN the system SHALL log the error with details and retain the prior working schema snapshot; no partial updates shall be applied

### Requirement 5: Safe File Path Handling

**User Story:** As a security engineer, I want output file paths constrained to the configured output directory, so that path traversal attacks cannot write files to arbitrary locations.

#### Acceptance Criteria

1. WHEN constructing an output file path THEN the Agent SHALL resolve the path and verify it is within the configured output directory
2. IF a constructed path would escape the output directory THEN the system SHALL reject the path and raise a security error
3. WHEN creating output filenames THEN the Agent SHALL sanitize any user-provided or external data used in the filename
4. WHEN the output directory is configured THEN the system SHALL verify it exists and is writable at startup
5. WHEN writing output files THEN the Agent SHALL use atomic write operations to prevent partial file corruption

### Requirement 6: Collection Timeout Enforcement

**User Story:** As a system operator, I want individual collection operations to have enforced timeouts, so that hung connections or slow responses cannot block the system indefinitely.

#### Acceptance Criteria

1. WHEN fetching data from an explorer THEN the Agent SHALL enforce a per-stablecoin collection timeout (configurable, default 3 minutes)
2. WHEN the per-stablecoin timeout is exceeded THEN the Agent SHALL abort that collection and continue with remaining stablecoins
3. WHEN the overall run timeout approaches (within 60 seconds) THEN the Agent SHALL log progress and prepare for graceful termination
4. WHEN timeout configuration is loaded THEN the system SHALL validate that (number_of_stablecoins × number_of_explorers × per_stablecoin_timeout) does not exceed the overall run timeout
5. IF timeout validation fails THEN the system SHALL either adjust per-stablecoin timeout dynamically (overall_timeout / (stablecoins × explorers)) with a minimum of 60 seconds, or reject the configuration with a clear error message
6. WHEN the overall timeout is reached mid-collection THEN the Agent SHALL abort the current stablecoin collection, persist partial results with "partial" status, and log the interruption

### Requirement 7: Request Audit Trail (Optional)

**User Story:** As a security auditor, I want all outbound API requests logged with relevant metadata, so that I can investigate suspicious activity or debug issues.

#### Acceptance Criteria

1. WHEN the Agent makes an outbound request THEN the system SHALL log the target domain, endpoint path, and timestamp
2. WHEN an outbound request completes THEN the system SHALL log the response status code and response time
3. WHEN logging request details THEN the system SHALL exclude sensitive parameters (API keys, tokens)
4. WHEN a request fails THEN the system SHALL log the failure reason with correlation ID for tracing

### Requirement 8: Response Caching (Optional)

**User Story:** As a system operator, I want frequently-accessed explorer responses cached, so that repeated requests reduce external API exposure and improve performance.

#### Acceptance Criteria

1. WHEN a cacheable response is received THEN the Agent SHALL store it with a configurable TTL (default 5 minutes)
2. WHEN a cached response exists and is not expired THEN the Agent SHALL return the cached data without making an external request
3. WHEN cache storage exceeds configured limits THEN the system SHALL evict oldest entries first
4. WHEN caching is enabled THEN the system SHALL provide cache hit/miss metrics for monitoring
