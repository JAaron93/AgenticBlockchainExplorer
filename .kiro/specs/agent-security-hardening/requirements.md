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
