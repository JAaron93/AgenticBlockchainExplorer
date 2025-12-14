"""API Response Schema Validation for the stablecoin explorer.

This module provides JSON schema validation for explorer API responses
to ensure data integrity and detect malformed or malicious responses.

Requirements: 4.8, 4.9, 4.10, 4.11
"""

import json
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import jsonschema
    from jsonschema import Draft7Validator, ValidationError as JsonSchemaValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    JsonSchemaValidationError = Exception  # type: ignore

logger = logging.getLogger(__name__)


class SchemaFallbackStrategy(Enum):
    """Fallback strategy when schema validation cannot be performed.
    
    Requirements: 4.17
    """
    FAIL_CLOSED = "fail-closed"  # Reject unvalidated responses (default)
    SKIP_VALIDATION = "skip-validation"  # Allow without validation, log warning
    PERMISSIVE_DEFAULT = "permissive-default"  # Use minimal built-in schema


class SchemaVersionClassification(Enum):
    """Classification of schema version mismatches.
    
    Requirements: 4.13
    """
    MAJOR = "major"  # Breaking changes, reject response
    MINOR = "minor"  # Non-breaking additions, allow with warning
    PATCH = "patch"  # Backward-compatible, allow silently


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""
    pass


class SchemaLoadError(Exception):
    """Raised when schema loading fails."""
    pass


@dataclass
class ValidationResult:
    """Result of schema validation.
    
    Attributes:
        is_valid: Whether the response passed validation
        errors: List of error descriptions (no raw values)
        field_paths: Paths to invalid fields (e.g., "result[0].hash")
        schema_version: Version of the schema used for validation
        nesting_depth_exceeded: Whether nesting depth limit was exceeded
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    field_paths: List[str] = field(default_factory=list)
    schema_version: Optional[str] = None
    nesting_depth_exceeded: bool = False


# Minimal built-in schema for permissive-default fallback
MINIMAL_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["status", "result"],
    "properties": {
        "status": {"type": "string"},
        "message": {"type": "string"},
        "result": {}
    }
}


class ResponseSchemaValidator:
    """Validates explorer API responses against JSON schemas.
    
    This class loads JSON schemas from a directory structure and validates
    API responses against them. It supports fallback strategies when
    validation cannot be performed.
    
    Requirements:
    - 4.8: Validate response structure against JSON schema
    - 4.9: Skip invalid responses and log warnings with field paths
    - 4.10: Load schemas from configurable location
    - 4.11: Detect schema version from response
    """
    
    MAX_NESTING_DEPTH = 10
    
    def __init__(
        self,
        schema_directory: Path = Path("schemas"),
        fallback_strategy: SchemaFallbackStrategy = SchemaFallbackStrategy.FAIL_CLOSED,
        enable_hot_reload: bool = False,
    ):
        """Initialize the schema validator.
        
        Args:
            schema_directory: Path to directory containing JSON schemas
            fallback_strategy: Strategy when validation cannot be performed
            enable_hot_reload: Whether to support atomic schema hot-reload
            
        Raises:
            SchemaLoadError: If required schemas are missing or invalid
        """
        self._schema_directory = Path(schema_directory)
        self._fallback_strategy = fallback_strategy
        self._enable_hot_reload = enable_hot_reload
        
        # Schema storage: {explorer: {endpoint: schema_dict}}
        self._schemas: Dict[str, Dict[str, dict]] = {}
        self._schema_versions: Dict[str, Dict[str, str]] = {}
        self._schema_load_errors: List[str] = []
        
        # Lock for thread-safe hot-reload
        self._lock = threading.RLock()
        
        # Metrics counters
        self._metrics_lock = threading.Lock()
        self._validation_failures = 0
        self._permissive_fallback_used = 0
        
        if not JSONSCHEMA_AVAILABLE:
            logger.warning(
                "jsonschema library not available, schema validation disabled"
            )
    
    def load_schemas(self) -> None:
        """Load all JSON schemas from the schema directory.
        
        Expected directory structure:
        schemas/
        ├── etherscan/
        │   ├── tokentx.json
        │   └── tokenholderlist.json
        ├── bscscan/
        │   └── ...
        └── polygonscan/
            └── ...
            
        Raises:
            SchemaLoadError: If required schema files are missing or invalid
            
        Requirements: 4.10, 4.15, 4.16
        """
        with self._lock:
            new_schemas: Dict[str, Dict[str, dict]] = {}
            new_versions: Dict[str, Dict[str, str]] = {}
            errors: List[str] = []
            
            if not self._schema_directory.exists():
                error_msg = f"Schema directory not found: {self._schema_directory}"
                logger.error(error_msg)
                raise SchemaLoadError(error_msg)
            
            # Iterate through explorer directories
            for explorer_dir in self._schema_directory.iterdir():
                if not explorer_dir.is_dir():
                    continue
                    
                explorer_name = explorer_dir.name.lower()
                new_schemas[explorer_name] = {}
                new_versions[explorer_name] = {}
                
                # Load each schema file
                for schema_file in explorer_dir.glob("*.json"):
                    endpoint_name = schema_file.stem.lower()
                    
                    try:
                        schema = self._load_schema_file(schema_file)
                        new_schemas[explorer_name][endpoint_name] = schema
                        
                        # Extract version from schema
                        version = schema.get("version", "unknown")
                        new_versions[explorer_name][endpoint_name] = version
                        
                        logger.debug(
                            f"Loaded schema: {explorer_name}/{endpoint_name} "
                            f"(version {version})"
                        )
                    except (json.JSONDecodeError, SchemaLoadError) as e:
                        error_msg = f"Failed to load schema {schema_file}: {e}"
                        errors.append(error_msg)
                        logger.error(error_msg)
            
            if errors:
                self._schema_load_errors = errors
                raise SchemaLoadError(
                    f"Schema loading failed with {len(errors)} errors: "
                    f"{'; '.join(errors[:3])}"
                )
            
            # Atomic swap of schemas (for hot-reload safety)
            self._schemas = new_schemas
            self._schema_versions = new_versions
            self._schema_load_errors = []
            
            logger.info(
                f"Loaded {sum(len(s) for s in new_schemas.values())} schemas "
                f"from {len(new_schemas)} explorers"
            )
    
    def _load_schema_file(self, path: Path) -> dict:
        """Load and validate a single schema file.
        
        Args:
            path: Path to the JSON schema file
            
        Returns:
            Parsed schema dictionary
            
        Raises:
            SchemaLoadError: If file is missing or contains invalid JSON/schema
            
        Requirements: 4.15, 4.16
        """
        if not path.exists():
            raise SchemaLoadError(f"Schema file not found: {path}")
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                schema = json.load(f)
        except json.JSONDecodeError as e:
            raise SchemaLoadError(
                f"Invalid JSON in schema file {path}: {e}"
            )
        
        # Validate the schema itself is valid JSON Schema
        if JSONSCHEMA_AVAILABLE:
            try:
                Draft7Validator.check_schema(schema)
            except jsonschema.SchemaError as e:
                raise SchemaLoadError(
                    f"Invalid JSON Schema syntax in {path}: {e.message}"
                )
        
        return schema
    
    def validate(
        self,
        response: dict,
        explorer: str,
        endpoint: str,
    ) -> ValidationResult:
        """Validate response against schema for explorer/endpoint.
        
        Args:
            response: The API response dictionary
            explorer: Explorer name (e.g., "etherscan")
            endpoint: API endpoint (e.g., "tokentx")
            
        Returns:
            ValidationResult with is_valid, errors, and field_paths
            
        Requirements: 4.8, 4.9
        """
        explorer = explorer.lower()
        endpoint = endpoint.lower()
        
        # Check nesting depth first
        if not self._check_nesting_depth(response):
            with self._metrics_lock:
                self._validation_failures += 1
            return ValidationResult(
                is_valid=False,
                errors=["Response nesting depth exceeds maximum allowed"],
                field_paths=["<root>"],
                nesting_depth_exceeded=True,
            )
        
        # Get schema and version atomically
        with self._lock:
            schema = self._get_schema(explorer, endpoint)
            schema_version = self._schema_versions.get(explorer, {}).get(
                endpoint, "unknown"
            )

        if schema is None:
            return self._handle_missing_schema(explorer, endpoint)
        
        # Perform validation
        if not JSONSCHEMA_AVAILABLE:
            logger.warning(
                "jsonschema not available, skipping validation"
            )
            return ValidationResult(
                is_valid=True,
                schema_version=schema_version,
            )
        
        return self._validate_against_schema(
            response, schema, schema_version
        )
    
    def _validate_against_schema(
        self,
        response: dict,
        schema: dict,
        schema_version: str,
    ) -> ValidationResult:
        """Perform JSON Schema validation.
        
        Args:
            response: Response to validate
            schema: JSON Schema to validate against
            schema_version: Version string for result
            
        Returns:
            ValidationResult
        """
        validator = Draft7Validator(schema)
        errors: List[str] = []
        field_paths: List[str] = []
        
        for error in validator.iter_errors(response):
            # Build field path from error path
            path = self._build_field_path(error.absolute_path)
            field_paths.append(path)
            
            # Create error message without raw values
            error_msg = self._sanitize_error_message(error, path)
            errors.append(error_msg)
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            with self._metrics_lock:
                self._validation_failures += 1
            logger.warning(
                f"Schema validation failed: {len(errors)} errors at "
                f"paths: {', '.join(field_paths[:5])}"
            )
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            field_paths=field_paths,
            schema_version=schema_version,
        )
    
    def _build_field_path(self, path) -> str:
        """Build a human-readable field path from jsonschema path.
        
        Args:
            path: Deque of path elements from jsonschema
            
        Returns:
            String like "result[0].hash"
        """
        if not path:
            return "<root>"
        
        parts = []
        for element in path:
            if isinstance(element, int):
                parts.append(f"[{element}]")
            else:
                if parts:
                    parts.append(f".{element}")
                else:
                    parts.append(str(element))
        
        return "".join(parts)
    
    def _sanitize_error_message(
        self,
        error: JsonSchemaValidationError,
        path: str,
    ) -> str:
        """Create error message without exposing raw values.
        
        Args:
            error: The jsonschema validation error
            path: The field path
            
        Returns:
            Sanitized error message
            
        Requirements: 4.9 (log warnings with field paths only)
        """
        # Map common validation error types to safe messages
        validator = error.validator
        
        if validator == "required":
            missing = error.validator_value
            if isinstance(missing, list):
                return f"Missing required field(s) at {path}"
            return f"Missing required field at {path}"
        
        if validator == "type":
            expected = error.validator_value
            return f"Invalid type at {path}: expected {expected}"
        
        if validator == "pattern":
            return f"Invalid format at {path}: does not match expected pattern"
        
        if validator == "enum":
            return f"Invalid value at {path}: not in allowed values"
        
        if validator == "maxItems":
            return f"Array at {path} exceeds maximum allowed items"
        
        if validator == "minItems":
            return f"Array at {path} has fewer than minimum required items"
        
        if validator == "additionalProperties":
            return f"Unexpected additional property at {path}"
        
        if validator == "oneOf":
            return f"Value at {path} does not match any allowed schema"
        
        # Generic fallback - don't include the actual value
        return f"Validation error at {path}: {validator} constraint violated"
    
    def _check_nesting_depth(self, obj: Any) -> bool:
        """Check if object nesting exceeds MAX_NESTING_DEPTH.
        
        Args:
            obj: Object to check
            
        Returns:
            True if within limits, False if exceeded
            
        Requirements: 4.8 (max nesting depth 10 levels)
        """
        stack = [(obj, 0)]
        
        while stack:
            current_obj, depth = stack.pop()
            
            if depth > self.MAX_NESTING_DEPTH:
                return False
            
            if isinstance(current_obj, dict):
                for value in current_obj.values():
                    stack.append((value, depth + 1))
            elif isinstance(current_obj, list):
                for item in current_obj:
                    stack.append((item, depth + 1))
        
        return True
    
    def _get_schema(
        self,
        explorer: str,
        endpoint: str,
    ) -> Optional[dict]:
        """Get schema for explorer/endpoint combination.
        
        Args:
            explorer: Explorer name
            endpoint: Endpoint name
            
        Returns:
            Schema dict or None if not found
        """
        explorer_schemas = self._schemas.get(explorer)
        if explorer_schemas is None:
            return None
        return explorer_schemas.get(endpoint)
    
    def _handle_missing_schema(
        self,
        explorer: str,
        endpoint: str,
    ) -> ValidationResult:
        """Handle case when schema is not found.
        
        Args:
            explorer: Explorer name
            endpoint: Endpoint name
            
        Returns:
            ValidationResult based on fallback strategy
            
        Requirements: 4.17, 4.18
        """
        if self._fallback_strategy == SchemaFallbackStrategy.FAIL_CLOSED:
            with self._metrics_lock:
                self._validation_failures += 1
            return ValidationResult(
                is_valid=False,
                errors=[
                    f"No schema found for {explorer}/{endpoint} "
                    f"and fallback strategy is fail-closed"
                ],
                field_paths=[],
            )
        
        if self._fallback_strategy == SchemaFallbackStrategy.SKIP_VALIDATION:
            logger.warning(
                f"No schema for {explorer}/{endpoint}, skipping validation"
            )
            return ValidationResult(
                is_valid=True,
                errors=[],
                field_paths=[],
            )
        
        # PERMISSIVE_DEFAULT - use minimal schema
        with self._metrics_lock:
            self._permissive_fallback_used += 1
        logger.warning(
            f"No schema for {explorer}/{endpoint}, using permissive default"
        )
        return self._validate_against_schema(
            {}, MINIMAL_SCHEMA, "permissive-default"
        )
    
    def get_schema_version(
        self,
        explorer: str,
        endpoint: str,
    ) -> Optional[str]:
        """Get schema version for logging/debugging.
        
        Args:
            explorer: Explorer name
            endpoint: Endpoint name
            
        Returns:
            Version string or None if schema not found
        """
        explorer = explorer.lower()
        endpoint = endpoint.lower()
        
        with self._lock:
            return self._schema_versions.get(explorer, {}).get(endpoint)
    
    def detect_response_version(
        self,
        response: dict,
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, str]:
        """Detect schema version from response.
        
        Checks:
        1. Response header "X-Schema-Version" first
        2. JSON body path "meta.schemaVersion" second
        3. Returns "unknown" if both are absent
        
        Args:
            response: The API response dictionary
            headers: Optional response headers
            
        Returns:
            Tuple of (version, source) where source is "header", "body", 
            or "unknown"
            
        Requirements: 4.11, 4.12
        """
        # Check header first
        if headers:
            header_version = headers.get("X-Schema-Version", "").strip()
            if header_version:
                return (header_version, "header")
        
        # Check body path meta.schemaVersion
        if isinstance(response, dict):
            meta = response.get("meta")
            if isinstance(meta, dict):
                body_version = meta.get("schemaVersion")
                if body_version and isinstance(body_version, str):
                    return (body_version.strip(), "body")
        
        return ("unknown", "unknown")
    
    def classify_version_mismatch(
        self,
        detected_version: str,
        expected_version: str,
    ) -> SchemaVersionClassification:
        """Classify a schema version mismatch by semantic level.
        
        Args:
            detected_version: Version detected from response
            expected_version: Version expected by local schema
            
        Returns:
            Classification level (MAJOR, MINOR, PATCH)
            
        Requirements: 4.13
        """
        if detected_version == "unknown" or expected_version == "unknown":
            return SchemaVersionClassification.MINOR
        
        try:
            detected_parts = detected_version.split(".")
            expected_parts = expected_version.split(".")
            
            # Compare major version
            if len(detected_parts) >= 1 and len(expected_parts) >= 1:
                if detected_parts[0] != expected_parts[0]:
                    return SchemaVersionClassification.MAJOR
            
            # Compare minor version
            if len(detected_parts) >= 2 and len(expected_parts) >= 2:
                if detected_parts[1] != expected_parts[1]:
                    return SchemaVersionClassification.MINOR
            
            # Patch or identical
            return SchemaVersionClassification.PATCH
            
        except (ValueError, IndexError):
            # Can't parse versions, treat as minor
            return SchemaVersionClassification.MINOR
    
    def hot_reload(self) -> bool:
        """Atomically reload all schemas from disk.
        
        Returns:
            True if reload succeeded, False otherwise
            
        Requirements: 4.20, 4.21, 4.22
        """
        if not self._enable_hot_reload:
            logger.warning("Hot reload not enabled")
            return False
        
        try:
            self.load_schemas()
            logger.info("Schema hot-reload completed successfully")
            return True
        except SchemaLoadError as e:
            logger.error(f"Schema hot-reload failed: {e}")
            # Keep existing schemas per requirement 4.22
            return False
    
    @property
    def validation_failures(self) -> int:
        """Get count of validation failures."""
        with self._metrics_lock:
            return self._validation_failures
    
    @property
    def permissive_fallback_used(self) -> int:
        """Get count of permissive fallback uses."""
        with self._metrics_lock:
            return self._permissive_fallback_used
    
    @property
    def loaded_schemas(self) -> Dict[str, List[str]]:
        """Get dictionary of loaded schemas by explorer."""
        with self._lock:
            return {
                explorer: list(endpoints.keys())
                for explorer, endpoints in self._schemas.items()
            }
