"""
Unit tests for API response schema validation.

Tests the ResponseSchemaValidator class for correct behavior with
valid and invalid API responses.

Requirements: 4.8, 4.9
"""

from pathlib import Path
from pathlib import Path

import pytest

from core.security.schema_validator import (
    ResponseSchemaValidator,
    SchemaFallbackStrategy,
    SchemaLoadError,
    SchemaVersionClassification,
    ValidationResult,
)


class TestResponseSchemaValidator:
    """Unit tests for ResponseSchemaValidator."""

    @pytest.fixture
    def validator(self):
        """Create a validator with loaded schemas."""
        v = ResponseSchemaValidator(
            schema_directory=Path("schemas"),
            fallback_strategy=SchemaFallbackStrategy.FAIL_CLOSED,
        )
        v.load_schemas()
        return v

    @pytest.fixture
    def valid_tokentx_response(self):
        """Create a valid tokentx API response."""
        return {
            "status": "1",
            "message": "OK",
            "result": [
                {
                    "blockNumber": "12345678",
                    "timeStamp": "1638316800",
                    "hash": "0x" + "a" * 64,
                    "from": "0x" + "1" * 40,
                    "to": "0x" + "2" * 40,
                    "value": "1000000000000000000",
                    "contractAddress": "0x" + "3" * 40,
                }
            ],
        }

    @pytest.fixture
    def valid_tokenholderlist_response(self):
        """Create a valid tokenholderlist API response."""
        return {
            "status": "1",
            "message": "OK",
            "result": [
                {
                    "TokenHolderAddress": "0x" + "1" * 40,
                    "TokenHolderQuantity": "1000000000000000000",
                }
            ],
        }

    def test_valid_response_passes_validation(self, validator, valid_tokentx_response):
        """Test that a valid response passes validation.
        
        Requirements: 4.8
        """
        result = validator.validate(
            valid_tokentx_response, "etherscan", "tokentx"
        )
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.field_paths) == 0

    def test_valid_tokenholderlist_passes_validation(
        self, validator, valid_tokenholderlist_response
    ):
        """Test that a valid tokenholderlist response passes validation.
        
        Requirements: 4.8
        """
        result = validator.validate(
            valid_tokenholderlist_response, "etherscan", "tokenholderlist"
        )
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_missing_required_field_fails_validation(self, validator):
        """Test that missing required field fails validation.
        
        Requirements: 4.8, 4.9
        """
        response = {
            "message": "OK",
            "result": [],
            # Missing "status" field
        }
        result = validator.validate(response, "etherscan", "tokentx")
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        # Error should mention the missing field
        assert any("status" in err.lower() or "required" in err.lower() 
                   for err in result.errors)

    def test_incorrect_type_fails_validation(self, validator):
        """Test that incorrect field type fails validation.
        
        Requirements: 4.8, 4.9
        """
        response = {
            "status": 123,  # Should be string, not int
            "message": "OK",
            "result": [],
        }
        result = validator.validate(response, "etherscan", "tokentx")
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        # Error should mention type issue
        assert any("type" in err.lower() for err in result.errors)

    def test_excessive_nesting_depth_fails_validation(self, validator):
        """Test that excessive nesting depth fails validation.
        
        Requirements: 4.8 (max nesting depth 10 levels)
        """
        # Create deeply nested object (15 levels)
        response = {"status": "1", "message": "OK", "result": []}
        current = response
        for i in range(15):
            current["nested"] = {"level": i}
            current = current["nested"]
        
        result = validator.validate(response, "etherscan", "tokentx")
        
        assert result.is_valid is False
        assert result.nesting_depth_exceeded is True

    def test_error_messages_contain_field_paths_not_raw_values(self, validator):
        """Test that error messages contain field paths, not raw values.
        
        Requirements: 4.9
        """
        # Create response with invalid transfer
        response = {
            "status": "1",
            "message": "OK",
            "result": [
                {
                    "blockNumber": "12345678",
                    "timeStamp": "1638316800",
                    "hash": "invalid_hash",  # Invalid format
                    "from": "0x" + "1" * 40,
                    "to": "0x" + "2" * 40,
                    "value": "1000000000000000000",
                    "contractAddress": "0x" + "3" * 40,
                }
            ],
        }
        result = validator.validate(response, "etherscan", "tokentx")
        
        assert result.is_valid is False
        # Field paths should be present
        assert len(result.field_paths) > 0
        # Error messages should NOT contain the raw invalid value
        for error in result.errors:
            assert "invalid_hash" not in error

    def test_invalid_address_pattern_fails_validation(self, validator):
        """Test that invalid address pattern fails validation.
        
        Requirements: 4.8
        """
        response = {
            "status": "1",
            "message": "OK",
            "result": [
                {
                    "blockNumber": "12345678",
                    "timeStamp": "1638316800",
                    "hash": "0x" + "a" * 64,
                    "from": "not_an_address",  # Invalid
                    "to": "0x" + "2" * 40,
                    "value": "1000000000000000000",
                    "contractAddress": "0x" + "3" * 40,
                }
            ],
        }
        result = validator.validate(response, "etherscan", "tokentx")
        
        assert result.is_valid is False
        # The validation should fail due to pattern mismatch
        assert len(result.errors) > 0

    def test_empty_result_array_passes_validation(self, validator):
        """Test that empty result array passes validation.
        
        Requirements: 4.8
        """
        response = {
            "status": "1",
            "message": "OK",
            "result": [],
        }
        result = validator.validate(response, "etherscan", "tokentx")
        
        assert result.is_valid is True

    def test_string_result_passes_validation(self, validator):
        """Test that string result (error message) passes validation.
        
        Requirements: 4.8
        """
        response = {
            "status": "0",
            "message": "No transactions found",
            "result": "No transactions found",
        }
        result = validator.validate(response, "etherscan", "tokentx")
        
        assert result.is_valid is True

    def test_all_explorers_have_schemas(self, validator):
        """Test that all expected explorers have schemas loaded."""
        loaded = validator.loaded_schemas
        
        assert "etherscan" in loaded
        assert "bscscan" in loaded
        assert "polygonscan" in loaded
        
        for explorer in ["etherscan", "bscscan", "polygonscan"]:
            assert "tokentx" in loaded[explorer]
            assert "tokenholderlist" in loaded[explorer]


class TestSchemaVersionDetection:
    """Tests for schema version detection."""

    @pytest.fixture
    def validator(self):
        """Create a validator."""
        v = ResponseSchemaValidator(
            schema_directory=Path("schemas"),
            fallback_strategy=SchemaFallbackStrategy.FAIL_CLOSED,
        )
        v.load_schemas()
        return v

    def test_detect_version_from_header(self, validator):
        """Test version detection from X-Schema-Version header."""
        response = {"status": "1", "message": "OK", "result": []}
        headers = {"X-Schema-Version": "2.0.0"}
        
        version, source = validator.detect_response_version(response, headers)
        
        assert version == "2.0.0"
        assert source == "header"

    def test_detect_version_from_body(self, validator):
        """Test version detection from meta.schemaVersion in body."""
        response = {
            "status": "1",
            "message": "OK",
            "result": [],
            "meta": {"schemaVersion": "1.5.0"},
        }
        
        version, source = validator.detect_response_version(response)
        
        assert version == "1.5.0"
        assert source == "body"

    def test_detect_version_unknown(self, validator):
        """Test version detection returns unknown when not present."""
        response = {"status": "1", "message": "OK", "result": []}
        
        version, source = validator.detect_response_version(response)
        
        assert version == "unknown"
        assert source == "unknown"

    def test_header_takes_precedence_over_body(self, validator):
        """Test that header version takes precedence over body."""
        response = {
            "status": "1",
            "message": "OK",
            "result": [],
            "meta": {"schemaVersion": "1.0.0"},
        }
        headers = {"X-Schema-Version": "2.0.0"}
        
        version, source = validator.detect_response_version(response, headers)
        
        assert version == "2.0.0"
        assert source == "header"


class TestSchemaVersionClassification:
    """Tests for schema version mismatch classification."""

    @pytest.fixture
    def validator(self):
        """Create a validator."""
        return ResponseSchemaValidator(
            schema_directory=Path("schemas"),
            fallback_strategy=SchemaFallbackStrategy.FAIL_CLOSED,
        )

    def test_major_version_mismatch(self, validator):
        """Test major version mismatch classification."""
        classification = validator.classify_version_mismatch("2.0.0", "1.0.0")
        assert classification == SchemaVersionClassification.MAJOR

    def test_minor_version_mismatch(self, validator):
        """Test minor version mismatch classification."""
        classification = validator.classify_version_mismatch("1.1.0", "1.0.0")
        assert classification == SchemaVersionClassification.MINOR

    def test_patch_version_mismatch(self, validator):
        """Test patch version mismatch classification."""
        classification = validator.classify_version_mismatch("1.0.1", "1.0.0")
        assert classification == SchemaVersionClassification.PATCH

    def test_unknown_version_treated_as_minor(self, validator):
        """Test unknown version treated as minor."""
        classification = validator.classify_version_mismatch("unknown", "1.0.0")
        assert classification == SchemaVersionClassification.MINOR


class TestSchemaFallbackStrategies:
    """Tests for schema fallback strategies."""

    def test_fail_closed_rejects_unknown_schema(self):
        """Test fail-closed strategy rejects unknown schemas."""
        validator = ResponseSchemaValidator(
            schema_directory=Path("schemas"),
            fallback_strategy=SchemaFallbackStrategy.FAIL_CLOSED,
        )
        validator.load_schemas()
        
        response = {"status": "1", "message": "OK", "result": []}
        result = validator.validate(response, "unknown_explorer", "unknown_endpoint")
        
        assert result.is_valid is False
        assert "fail-closed" in result.errors[0].lower()

    def test_skip_validation_allows_unknown_schema(self):
        """Test skip-validation strategy allows unknown schemas."""
        validator = ResponseSchemaValidator(
            schema_directory=Path("schemas"),
            fallback_strategy=SchemaFallbackStrategy.SKIP_VALIDATION,
        )
        validator.load_schemas()
        
        response = {"status": "1", "message": "OK", "result": []}
        result = validator.validate(response, "unknown_explorer", "unknown_endpoint")
        
        assert result.is_valid is True


class TestSchemaLoading:
    """Tests for schema loading functionality."""

    def test_load_schemas_from_directory(self):
        """Test loading schemas from directory."""
        validator = ResponseSchemaValidator(
            schema_directory=Path("schemas"),
            fallback_strategy=SchemaFallbackStrategy.FAIL_CLOSED,
        )
        validator.load_schemas()
        
        loaded = validator.loaded_schemas
        assert len(loaded) > 0
        assert "etherscan" in loaded

    def test_missing_schema_directory_raises_error(self):
        """Test that missing schema directory raises error."""
        validator = ResponseSchemaValidator(
            schema_directory=Path("nonexistent_directory"),
            fallback_strategy=SchemaFallbackStrategy.FAIL_CLOSED,
        )
        
        with pytest.raises(SchemaLoadError):
            validator.load_schemas()

    def test_get_schema_version(self):
        """Test getting schema version."""
        validator = ResponseSchemaValidator(
            schema_directory=Path("schemas"),
            fallback_strategy=SchemaFallbackStrategy.FAIL_CLOSED,
        )
        validator.load_schemas()
        
        version = validator.get_schema_version("etherscan", "tokentx")
        assert version is not None
        assert version == "1.0.0"

    def test_validation_failures_counter(self):
        """Test validation failures counter increments."""
        validator = ResponseSchemaValidator(
            schema_directory=Path("schemas"),
            fallback_strategy=SchemaFallbackStrategy.FAIL_CLOSED,
        )
        validator.load_schemas()
        
        initial_count = validator.validation_failures
        
        # Trigger a validation failure
        response = {"message": "OK", "result": []}  # Missing status
        validator.validate(response, "etherscan", "tokentx")
        
        assert validator.validation_failures == initial_count + 1
