"""
Property-based tests for API response schema validation.

These tests use Hypothesis to verify correctness properties defined in the
design document.

**Feature: agent-security-hardening**
"""

from pathlib import Path
from typing import Any, Dict, List

from hypothesis import given, strategies as st, settings

from core.security.schema_validator import (
    ResponseSchemaValidator,
    SchemaFallbackStrategy,
    ValidationResult,
)


# =============================================================================
# Test Data Strategies
# =============================================================================


def valid_hex_address():
    """Generate valid 40-char hex address."""
    return st.text(
        alphabet="0123456789abcdefABCDEF",
        min_size=40,
        max_size=40,
    ).map(lambda x: f"0x{x}")


def valid_tx_hash():
    """Generate valid 64-char hex transaction hash."""
    return st.text(
        alphabet="0123456789abcdefABCDEF",
        min_size=64,
        max_size=64,
    ).map(lambda x: f"0x{x}")


def valid_numeric_string():
    """Generate valid numeric string."""
    return st.integers(min_value=0, max_value=10**18).map(str)


@st.composite
def valid_token_transfer(draw):
    """Generate a valid token transfer object matching schema."""
    return {
        "blockNumber": draw(valid_numeric_string()),
        "timeStamp": draw(valid_numeric_string()),
        "hash": draw(valid_tx_hash()),
        "from": draw(valid_hex_address()),
        "to": draw(valid_hex_address()),
        "value": draw(valid_numeric_string()),
        "contractAddress": draw(valid_hex_address()),
    }


@st.composite
def valid_tokentx_response(draw):
    """Generate a valid tokentx API response."""
    num_transfers = draw(st.integers(min_value=0, max_value=5))
    transfers = [draw(valid_token_transfer()) for _ in range(num_transfers)]
    return {
        "status": draw(st.sampled_from(["0", "1"])),
        "message": draw(st.text(min_size=1, max_size=50)),
        "result": transfers,
    }


@st.composite
def valid_token_holder(draw):
    """Generate a valid token holder object matching schema."""
    return {
        "TokenHolderAddress": draw(valid_hex_address()),
        "TokenHolderQuantity": draw(valid_numeric_string()),
    }


@st.composite
def valid_tokenholderlist_response(draw):
    """Generate a valid tokenholderlist API response."""
    num_holders = draw(st.integers(min_value=0, max_value=5))
    holders = [draw(valid_token_holder()) for _ in range(num_holders)]
    return {
        "status": draw(st.sampled_from(["0", "1"])),
        "message": draw(st.text(min_size=1, max_size=50)),
        "result": holders,
    }


@st.composite
def response_missing_required_field(draw):
    """Generate a response missing a required field."""
    response = draw(valid_tokentx_response())
    # Remove one of the required fields
    field_to_remove = draw(st.sampled_from(["status", "message", "result"]))
    del response[field_to_remove]
    return response, field_to_remove


@st.composite
def response_with_wrong_type(draw):
    """Generate a response with wrong type for a field."""
    response = draw(valid_tokentx_response())
    # Change status to wrong type (should be string)
    wrong_type = draw(st.sampled_from([
        123,
        True,
        ["list"],
        {"dict": "value"},
    ]))
    response["status"] = wrong_type
    return response


@st.composite
def transfer_with_invalid_address(draw):
    """Generate a transfer with invalid address format."""
    transfer = draw(valid_token_transfer())
    # Corrupt the address
    invalid_address = draw(st.sampled_from([
        "not_an_address",
        "0x" + "g" * 40,  # Invalid hex char
        "0x" + "a" * 39,  # Too short
        "0x" + "a" * 41,  # Too long
        "",
    ]))
    field = draw(st.sampled_from(["from", "to", "contractAddress"]))
    transfer[field] = invalid_address
    return transfer


@st.composite
def transfer_with_invalid_hash(draw):
    """Generate a transfer with invalid transaction hash."""
    transfer = draw(valid_token_transfer())
    invalid_hash = draw(st.sampled_from([
        "not_a_hash",
        "0x" + "g" * 64,  # Invalid hex char
        "0x" + "a" * 63,  # Too short
        "0x" + "a" * 65,  # Too long
        "",
    ]))
    transfer["hash"] = invalid_hash
    return transfer


@st.composite
def deeply_nested_object(draw, depth: int = 15):
    """Generate an object with excessive nesting depth."""
    obj: Dict[str, Any] = {"status": "1", "message": "OK", "result": []}
    current = obj
    for i in range(depth):
        current["nested"] = {"level": i}
        current = current["nested"]
    return obj


# =============================================================================
# Property Tests for Schema Validation
# =============================================================================


class TestSchemaValidationRejectsInvalidStructure:
    """
    Property tests for schema validation correctness.

    **Feature: agent-security-hardening, Property 11: Schema Validation
    Rejects Invalid Structure**

    For any API response with missing required fields, incorrect types,
    or nesting depth exceeding the limit, the schema validator SHALL
    return is_valid=False with error descriptions containing only field
    paths (not raw values).

    **Validates: Requirements 4.8, 4.9**
    """

    @classmethod
    def setup_class(cls):
        """Set up validator with loaded schemas."""
        cls.validator = ResponseSchemaValidator(
            # Resolve schemas directory relative to this test file
            # tests/test_schema_validator_properties.py -> ../schemas
            schema_directory=Path(__file__).resolve().parent.parent / "schemas",
            fallback_strategy=SchemaFallbackStrategy.FAIL_CLOSED,
        )
        cls.validator.load_schemas()

    @settings(max_examples=100)
    @given(response=valid_tokentx_response())
    def test_property_11_valid_tokentx_response_accepted(self, response):
        """
        **Feature: agent-security-hardening, Property 11: Schema Validation
        Rejects Invalid Structure**

        For any valid tokentx response matching the schema, validate()
        SHALL return is_valid=True.

        **Validates: Requirements 4.8, 4.9**
        """
        result = self.validator.validate(response, "etherscan", "tokentx")
        assert result.is_valid is True, (
            f"Valid response was rejected: {result.errors}"
        )

    @settings(max_examples=100)
    @given(response=valid_tokenholderlist_response())
    def test_property_11_valid_tokenholderlist_response_accepted(self, response):
        """
        **Feature: agent-security-hardening, Property 11: Schema Validation
        Rejects Invalid Structure**

        For any valid tokenholderlist response matching the schema,
        validate() SHALL return is_valid=True.

        **Validates: Requirements 4.8, 4.9**
        """
        result = self.validator.validate(
            response, "etherscan", "tokenholderlist"
        )
        assert result.is_valid is True, (
            f"Valid response was rejected: {result.errors}"
        )

    @settings(max_examples=100)
    @given(data=response_missing_required_field())
    def test_property_11_missing_required_field_rejected(self, data):
        """
        **Feature: agent-security-hardening, Property 11: Schema Validation
        Rejects Invalid Structure**

        For any response missing a required field, validate() SHALL
        return is_valid=False.

        **Validates: Requirements 4.8, 4.9**
        """
        response, missing_field = data
        result = self.validator.validate(response, "etherscan", "tokentx")

        assert result.is_valid is False, (
            f"Response missing '{missing_field}' was incorrectly accepted"
        )
        assert len(result.errors) > 0, "Expected error messages"

    @settings(max_examples=100)
    @given(response=response_with_wrong_type())
    def test_property_11_wrong_type_rejected(self, response):
        """
        **Feature: agent-security-hardening, Property 11: Schema Validation
        Rejects Invalid Structure**

        For any response with incorrect field types, validate() SHALL
        return is_valid=False.

        **Validates: Requirements 4.8, 4.9**
        """
        result = self.validator.validate(response, "etherscan", "tokentx")

        assert result.is_valid is False, (
            "Response with wrong type was incorrectly accepted"
        )
        assert len(result.errors) > 0, "Expected error messages"

    @settings(max_examples=100)
    @given(transfer=transfer_with_invalid_address())
    def test_property_11_invalid_address_pattern_rejected(self, transfer):
        """
        **Feature: agent-security-hardening, Property 11: Schema Validation
        Rejects Invalid Structure**

        For any transfer with invalid address format, validate() SHALL
        return is_valid=False when the transfer is in a response.

        **Validates: Requirements 4.8, 4.9**
        """
        response = {
            "status": "1",
            "message": "OK",
            "result": [transfer],
        }
        result = self.validator.validate(response, "etherscan", "tokentx")

        assert result.is_valid is False, (
            "Response with invalid address was incorrectly accepted"
        )

    @settings(max_examples=100)
    @given(transfer=transfer_with_invalid_hash())
    def test_property_11_invalid_hash_pattern_rejected(self, transfer):
        """
        **Feature: agent-security-hardening, Property 11: Schema Validation
        Rejects Invalid Structure**

        For any transfer with invalid transaction hash format, validate()
        SHALL return is_valid=False when the transfer is in a response.

        **Validates: Requirements 4.8, 4.9**
        """
        response = {
            "status": "1",
            "message": "OK",
            "result": [transfer],
        }
        result = self.validator.validate(response, "etherscan", "tokentx")

        assert result.is_valid is False, (
            "Response with invalid hash was incorrectly accepted"
        )

    @settings(max_examples=50)
    @given(obj=deeply_nested_object())
    def test_property_11_excessive_nesting_rejected(self, obj):
        """
        **Feature: agent-security-hardening, Property 11: Schema Validation
        Rejects Invalid Structure**

        For any response with nesting depth exceeding MAX_NESTING_DEPTH (10),
        validate() SHALL return is_valid=False with nesting_depth_exceeded=True.

        **Validates: Requirements 4.8, 4.9**
        """
        result = self.validator.validate(obj, "etherscan", "tokentx")

        assert result.is_valid is False, (
            "Response with excessive nesting was incorrectly accepted"
        )
        assert result.nesting_depth_exceeded is True, (
            "Expected nesting_depth_exceeded flag to be True"
        )

    @settings(max_examples=100)
    @given(response=valid_tokentx_response())
    def test_property_11_error_messages_contain_field_paths_not_values(
        self, response
    ):
        """
        **Feature: agent-security-hardening, Property 11: Schema Validation
        Rejects Invalid Structure**

        For any validation error, the error message SHALL contain field
        paths but NOT raw values from the response.

        **Validates: Requirements 4.8, 4.9**
        """
        # Create an invalid response by removing a required field
        response = response.copy()
        response.pop("status", None)

        result = self.validator.validate(response, "etherscan", "tokentx")

        # Check that errors don't contain raw values
        # Check that errors don't contain raw values
        raw_values = []
        
        # Collect potential raw values from response (strings > 5 chars)
        def collect_strings(obj):
            if isinstance(obj, str):
                if len(obj) > 5:
                    raw_values.append(obj)
            elif isinstance(obj, dict):
                for v in obj.values():
                    collect_strings(v)
            elif isinstance(obj, list):
                for item in obj:
                    collect_strings(item)
                    
        collect_strings(response)
        
        for error in result.errors:
            # Error messages should reference paths
            assert "Missing required" in error or "Invalid" in error or \
                   "Validation error" in error, (
                f"Unexpected error format: {error}"
            )
            
            # Error messages should NOT contain raw data
            for val in raw_values:
                assert val not in error, (
                    f"Sensitive data leaked in error message! Found value '{val}' "
                    f"in error: '{error}'"
                )

    @settings(max_examples=100)
    @given(response=valid_tokentx_response())
    def test_property_11_validation_is_deterministic(self, response):
        """
        **Feature: agent-security-hardening, Property 11: Schema Validation
        Rejects Invalid Structure**

        For any response, calling validate() multiple times SHALL return
        the same result.

        **Validates: Requirements 4.8, 4.9**
        """
        result1 = self.validator.validate(response, "etherscan", "tokentx")
        result2 = self.validator.validate(response, "etherscan", "tokentx")
        result3 = self.validator.validate(response, "etherscan", "tokentx")

        assert result1.is_valid == result2.is_valid == result3.is_valid, (
            "Non-deterministic validation results"
        )

    @settings(max_examples=100)
    @given(response=valid_tokentx_response())
    def test_property_11_all_explorers_use_same_schema_format(self, response):
        """
        **Feature: agent-security-hardening, Property 11: Schema Validation
        Rejects Invalid Structure**

        For any valid response, validation against all explorers with
        the same endpoint SHALL produce consistent results (all accept
        or all reject).

        **Validates: Requirements 4.8, 4.9**
        """
        explorers = ["etherscan", "bscscan", "polygonscan"]
        results = [
            self.validator.validate(response, explorer, "tokentx")
            for explorer in explorers
        ]

        # All should have the same validity
        validities = [r.is_valid for r in results]
        assert all(v == validities[0] for v in validities), (
            f"Inconsistent validation across explorers: {validities}"
        )
