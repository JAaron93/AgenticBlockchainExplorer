"""
Property-based tests for blockchain data validation.

These tests use Hypothesis to verify correctness properties defined in the design document.

**Feature: agent-security-hardening**
"""

import re

from hypothesis import given, strategies as st, settings

from core.security.blockchain_validator import BlockchainDataValidator


# =============================================================================
# Test Data Strategies for Address Validation
# =============================================================================


def valid_hex_chars():
    """Generate valid hexadecimal characters (0-9, a-f, A-F)."""
    return "0123456789abcdefABCDEF"


def valid_address_body():
    """Generate exactly 40 valid hex characters for address body."""
    return st.text(
        alphabet=valid_hex_chars(),
        min_size=40,
        max_size=40,
    )


@st.composite
def valid_ethereum_address(draw):
    """Generate a valid Ethereum-style address matching ^0x[a-fA-F0-9]{40}$."""
    body = draw(valid_address_body())
    return f"0x{body}"


@st.composite
def invalid_address_wrong_prefix(draw):
    """Generate an address with wrong prefix (not 0x)."""
    body = draw(valid_address_body())
    # Use various invalid prefixes
    prefix = draw(st.sampled_from([
        "0X",  # uppercase X
        "1x",  # wrong first char
        "x",   # missing 0
        "",    # no prefix
        "0",   # incomplete prefix
        "00",  # wrong prefix
        "0y",  # wrong second char
    ]))
    return f"{prefix}{body}"


@st.composite
def invalid_address_wrong_length(draw):
    """Generate an address with wrong body length (not 40 chars)."""
    # Generate body with length != 40
    length = draw(st.integers(min_value=0, max_value=100).filter(lambda x: x != 40))
    body = draw(st.text(alphabet=valid_hex_chars(), min_size=length, max_size=length))
    return f"0x{body}"


@st.composite
def invalid_address_non_hex_chars(draw):
    """Generate an address with non-hex characters in body."""
    # Generate a valid-length body but with at least one non-hex char
    non_hex_chars = "ghijklmnopqrstuvwxyzGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+-=[]{}|;':\",./<>? "
    
    # Generate position for non-hex char
    position = draw(st.integers(min_value=0, max_value=39))
    
    # Generate the body with one non-hex char
    hex_chars = valid_hex_chars()
    body_chars = [draw(st.sampled_from(hex_chars)) for _ in range(40)]
    body_chars[position] = draw(st.sampled_from(non_hex_chars))
    body = "".join(body_chars)
    
    return f"0x{body}"


def non_string_values():
    """Generate non-string values that should fail validation."""
    return st.one_of(
        st.integers(),
        st.floats(allow_nan=False),
        st.booleans(),
        st.none(),
        st.lists(st.text(), max_size=3),
        st.dictionaries(st.text(), st.text(), max_size=3),
    )


# =============================================================================
# Property Tests for Address Validation
# =============================================================================


class TestAddressValidationCorrectness:
    """
    Property tests for address validation correctness.
    
    **Feature: agent-security-hardening, Property 5: Address Validation Correctness**
    
    For any string, the address validator SHALL accept it if and only if
    it matches the pattern "^0x[a-fA-F0-9]{40}$".
    
    **Validates: Requirements 4.1**
    """

    @settings(max_examples=100)
    @given(address=valid_ethereum_address())
    def test_property_5_valid_addresses_accepted(self, address):
        """
        **Feature: agent-security-hardening, Property 5: Address Validation Correctness**
        
        For any string matching ^0x[a-fA-F0-9]{40}$, validate_address()
        SHALL return True.
        
        **Validates: Requirements 4.1**
        """
        validator = BlockchainDataValidator()
        
        # Valid addresses should be accepted
        assert validator.validate_address(address) is True, (
            f"Valid address '{address}' was rejected"
        )

    @settings(max_examples=100)
    @given(address=invalid_address_wrong_prefix())
    def test_property_5_wrong_prefix_rejected(self, address):
        """
        **Feature: agent-security-hardening, Property 5: Address Validation Correctness**
        
        For any string with a prefix other than "0x", validate_address()
        SHALL return False.
        
        **Validates: Requirements 4.1**
        """
        validator = BlockchainDataValidator()
        
        # Addresses with wrong prefix should be rejected
        assert validator.validate_address(address) is False, (
            f"Address with wrong prefix '{address}' was incorrectly accepted"
        )

    @settings(max_examples=100)
    @given(address=invalid_address_wrong_length())
    def test_property_5_wrong_length_rejected(self, address):
        """
        **Feature: agent-security-hardening, Property 5: Address Validation Correctness**
        
        For any string with body length != 40 hex characters,
        validate_address() SHALL return False.
        
        **Validates: Requirements 4.1**
        """
        validator = BlockchainDataValidator()
        
        # Addresses with wrong length should be rejected
        assert validator.validate_address(address) is False, (
            f"Address with wrong length '{address}' (len={len(address)}) was incorrectly accepted"
        )

    @settings(max_examples=100)
    @given(address=invalid_address_non_hex_chars())
    def test_property_5_non_hex_chars_rejected(self, address):
        """
        **Feature: agent-security-hardening, Property 5: Address Validation Correctness**
        
        For any string containing non-hexadecimal characters in the body,
        validate_address() SHALL return False.
        
        **Validates: Requirements 4.1**
        """
        validator = BlockchainDataValidator()
        
        # Addresses with non-hex chars should be rejected
        assert validator.validate_address(address) is False, (
            f"Address with non-hex chars '{address}' was incorrectly accepted"
        )

    @settings(max_examples=100)
    @given(value=non_string_values())
    def test_property_5_non_string_values_rejected(self, value):
        """
        **Feature: agent-security-hardening, Property 5: Address Validation Correctness**
        
        For any non-string value, validate_address() SHALL return False.
        
        **Validates: Requirements 4.1**
        """
        validator = BlockchainDataValidator()
        
        # Non-string values should be rejected
        assert validator.validate_address(value) is False, (
            f"Non-string value {type(value).__name__} was incorrectly accepted"
        )

    @settings(max_examples=100)
    @given(address=st.text(max_size=100))
    def test_property_5_arbitrary_strings_validation_consistency(self, address):
        """
        **Feature: agent-security-hardening, Property 5: Address Validation Correctness**
        
        For any arbitrary string, validate_address() SHALL return True
        if and only if the string matches ^0x[a-fA-F0-9]{40}$.
        
        This is the core property test that verifies the validator
        correctly implements the specification.
        
        **Validates: Requirements 4.1**
        """
        import re
        
        validator = BlockchainDataValidator()
        pattern = re.compile(r"^0x[a-fA-F0-9]{40}$")
        
        # The validator result should match the regex pattern
        expected = bool(pattern.match(address))
        actual = validator.validate_address(address)
        
        assert actual == expected, (
            f"Validation mismatch for '{address}': "
            f"expected {expected}, got {actual}"
        )

    @settings(max_examples=100)
    @given(address=valid_ethereum_address())
    def test_property_5_validation_is_deterministic(self, address):
        """
        **Feature: agent-security-hardening, Property 5: Address Validation Correctness**
        
        For any address, calling validate_address() multiple times
        SHALL return the same result.
        
        **Validates: Requirements 4.1**
        """
        validator = BlockchainDataValidator()
        
        # Multiple calls should return the same result
        result1 = validator.validate_address(address)
        result2 = validator.validate_address(address)
        result3 = validator.validate_address(address)
        
        assert result1 == result2 == result3, (
            f"Non-deterministic validation for '{address}': "
            f"got {result1}, {result2}, {result3}"
        )

    @settings(max_examples=100)
    @given(
        lower_body=st.text(alphabet="0123456789abcdef", min_size=40, max_size=40),
    )
    def test_property_5_case_insensitive_hex_accepted(self, lower_body):
        """
        **Feature: agent-security-hardening, Property 5: Address Validation Correctness**
        
        For any valid address, both lowercase and uppercase hex characters
        SHALL be accepted (the pattern allows [a-fA-F0-9]).
        
        **Validates: Requirements 4.1**
        """
        validator = BlockchainDataValidator()
        
        # Lowercase address
        lower_address = f"0x{lower_body}"
        # Uppercase address
        upper_address = f"0x{lower_body.upper()}"
        # Mixed case address
        mixed_body = "".join(
            c.upper() if i % 2 == 0 else c.lower()
            for i, c in enumerate(lower_body)
        )
        mixed_address = f"0x{mixed_body}"
        
        # All case variants should be valid
        assert validator.validate_address(lower_address) is True, (
            f"Lowercase address '{lower_address}' was rejected"
        )
        assert validator.validate_address(upper_address) is True, (
            f"Uppercase address '{upper_address}' was rejected"
        )
        assert validator.validate_address(mixed_address) is True, (
            f"Mixed case address '{mixed_address}' was rejected"
        )


# =============================================================================
# Test Data Strategies for Transaction Hash Validation
# =============================================================================


def valid_tx_hash_body():
    """Generate exactly 64 valid hex characters for tx hash body."""
    return st.text(
        alphabet=valid_hex_chars(),
        min_size=64,
        max_size=64,
    )


@st.composite
def valid_transaction_hash(draw):
    """Generate a valid transaction hash matching ^0x[a-fA-F0-9]{64}$."""
    body = draw(valid_tx_hash_body())
    return f"0x{body}"


@st.composite
def invalid_tx_hash_wrong_prefix(draw):
    """Generate a tx hash with wrong prefix (not 0x)."""
    body = draw(valid_tx_hash_body())
    prefix = draw(st.sampled_from([
        "0X",  # uppercase X
        "1x",  # wrong first char
        "x",   # missing 0
        "",    # no prefix
        "0",   # incomplete prefix
        "00",  # wrong prefix
        "0y",  # wrong second char
    ]))
    return f"{prefix}{body}"


@st.composite
def invalid_tx_hash_wrong_length(draw):
    """Generate a tx hash with wrong body length (not 64 chars)."""
    length = draw(
        st.integers(min_value=0, max_value=128).filter(lambda x: x != 64)
    )
    body = draw(st.text(
        alphabet=valid_hex_chars(),
        min_size=length,
        max_size=length
    ))
    return f"0x{body}"


@st.composite
def invalid_tx_hash_non_hex_chars(draw):
    """Generate a tx hash with non-hex characters in body."""
    non_hex = "ghijklmnopqrstuvwxyzGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+-="
    position = draw(st.integers(min_value=0, max_value=63))
    hex_chars = valid_hex_chars()
    body_chars = [draw(st.sampled_from(hex_chars)) for _ in range(64)]
    body_chars[position] = draw(st.sampled_from(non_hex))
    body = "".join(body_chars)
    return f"0x{body}"


# =============================================================================
# Property Tests for Transaction Hash Validation
# =============================================================================


class TestTransactionHashValidationCorrectness:
    """
    Property tests for transaction hash validation correctness.

    **Feature: agent-security-hardening, Property 6: Transaction Hash
    Validation Correctness**

    For any string, the transaction hash validator SHALL accept it if and
    only if it matches the pattern "^0x[a-fA-F0-9]{64}$".

    **Validates: Requirements 4.2**
    """

    @settings(max_examples=100)
    @given(tx_hash=valid_transaction_hash())
    def test_property_6_valid_tx_hashes_accepted(self, tx_hash):
        """
        **Feature: agent-security-hardening, Property 6: Transaction Hash
        Validation Correctness**

        For any string matching ^0x[a-fA-F0-9]{64}$, validate_tx_hash()
        SHALL return True.

        **Validates: Requirements 4.2**
        """
        validator = BlockchainDataValidator()
        assert validator.validate_tx_hash(tx_hash) is True, (
            f"Valid tx hash '{tx_hash}' was rejected"
        )

    @settings(max_examples=100)
    @given(tx_hash=invalid_tx_hash_wrong_prefix())
    def test_property_6_wrong_prefix_rejected(self, tx_hash):
        """
        **Feature: agent-security-hardening, Property 6: Transaction Hash
        Validation Correctness**

        For any string with a prefix other than "0x", validate_tx_hash()
        SHALL return False.

        **Validates: Requirements 4.2**
        """
        validator = BlockchainDataValidator()
        assert validator.validate_tx_hash(tx_hash) is False, (
            f"Tx hash with wrong prefix '{tx_hash}' was incorrectly accepted"
        )

    @settings(max_examples=100)
    @given(tx_hash=invalid_tx_hash_wrong_length())
    def test_property_6_wrong_length_rejected(self, tx_hash):
        """
        **Feature: agent-security-hardening, Property 6: Transaction Hash
        Validation Correctness**

        For any string with body length != 64 hex characters,
        validate_tx_hash() SHALL return False.

        **Validates: Requirements 4.2**
        """
        validator = BlockchainDataValidator()
        assert validator.validate_tx_hash(tx_hash) is False, (
            f"Tx hash with wrong length '{tx_hash}' was incorrectly accepted"
        )

    @settings(max_examples=100)
    @given(tx_hash=invalid_tx_hash_non_hex_chars())
    def test_property_6_non_hex_chars_rejected(self, tx_hash):
        """
        **Feature: agent-security-hardening, Property 6: Transaction Hash
        Validation Correctness**

        For any string containing non-hexadecimal characters in the body,
        validate_tx_hash() SHALL return False.

        **Validates: Requirements 4.2**
        """
        validator = BlockchainDataValidator()
        assert validator.validate_tx_hash(tx_hash) is False, (
            f"Tx hash with non-hex chars '{tx_hash}' was incorrectly accepted"
        )

    @settings(max_examples=100)
    @given(value=non_string_values())
    def test_property_6_non_string_values_rejected(self, value):
        """
        **Feature: agent-security-hardening, Property 6: Transaction Hash
        Validation Correctness**

        For any non-string value, validate_tx_hash() SHALL return False.

        **Validates: Requirements 4.2**
        """
        validator = BlockchainDataValidator()
        assert validator.validate_tx_hash(value) is False, (
            f"Non-string value {type(value).__name__} was incorrectly accepted"
        )

    @settings(max_examples=100)
    @given(tx_hash=st.text(max_size=150))
    def test_property_6_arbitrary_strings_validation_consistency(self, tx_hash):
        """
        **Feature: agent-security-hardening, Property 6: Transaction Hash
        Validation Correctness**

        For any arbitrary string, validate_tx_hash() SHALL return True
        if and only if the string matches ^0x[a-fA-F0-9]{64}$.

        This is the core property test that verifies the validator
        correctly implements the specification.

        **Validates: Requirements 4.2**
        """
        validator = BlockchainDataValidator()
        pattern = re.compile(r"^0x[a-fA-F0-9]{64}$")

        expected = bool(pattern.match(tx_hash))
        actual = validator.validate_tx_hash(tx_hash)

        assert actual == expected, (
            f"Validation mismatch for '{tx_hash}': "
            f"expected {expected}, got {actual}"
        )

    @settings(max_examples=100)
    @given(tx_hash=valid_transaction_hash())
    def test_property_6_validation_is_deterministic(self, tx_hash):
        """
        **Feature: agent-security-hardening, Property 6: Transaction Hash
        Validation Correctness**

        For any tx hash, calling validate_tx_hash() multiple times
        SHALL return the same result.

        **Validates: Requirements 4.2**
        """
        validator = BlockchainDataValidator()

        result1 = validator.validate_tx_hash(tx_hash)
        result2 = validator.validate_tx_hash(tx_hash)
        result3 = validator.validate_tx_hash(tx_hash)

        assert result1 == result2 == result3, (
            f"Non-deterministic validation for '{tx_hash}': "
            f"got {result1}, {result2}, {result3}"
        )

    @settings(max_examples=100)
    @given(
        lower_body=st.text(
            alphabet="0123456789abcdef",
            min_size=64,
            max_size=64
        ),
    )
    def test_property_6_case_insensitive_hex_accepted(self, lower_body):
        """
        **Feature: agent-security-hardening, Property 6: Transaction Hash
        Validation Correctness**

        For any valid tx hash, both lowercase and uppercase hex characters
        SHALL be accepted (the pattern allows [a-fA-F0-9]).

        **Validates: Requirements 4.2**
        """
        validator = BlockchainDataValidator()

        lower_hash = f"0x{lower_body}"
        upper_hash = f"0x{lower_body.upper()}"
        mixed_body = "".join(
            c.upper() if i % 2 == 0 else c.lower()
            for i, c in enumerate(lower_body)
        )
        mixed_hash = f"0x{mixed_body}"

        assert validator.validate_tx_hash(lower_hash) is True, (
            f"Lowercase tx hash '{lower_hash}' was rejected"
        )
        assert validator.validate_tx_hash(upper_hash) is True, (
            f"Uppercase tx hash '{upper_hash}' was rejected"
        )
        assert validator.validate_tx_hash(mixed_hash) is True, (
            f"Mixed case tx hash '{mixed_hash}' was rejected"
        )


# =============================================================================
# Test Data Strategies for Amount Validation
# =============================================================================


@st.composite
def valid_integer_amount(draw):
    """Generate a valid integer amount string (no decimal point)."""
    # Generate integers from 0 to a reasonable large value
    # We'll test the MAX_AMOUNT boundary separately
    value = draw(st.integers(min_value=0, max_value=10**50))
    return str(value)


@st.composite
def valid_decimal_amount(draw):
    """Generate a valid decimal amount with 1-18 fractional digits."""
    integer_part = draw(st.integers(min_value=0, max_value=10**30))
    # Generate 1-18 fractional digits
    frac_digits = draw(st.integers(min_value=1, max_value=18))
    # Generate fractional part (avoid trailing zeros for cleaner tests)
    frac_value = draw(st.integers(min_value=1, max_value=10**frac_digits - 1))
    frac_str = str(frac_value).zfill(frac_digits)
    return f"{integer_part}.{frac_str}"


@st.composite
def valid_amount(draw):
    """Generate any valid amount (integer or decimal)."""
    return draw(st.one_of(valid_integer_amount(), valid_decimal_amount()))


@st.composite
def invalid_amount_scientific_notation(draw):
    """Generate amounts in scientific notation (should be rejected)."""
    base = draw(st.integers(min_value=1, max_value=999))
    exp = draw(st.integers(min_value=1, max_value=50))
    notation = draw(st.sampled_from(["e", "E", "e+", "E+", "e-", "E-"]))
    return f"{base}{notation}{exp}"


@st.composite
def invalid_amount_negative(draw):
    """Generate negative amounts (should be rejected)."""
    value = draw(st.integers(min_value=1, max_value=10**30))
    return f"-{value}"


@st.composite
def invalid_amount_too_many_decimals(draw):
    """Generate amounts with more than 18 fractional digits."""
    integer_part = draw(st.integers(min_value=0, max_value=10**10))
    # Generate 19-30 fractional digits
    frac_digits = draw(st.integers(min_value=19, max_value=30))
    frac_value = draw(st.integers(min_value=1, max_value=10**frac_digits - 1))
    frac_str = str(frac_value).zfill(frac_digits)
    return f"{integer_part}.{frac_str}"


@st.composite
def invalid_amount_multiple_decimals(draw):
    """Generate amounts with multiple decimal points."""
    part1 = draw(st.integers(min_value=0, max_value=1000))
    part2 = draw(st.integers(min_value=0, max_value=1000))
    part3 = draw(st.integers(min_value=0, max_value=1000))
    return f"{part1}.{part2}.{part3}"


@st.composite
def invalid_amount_non_numeric(draw):
    """Generate amounts with non-numeric characters."""
    # Mix digits with letters or special chars
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+-="
    position = draw(st.integers(min_value=0, max_value=9))
    digits = [str(draw(st.integers(min_value=0, max_value=9))) for _ in range(10)]
    digits[position] = draw(st.sampled_from(chars))
    return "".join(digits)


@st.composite
def invalid_amount_leading_decimal(draw):
    """Generate amounts starting with decimal point (e.g., '.123')."""
    frac_digits = draw(st.integers(min_value=1, max_value=18))
    frac_value = draw(st.integers(min_value=1, max_value=10**frac_digits - 1))
    return f".{frac_value}"


@st.composite
def invalid_amount_trailing_decimal(draw):
    """Generate amounts ending with decimal point (e.g., '123.')."""
    integer_part = draw(st.integers(min_value=0, max_value=10**10))
    return f"{integer_part}."


# =============================================================================
# Property Tests for Amount Validation
# =============================================================================


class TestAmountValidationCorrectness:
    """
    Property tests for amount validation correctness.

    **Feature: agent-security-hardening, Property 7: Amount Validation
    Correctness**

    For any string, the amount validator SHALL accept it if and only if
    it matches "^[0-9]+(\\.[0-9]{1,18})?$" AND the numeric value is
    <= 2^256-1.

    **Validates: Requirements 4.3**
    """

    # Maximum amount value (2^256 - 1)
    MAX_AMOUNT = 2**256 - 1

    @settings(max_examples=100)
    @given(amount=valid_integer_amount())
    def test_property_7_valid_integer_amounts_accepted(self, amount):
        """
        **Feature: agent-security-hardening, Property 7: Amount Validation
        Correctness**

        For any valid integer amount string, validate_amount()
        SHALL return True if value <= 2^256-1.

        **Validates: Requirements 4.3**
        """
        validator = BlockchainDataValidator()
        value = int(amount)

        if value <= self.MAX_AMOUNT:
            assert validator.validate_amount(amount) is True, (
                f"Valid integer amount '{amount}' was rejected"
            )


    @settings(max_examples=100)
    @given(amount=valid_decimal_amount())
    def test_property_7_valid_decimal_amounts_accepted(self, amount):
        """
        **Feature: agent-security-hardening, Property 7: Amount Validation
        Correctness**

        For any valid decimal amount with 1-18 fractional digits,
        validate_amount() SHALL return True if value <= 2^256-1.

        **Validates: Requirements 4.3**
        """
        from decimal import Decimal
        validator = BlockchainDataValidator()
        value = Decimal(amount)

        if value <= self.MAX_AMOUNT:
            assert validator.validate_amount(amount) is True, (
                f"Valid decimal amount '{amount}' was rejected"
            )

    @settings(max_examples=100)
    @given(amount=invalid_amount_scientific_notation())
    def test_property_7_scientific_notation_rejected(self, amount):
        """
        **Feature: agent-security-hardening, Property 7: Amount Validation
        Correctness**

        For any amount in scientific notation, validate_amount()
        SHALL return False.

        **Validates: Requirements 4.3**
        """
        validator = BlockchainDataValidator()
        assert validator.validate_amount(amount) is False, (
            f"Scientific notation '{amount}' was incorrectly accepted"
        )


    @settings(max_examples=100)
    @given(amount=invalid_amount_negative())
    def test_property_7_negative_amounts_rejected(self, amount):
        """
        **Feature: agent-security-hardening, Property 7: Amount Validation
        Correctness**

        For any negative amount, validate_amount() SHALL return False.

        **Validates: Requirements 4.3**
        """
        validator = BlockchainDataValidator()
        assert validator.validate_amount(amount) is False, (
            f"Negative amount '{amount}' was incorrectly accepted"
        )

    @settings(max_examples=100)
    @given(amount=invalid_amount_too_many_decimals())
    def test_property_7_too_many_decimals_rejected(self, amount):
        """
        **Feature: agent-security-hardening, Property 7: Amount Validation
        Correctness**

        For any amount with more than 18 fractional digits,
        validate_amount() SHALL return False.

        **Validates: Requirements 4.3**
        """
        validator = BlockchainDataValidator()
        assert validator.validate_amount(amount) is False, (
            f"Amount with >18 decimals '{amount}' was incorrectly accepted"
        )


    @settings(max_examples=100)
    @given(amount=invalid_amount_multiple_decimals())
    def test_property_7_multiple_decimals_rejected(self, amount):
        """
        **Feature: agent-security-hardening, Property 7: Amount Validation
        Correctness**

        For any amount with multiple decimal points,
        validate_amount() SHALL return False.

        **Validates: Requirements 4.3**
        """
        validator = BlockchainDataValidator()
        assert validator.validate_amount(amount) is False, (
            f"Amount with multiple decimals '{amount}' was accepted"
        )

    @settings(max_examples=100)
    @given(amount=invalid_amount_non_numeric())
    def test_property_7_non_numeric_chars_rejected(self, amount):
        """
        **Feature: agent-security-hardening, Property 7: Amount Validation
        Correctness**

        For any amount containing non-numeric characters,
        validate_amount() SHALL return False.

        **Validates: Requirements 4.3**
        """
        validator = BlockchainDataValidator()
        assert validator.validate_amount(amount) is False, (
            f"Non-numeric amount '{amount}' was incorrectly accepted"
        )


    @settings(max_examples=100)
    @given(amount=invalid_amount_leading_decimal())
    def test_property_7_leading_decimal_rejected(self, amount):
        """
        **Feature: agent-security-hardening, Property 7: Amount Validation
        Correctness**

        For any amount starting with a decimal point (e.g., '.123'),
        validate_amount() SHALL return False.

        **Validates: Requirements 4.3**
        """
        validator = BlockchainDataValidator()
        assert validator.validate_amount(amount) is False, (
            f"Leading decimal amount '{amount}' was incorrectly accepted"
        )

    @settings(max_examples=100)
    @given(amount=invalid_amount_trailing_decimal())
    def test_property_7_trailing_decimal_rejected(self, amount):
        """
        **Feature: agent-security-hardening, Property 7: Amount Validation
        Correctness**

        For any amount ending with a decimal point (e.g., '123.'),
        validate_amount() SHALL return False.

        **Validates: Requirements 4.3**
        """
        validator = BlockchainDataValidator()
        assert validator.validate_amount(amount) is False, (
            f"Trailing decimal amount '{amount}' was incorrectly accepted"
        )


    @settings(max_examples=100)
    @given(value=non_string_values())
    def test_property_7_non_string_values_rejected(self, value):
        """
        **Feature: agent-security-hardening, Property 7: Amount Validation
        Correctness**

        For any non-string value, validate_amount() SHALL return False.

        **Validates: Requirements 4.3**
        """
        validator = BlockchainDataValidator()
        assert validator.validate_amount(value) is False, (
            f"Non-string value {type(value).__name__} was accepted"
        )

    @settings(max_examples=100)
    @given(amount=st.text(max_size=100))
    def test_property_7_arbitrary_strings_validation_consistency(self, amount):
        """
        **Feature: agent-security-hardening, Property 7: Amount Validation
        Correctness**

        For any arbitrary string, validate_amount() SHALL return True
        if and only if the string matches ^[0-9]+(\\.[0-9]{1,18})?$
        AND the numeric value is <= 2^256-1.

        This is the core property test that verifies the validator
        correctly implements the specification.

        **Validates: Requirements 4.3**
        """
        from decimal import Decimal, InvalidOperation
        validator = BlockchainDataValidator()
        pattern = re.compile(r"^[0-9]+(\.[0-9]{1,18})?$")

        # Check pattern match
        pattern_matches = bool(pattern.match(amount))

        # Check bounds if pattern matches
        within_bounds = False
        if pattern_matches:
            try:
                value = Decimal(amount)
                within_bounds = value <= self.MAX_AMOUNT
            except InvalidOperation:
                within_bounds = False

        expected = pattern_matches and within_bounds
        actual = validator.validate_amount(amount)

        assert actual == expected, (
            f"Validation mismatch for '{amount}': "
            f"expected {expected}, got {actual}"
        )


    @settings(max_examples=100)
    @given(amount=valid_amount())
    def test_property_7_validation_is_deterministic(self, amount):
        """
        **Feature: agent-security-hardening, Property 7: Amount Validation
        Correctness**

        For any amount, calling validate_amount() multiple times
        SHALL return the same result.

        **Validates: Requirements 4.3**
        """
        validator = BlockchainDataValidator()

        result1 = validator.validate_amount(amount)
        result2 = validator.validate_amount(amount)
        result3 = validator.validate_amount(amount)

        assert result1 == result2 == result3, (
            f"Non-deterministic validation for '{amount}': "
            f"got {result1}, {result2}, {result3}"
        )

    def test_property_7_max_amount_boundary(self):
        """
        **Feature: agent-security-hardening, Property 7: Amount Validation
        Correctness**

        The maximum valid amount (2^256-1) SHALL be accepted,
        and any value exceeding it SHALL be rejected.

        **Validates: Requirements 4.3**
        """
        validator = BlockchainDataValidator()

        # Exactly at max should be valid
        max_amount = str(self.MAX_AMOUNT)
        assert validator.validate_amount(max_amount) is True, (
            f"Max amount (2^256-1) was rejected"
        )

        # One above max should be invalid
        over_max = str(self.MAX_AMOUNT + 1)
        assert validator.validate_amount(over_max) is False, (
            f"Amount exceeding 2^256-1 was incorrectly accepted"
        )


    @settings(max_examples=100)
    @given(
        integer_part=st.integers(min_value=0, max_value=10**20),
        frac_digits=st.integers(min_value=1, max_value=18),
    )
    def test_property_7_valid_fractional_digit_counts(
        self, integer_part, frac_digits
    ):
        """
        **Feature: agent-security-hardening, Property 7: Amount Validation
        Correctness**

        For any amount with 1-18 fractional digits, validate_amount()
        SHALL return True (assuming value <= 2^256-1).

        **Validates: Requirements 4.3**
        """
        validator = BlockchainDataValidator()

        # Generate fractional part with exact digit count
        frac_value = "1" * frac_digits
        amount = f"{integer_part}.{frac_value}"

        assert validator.validate_amount(amount) is True, (
            f"Amount with {frac_digits} fractional digits was rejected"
        )

    def test_property_7_zero_amount_valid(self):
        """
        **Feature: agent-security-hardening, Property 7: Amount Validation
        Correctness**

        Zero amounts in various formats SHALL be accepted.

        **Validates: Requirements 4.3**
        """
        validator = BlockchainDataValidator()

        # Various zero representations
        assert validator.validate_amount("0") is True
        assert validator.validate_amount("00") is True
        assert validator.validate_amount("0.0") is True
        assert validator.validate_amount("0.000000000000000000") is True


# =============================================================================
# Property Tests for Address Normalization Idempotence
# =============================================================================


class TestAddressNormalizationIdempotence:
    """
    Property tests for address normalization idempotence.

    **Feature: agent-security-hardening, Property 8: Address Normalization
    Idempotence**

    For any valid address, normalizing it twice SHALL produce the same
    result as normalizing once (lowercase).

    **Validates: Requirements 4.5**
    """

    @settings(max_examples=100)
    @given(address=valid_ethereum_address())
    def test_property_8_normalization_idempotence(self, address):
        """
        **Feature: agent-security-hardening, Property 8: Address Normalization
        Idempotence**

        For any valid address, normalize_address(normalize_address(address))
        SHALL equal normalize_address(address).

        **Validates: Requirements 4.5**
        """
        validator = BlockchainDataValidator()

        # Normalize once
        normalized_once = validator.normalize_address(address)
        # Normalize twice
        normalized_twice = validator.normalize_address(normalized_once)

        assert normalized_once == normalized_twice, (
            f"Normalization is not idempotent for '{address}': "
            f"once='{normalized_once}', twice='{normalized_twice}'"
        )

    @settings(max_examples=100)
    @given(address=valid_ethereum_address())
    def test_property_8_normalization_produces_lowercase(self, address):
        """
        **Feature: agent-security-hardening, Property 8: Address Normalization
        Idempotence**

        For any valid address, normalize_address() SHALL produce a
        lowercase result.

        **Validates: Requirements 4.5**
        """
        validator = BlockchainDataValidator()

        normalized = validator.normalize_address(address)

        assert normalized == normalized.lower(), (
            f"Normalized address '{normalized}' is not lowercase"
        )
        assert normalized == address.lower(), (
            f"Normalized address '{normalized}' does not match "
            f"expected lowercase '{address.lower()}'"
        )

    @settings(max_examples=100)
    @given(
        lower_body=st.text(alphabet="0123456789abcdef", min_size=40, max_size=40),
    )
    def test_property_8_case_variants_normalize_to_same_value(self, lower_body):
        """
        **Feature: agent-security-hardening, Property 8: Address Normalization
        Idempotence**

        For any valid address, all case variants (lowercase, uppercase,
        mixed case) SHALL normalize to the same lowercase value.

        **Validates: Requirements 4.5**
        """
        validator = BlockchainDataValidator()

        # Create different case variants
        lower_address = f"0x{lower_body}"
        upper_address = f"0x{lower_body.upper()}"
        mixed_body = "".join(
            c.upper() if i % 2 == 0 else c.lower()
            for i, c in enumerate(lower_body)
        )
        mixed_address = f"0x{mixed_body}"

        # All should normalize to the same lowercase value
        expected = lower_address.lower()

        assert validator.normalize_address(lower_address) == expected, (
            f"Lowercase address normalization failed"
        )
        assert validator.normalize_address(upper_address) == expected, (
            f"Uppercase address normalization failed"
        )
        assert validator.normalize_address(mixed_address) == expected, (
            f"Mixed case address normalization failed"
        )

    @settings(max_examples=100)
    @given(address=valid_ethereum_address())
    def test_property_8_normalized_address_is_valid(self, address):
        """
        **Feature: agent-security-hardening, Property 8: Address Normalization
        Idempotence**

        For any valid address, the normalized result SHALL also be a
        valid address.

        **Validates: Requirements 4.5**
        """
        validator = BlockchainDataValidator()

        normalized = validator.normalize_address(address)

        # The normalized address should still be valid
        assert validator.validate_address(normalized) is True, (
            f"Normalized address '{normalized}' is not valid"
        )

    @settings(max_examples=100)
    @given(address=valid_ethereum_address())
    def test_property_8_normalization_preserves_prefix(self, address):
        """
        **Feature: agent-security-hardening, Property 8: Address Normalization
        Idempotence**

        For any valid address, normalization SHALL preserve the "0x" prefix.

        **Validates: Requirements 4.5**
        """
        validator = BlockchainDataValidator()

        normalized = validator.normalize_address(address)

        assert normalized.startswith("0x"), (
            f"Normalized address '{normalized}' does not start with '0x'"
        )

    @settings(max_examples=100)
    @given(address=valid_ethereum_address())
    def test_property_8_normalization_preserves_length(self, address):
        """
        **Feature: agent-security-hardening, Property 8: Address Normalization
        Idempotence**

        For any valid address, normalization SHALL preserve the length
        (42 characters: "0x" + 40 hex chars).

        **Validates: Requirements 4.5**
        """
        validator = BlockchainDataValidator()

        normalized = validator.normalize_address(address)

        assert len(normalized) == 42, (
            f"Normalized address '{normalized}' has wrong length "
            f"({len(normalized)} != 42)"
        )
        assert len(normalized) == len(address), (
            f"Normalization changed address length from {len(address)} "
            f"to {len(normalized)}"
        )

    @settings(max_examples=100)
    @given(
        n=st.integers(min_value=1, max_value=10),
        address=valid_ethereum_address(),
    )
    def test_property_8_multiple_normalizations_idempotent(self, n, address):
        """
        **Feature: agent-security-hardening, Property 8: Address Normalization
        Idempotence**

        For any valid address, normalizing it N times (N >= 1) SHALL
        produce the same result as normalizing once.

        **Validates: Requirements 4.5**
        """
        validator = BlockchainDataValidator()

        # Normalize once
        normalized_once = validator.normalize_address(address)

        # Normalize N times
        result = address
        for _ in range(n):
            result = validator.normalize_address(result)

        assert result == normalized_once, (
            f"Normalizing {n} times produced different result: "
            f"once='{normalized_once}', {n}x='{result}'"
        )

    @settings(max_examples=100)
    @given(address=valid_ethereum_address())
    def test_property_8_normalization_is_deterministic(self, address):
        """
        **Feature: agent-security-hardening, Property 8: Address Normalization
        Idempotence**

        For any address, calling normalize_address() multiple times
        SHALL return the same result.

        **Validates: Requirements 4.5**
        """
        validator = BlockchainDataValidator()

        result1 = validator.normalize_address(address)
        result2 = validator.normalize_address(address)
        result3 = validator.normalize_address(address)

        assert result1 == result2 == result3, (
            f"Non-deterministic normalization for '{address}': "
            f"got '{result1}', '{result2}', '{result3}'"
        )


# =============================================================================
# Unit Tests for Validation Error Handling
# =============================================================================


class TestValidationErrorHandling:
    """
    Unit tests for validation error handling behavior.

    These tests verify that:
    1. Invalid records are skipped (not processed)
    2. Warning logs contain field name only (not the invalid value)

    **Validates: Requirements 4.4**
    """

    def test_validate_record_skips_invalid_address(self):
        """
        Test that validate_record returns is_valid=False for invalid address
        and the record can be skipped.

        **Validates: Requirements 4.4**
        """
        validator = BlockchainDataValidator()

        # Invalid address (wrong length)
        results = validator.validate_record(address="0x123")

        assert len(results) == 1
        assert results[0].is_valid is False
        assert results[0].field_name == "address"
        assert results[0].error_message is not None

    def test_validate_record_skips_invalid_tx_hash(self):
        """
        Test that validate_record returns is_valid=False for invalid tx hash
        and the record can be skipped.

        **Validates: Requirements 4.4**
        """
        validator = BlockchainDataValidator()

        # Invalid tx hash (wrong length)
        results = validator.validate_record(tx_hash="0xabc")

        assert len(results) == 1
        assert results[0].is_valid is False
        assert results[0].field_name == "tx_hash"
        assert results[0].error_message is not None

    def test_validate_record_skips_invalid_amount(self):
        """
        Test that validate_record returns is_valid=False for invalid amount
        and the record can be skipped.

        **Validates: Requirements 4.4**
        """
        validator = BlockchainDataValidator()

        # Invalid amount (negative)
        results = validator.validate_record(amount="-100")

        assert len(results) == 1
        assert results[0].is_valid is False
        assert results[0].field_name == "amount"
        assert results[0].error_message is not None

    def test_validate_record_skips_invalid_timestamp(self):
        """
        Test that validate_record returns is_valid=False for invalid timestamp
        and the record can be skipped.

        **Validates: Requirements 4.4**
        """
        validator = BlockchainDataValidator()

        # Invalid timestamp (before genesis)
        results = validator.validate_record(timestamp=1000000, chain="ethereum")

        assert len(results) == 1
        assert results[0].is_valid is False
        assert results[0].field_name == "timestamp"
        assert results[0].error_message is not None

    def test_validate_record_skips_invalid_block_number(self):
        """
        Test that validate_record returns is_valid=False for invalid block number
        and the record can be skipped.

        **Validates: Requirements 4.4**
        """
        validator = BlockchainDataValidator()

        # Invalid block number (negative)
        results = validator.validate_record(block_number=-1)

        assert len(results) == 1
        assert results[0].is_valid is False
        assert results[0].field_name == "block_number"
        assert results[0].error_message is not None

    def test_validate_record_multiple_invalid_fields(self):
        """
        Test that validate_record handles multiple invalid fields correctly,
        allowing each to be skipped independently.

        **Validates: Requirements 4.4**
        """
        validator = BlockchainDataValidator()

        # Multiple invalid fields
        results = validator.validate_record(
            address="invalid",
            tx_hash="invalid",
            amount="not-a-number",
        )

        assert len(results) == 3
        # All should be invalid
        assert all(r.is_valid is False for r in results)
        # Each should have the correct field name
        field_names = {r.field_name for r in results}
        assert field_names == {"address", "tx_hash", "amount"}

    def test_validate_record_mixed_valid_invalid_fields(self):
        """
        Test that validate_record correctly identifies valid and invalid fields
        in the same record, allowing selective skipping.

        **Validates: Requirements 4.4**
        """
        validator = BlockchainDataValidator()

        # Mix of valid and invalid fields
        valid_address = "0x" + "a" * 40
        results = validator.validate_record(
            address=valid_address,
            tx_hash="invalid",
            amount="100.5",
        )

        assert len(results) == 3

        # Find results by field name
        address_result = next(r for r in results if r.field_name == "address")
        tx_hash_result = next(r for r in results if r.field_name == "tx_hash")
        amount_result = next(r for r in results if r.field_name == "amount")

        assert address_result.is_valid is True
        assert tx_hash_result.is_valid is False
        assert amount_result.is_valid is True

    def test_warning_logging_contains_field_name_only(self, caplog):
        """
        Test that warning logs contain field name only, not the invalid value.

        Per Requirement 4.4: "IF any blockchain data field fails validation
        THEN the Agent SHALL skip that record and log a warning with the
        field name (not the invalid value)"

        **Validates: Requirements 4.4**
        """
        import logging

        validator = BlockchainDataValidator()

        # Invalid values that should NOT appear in logs
        invalid_address = "0xMALICIOUS_INJECTION_ATTEMPT"
        invalid_tx_hash = "0xSECRET_DATA_LEAK"
        invalid_amount = "SENSITIVE_INFO_123"

        with caplog.at_level(logging.WARNING):
            validator.validate_record(
                address=invalid_address,
                tx_hash=invalid_tx_hash,
                amount=invalid_amount,
            )

        # Check that warnings were logged
        assert len(caplog.records) >= 3

        # Check that field names appear in logs
        log_text = caplog.text
        assert "address" in log_text
        assert "tx_hash" in log_text
        assert "amount" in log_text

        # Check that invalid values do NOT appear in logs
        assert "MALICIOUS_INJECTION_ATTEMPT" not in log_text
        assert "SECRET_DATA_LEAK" not in log_text
        assert "SENSITIVE_INFO_123" not in log_text

    def test_warning_logging_for_invalid_address(self, caplog):
        """
        Test that warning log for invalid address contains only field name.

        **Validates: Requirements 4.4**
        """
        import logging

        validator = BlockchainDataValidator()
        invalid_address = "0xATTACKER_PAYLOAD_HERE"

        with caplog.at_level(logging.WARNING):
            validator.validate_record(address=invalid_address)

        # Should have exactly one warning
        warning_records = [
            r for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert len(warning_records) == 1

        # Warning should mention field name
        assert "address" in warning_records[0].message

        # Warning should NOT contain the invalid value
        assert "ATTACKER_PAYLOAD_HERE" not in warning_records[0].message

    def test_warning_logging_for_invalid_tx_hash(self, caplog):
        """
        Test that warning log for invalid tx hash contains only field name.

        **Validates: Requirements 4.4**
        """
        import logging

        validator = BlockchainDataValidator()
        invalid_hash = "0xSENSITIVE_TRANSACTION_DATA"

        with caplog.at_level(logging.WARNING):
            validator.validate_record(tx_hash=invalid_hash)

        warning_records = [
            r for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert len(warning_records) == 1
        assert "tx_hash" in warning_records[0].message
        assert "SENSITIVE_TRANSACTION_DATA" not in warning_records[0].message

    def test_warning_logging_for_invalid_amount(self, caplog):
        """
        Test that warning log for invalid amount contains only field name.

        **Validates: Requirements 4.4**
        """
        import logging

        validator = BlockchainDataValidator()
        invalid_amount = "SECRET_BALANCE_999999"

        with caplog.at_level(logging.WARNING):
            validator.validate_record(amount=invalid_amount)

        warning_records = [
            r for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert len(warning_records) == 1
        assert "amount" in warning_records[0].message
        assert "SECRET_BALANCE_999999" not in warning_records[0].message

    def test_warning_logging_for_invalid_timestamp(self, caplog):
        """
        Test that warning log for invalid timestamp contains only field name.

        **Validates: Requirements 4.4**
        """
        import logging

        validator = BlockchainDataValidator()
        # Timestamp before genesis
        invalid_timestamp = 1000000

        with caplog.at_level(logging.WARNING):
            validator.validate_record(timestamp=invalid_timestamp, chain="ethereum")

        warning_records = [
            r for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert len(warning_records) == 1
        assert "timestamp" in warning_records[0].message
        # The numeric value should not appear in the message
        assert "1000000" not in warning_records[0].message

    def test_warning_logging_for_invalid_block_number(self, caplog):
        """
        Test that warning log for invalid block number contains only field name.

        **Validates: Requirements 4.4**
        """
        import logging

        validator = BlockchainDataValidator()
        invalid_block = -999

        with caplog.at_level(logging.WARNING):
            validator.validate_record(block_number=invalid_block)

        warning_records = [
            r for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert len(warning_records) == 1
        assert "block_number" in warning_records[0].message
        assert "-999" not in warning_records[0].message

    def test_no_warning_for_valid_record(self, caplog):
        """
        Test that no warning is logged for valid records.

        **Validates: Requirements 4.4**
        """
        import logging
        import time

        validator = BlockchainDataValidator()

        # All valid fields
        valid_address = "0x" + "a" * 40
        valid_tx_hash = "0x" + "b" * 64
        valid_amount = "1000.123456789012345678"
        valid_timestamp = int(time.time()) - 3600  # 1 hour ago
        valid_block = 1000000

        with caplog.at_level(logging.WARNING):
            results = validator.validate_record(
                address=valid_address,
                tx_hash=valid_tx_hash,
                amount=valid_amount,
                timestamp=valid_timestamp,
                block_number=valid_block,
                chain="ethereum",
            )

        # All should be valid
        assert all(r.is_valid for r in results)

        # No warnings should be logged
        warning_records = [
            r for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert len(warning_records) == 0

    def test_skip_behavior_allows_filtering_invalid_records(self):
        """
        Test that the validation results can be used to filter out invalid
        records from a batch.

        **Validates: Requirements 4.4**
        """
        validator = BlockchainDataValidator()

        # Simulate a batch of records with some invalid
        records = [
            {"address": "0x" + "a" * 40, "amount": "100"},  # Valid
            {"address": "invalid", "amount": "200"},  # Invalid address
            {"address": "0x" + "b" * 40, "amount": "-50"},  # Invalid amount
            {"address": "0x" + "c" * 40, "amount": "300"},  # Valid
        ]

        valid_records = []
        for record in records:
            results = validator.validate_record(
                address=record["address"],
                amount=record["amount"],
            )
            # Skip if any field is invalid
            if all(r.is_valid for r in results):
                valid_records.append(record)

        # Should have filtered to only valid records
        assert len(valid_records) == 2
        assert valid_records[0]["amount"] == "100"
        assert valid_records[1]["amount"] == "300"

    def test_error_message_does_not_contain_invalid_value(self):
        """
        Test that error messages in ValidationResult do not contain
        the actual invalid value.

        **Validates: Requirements 4.4**
        """
        validator = BlockchainDataValidator()

        # Test with various invalid values
        test_cases = [
            ("address", "0xSECRET_ADDRESS_DATA"),
            ("tx_hash", "0xPRIVATE_TX_INFO"),
            ("amount", "CONFIDENTIAL_AMOUNT"),
        ]

        for field_name, invalid_value in test_cases:
            results = validator.validate_record(**{field_name: invalid_value})

            assert len(results) == 1
            result = results[0]

            # Error message should exist but not contain the invalid value
            assert result.error_message is not None
            assert invalid_value not in result.error_message, (
                f"Error message for {field_name} contains invalid value: "
                f"{result.error_message}"
            )

    def test_timestamp_validation_without_chain_logs_warning(self, caplog):
        """
        Test that timestamp validation without chain parameter logs
        appropriate warning.

        **Validates: Requirements 4.4**
        """
        import logging

        validator = BlockchainDataValidator()

        with caplog.at_level(logging.WARNING):
            results = validator.validate_record(timestamp=1600000000)

        # Should have one result for timestamp
        assert len(results) == 1
        assert results[0].is_valid is False
        assert results[0].field_name == "timestamp"
        assert "chain" in results[0].error_message.lower()
