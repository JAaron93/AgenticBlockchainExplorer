"""
Property-based tests for safe file path handling.

These tests use Hypothesis to verify correctness properties defined in the design document.

**Feature: agent-security-hardening, Property 9: Path Containment**
**Validates: Requirements 5.1, 5.2**
"""

import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
from hypothesis import given, strategies as st, settings, assume
import unittest.mock as mock

from core.security.safe_path_handler import (
    SafePathHandler,
    PathTraversalError,
    InvalidFilenameError,
)


# =============================================================================
# Helper Functions
# =============================================================================


@contextmanager
def temp_base_directory():
    """Create a temporary base directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Test Data Strategies
# =============================================================================


def safe_path_component():
    """Generate a safe path component (no traversal sequences).
    
    Path components must:
    - Not be empty
    - Not contain path separators (/ or \\)
    - Not be '.' or '..'
    - Not contain null bytes
    """
    return st.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
        min_size=1,
        max_size=50,
    ).filter(lambda x: x not in (".", "..") and "\x00" not in x)


def traversal_sequence():
    """Generate path traversal sequences."""
    return st.sampled_from([
        "..",
        "../",
        "..\\",
        "../..",
        "../../",
        "..\\..\\",
        "./../",
        ".\\.\\..\\",
    ])


@st.composite
def safe_relative_path(draw):
    """Generate a safe relative path with 1-4 components."""
    num_components = draw(st.integers(min_value=1, max_value=4))
    components = [draw(safe_path_component()) for _ in range(num_components)]
    return "/".join(components)


@st.composite
def path_with_traversal(draw):
    """Generate a path that contains traversal sequences.
    
    Returns: (path_parts, traversal_type)
    """
    traversal_type = draw(st.sampled_from([
        "prefix",      # ../safe
        "middle",      # safe/../other
        "suffix",      # safe/..
        "multiple",    # ../safe/../other
    ]))
    
    safe_part = draw(safe_path_component())
    
    if traversal_type == "prefix":
        parts = ["..", safe_part]
    elif traversal_type == "middle":
        other_part = draw(safe_path_component())
        parts = [safe_part, "..", other_part]
    elif traversal_type == "suffix":
        parts = [safe_part, ".."]
    else:  # multiple
        other_part = draw(safe_path_component())
        parts = ["..", safe_part, "..", other_part]
    
    return parts, traversal_type


@st.composite
def absolute_path_outside_base(draw, base_dir: str):
    """Generate an absolute path that is outside the base directory."""
    # Generate a path that starts from root but goes elsewhere
    components = [draw(safe_path_component()) for _ in range(2)]
    
    # Use anchor from base_dir for cross-platform root
    anchor = Path(base_dir).anchor
    path_obj = Path(anchor, *components)
    
    # Make sure it's not accidentally inside base_dir
    try:
        path_obj.relative_to(base_dir)
        # If relative_to succeeds, it is inside the base_dir
        assume(False)
    except ValueError:
        pass
    
    return str(path_obj)


# =============================================================================
# Property Tests
# =============================================================================


class TestPathContainment:
    """
    Property tests for path containment.
    
    **Feature: agent-security-hardening, Property 9: Path Containment**
    
    For any path constructed via SafePathHandler, the resolved absolute path
    SHALL be within the configured base directory.
    
    **Validates: Requirements 5.1, 5.2**
    """

    @settings(max_examples=100)
    @given(path_parts=st.lists(safe_path_component(), min_size=1, max_size=5))
    def test_property_9_safe_join_stays_within_base(self, path_parts):
        """
        **Feature: agent-security-hardening, Property 9: Path Containment**
        
        For any sequence of safe path components, safe_join() SHALL return
        a path that is within the base directory.
        
        **Validates: Requirements 5.1, 5.2**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)
            
            # Join the path parts
            result = handler.safe_join(*path_parts)
            
            # The result should be within the base directory
            assert handler.validate_path(result) is True, (
                f"safe_join() result '{result}' should be within base directory "
                f"'{temp_base_dir}'"
            )
            
            # The result should start with the base directory (resolved)
            # Note: On macOS, /var is a symlink to /private/var, so we need
            # to compare resolved paths
            resolved_base = temp_base_dir.resolve()
            try:
                result.relative_to(resolved_base)
            except ValueError:
                pytest.fail(
                    f"safe_join() result '{result}' is not relative to base "
                    f"directory '{resolved_base}'"
                )

    @settings(max_examples=100)
    @given(data=path_with_traversal())
    def test_property_9_traversal_sequences_rejected(self, data):
        """
        **Feature: agent-security-hardening, Property 9: Path Containment**
        
        For any path containing traversal sequences (.., ../, etc.),
        safe_join() SHALL raise PathTraversalError.
        
        **Validates: Requirements 5.1, 5.2**
        """
        with temp_base_directory() as temp_base_dir:
            path_parts, traversal_type = data
            handler = SafePathHandler(temp_base_dir)
            
            # Attempting to join path with traversal should raise error
            with pytest.raises(PathTraversalError) as exc_info:
                handler.safe_join(*path_parts)
            
            assert "traversal" in str(exc_info.value).lower() or "escapes" in str(exc_info.value).lower(), (
                f"Error message should mention traversal or escaping. "
                f"Got: {exc_info.value}"
            )

    @settings(max_examples=100)
    @given(safe_path=safe_relative_path())
    def test_property_9_validate_path_accepts_contained_paths(self, safe_path):
        """
        **Feature: agent-security-hardening, Property 9: Path Containment**
        
        For any path that is within the base directory, validate_path()
        SHALL return True.
        
        **Validates: Requirements 5.1, 5.2**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)
            
            # Construct a path within the base directory
            contained_path = temp_base_dir / safe_path
            
            # validate_path should return True
            assert handler.validate_path(contained_path) is True, (
                f"validate_path() should return True for contained path "
                f"'{contained_path}'"
            )

    @settings(max_examples=100)
    @given(
        safe_component=safe_path_component(),
        num_traversals=st.integers(min_value=1, max_value=10),
    )
    def test_property_9_multiple_traversals_rejected(self, safe_component, num_traversals):
        """
        **Feature: agent-security-hardening, Property 9: Path Containment**
        
        For any path with multiple traversal sequences, safe_join()
        SHALL raise PathTraversalError regardless of how many traversals.
        
        **Validates: Requirements 5.1, 5.2**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)
            
            # Create path with multiple traversals
            traversals = [".."] * num_traversals
            path_parts = traversals + [safe_component]
            
            # Should raise PathTraversalError
            with pytest.raises(PathTraversalError):
                handler.safe_join(*path_parts)

    @settings(max_examples=100)
    @given(safe_path=safe_relative_path())
    def test_property_9_resolved_path_is_absolute(self, safe_path):
        """
        **Feature: agent-security-hardening, Property 9: Path Containment**
        
        For any safe path, safe_join() SHALL return an absolute path.
        
        **Validates: Requirements 5.1**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)
            
            # Split the path into components
            parts = safe_path.split("/")
            
            # Join the path
            result = handler.safe_join(*parts)
            
            # Result should be absolute
            assert result.is_absolute(), (
                f"safe_join() should return absolute path. Got: '{result}'"
            )

    @settings(max_examples=100)
    @given(
        component1=safe_path_component(),
        component2=safe_path_component(),
    )
    def test_property_9_safe_join_is_deterministic(self, component1, component2):
        """
        **Feature: agent-security-hardening, Property 9: Path Containment**
        
        For any set of path components, safe_join() SHALL return the same
        result when called multiple times with the same inputs.
        
        **Validates: Requirements 5.1**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)
            
            # Call safe_join multiple times
            result1 = handler.safe_join(component1, component2)
            result2 = handler.safe_join(component1, component2)
            result3 = handler.safe_join(component1, component2)
            
            # All results should be identical
            assert result1 == result2 == result3, (
                f"safe_join() should be deterministic. "
                f"Got: {result1}, {result2}, {result3}"
            )

    @settings(max_examples=100)
    @given(safe_component=safe_path_component())
    def test_property_9_null_byte_injection_rejected(self, safe_component):
        """
        **Feature: agent-security-hardening, Property 9: Path Containment**
        
        For any path containing null bytes, safe_join() SHALL raise
        PathTraversalError (null bytes can be used to bypass checks).
        
        **Validates: Requirements 5.1, 5.2**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)
            
            # Inject null byte into path component
            malicious_component = safe_component + "\x00" + ".."
            
            # Should raise PathTraversalError
            with pytest.raises(PathTraversalError):
                handler.safe_join(malicious_component)

    def test_property_9_empty_parts_returns_base(self):
        """
        **Feature: agent-security-hardening, Property 9: Path Containment**
        
        When safe_join() is called with no parts, it SHALL return the
        base directory itself.
        
        **Validates: Requirements 5.1**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)
            
            # Call with no parts
            result = handler.safe_join()
            
            # Should return base directory
            assert result == temp_base_dir.resolve(), (
                f"safe_join() with no parts should return base directory. "
                f"Expected: {temp_base_dir.resolve()}, Got: {result}"
            )

    @settings(max_examples=100)
    @given(
        component=safe_path_component(),
        depth=st.integers(min_value=1, max_value=20),
    )
    def test_property_9_deep_nesting_stays_contained(self, component, depth):
        """
        **Feature: agent-security-hardening, Property 9: Path Containment**
        
        For any deeply nested path (up to 20 levels), safe_join() SHALL
        return a path within the base directory.
        
        **Validates: Requirements 5.1, 5.2**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)
            
            # Create deeply nested path
            parts = [component] * depth
            
            # Join should succeed and stay within base
            result = handler.safe_join(*parts)
            
            assert handler.validate_path(result) is True, (
                f"Deeply nested path (depth={depth}) should stay within base"
            )

    @settings(max_examples=50)
    @given(safe_component=safe_path_component())
    def test_property_9_symlink_resolution_stays_contained(self, safe_component):
        """
        **Feature: agent-security-hardening, Property 9: Path Containment**
        
        For any path, safe_join() SHALL resolve symlinks and verify the
        final resolved path is within the base directory.
        
        **Validates: Requirements 5.1, 5.2**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)
            subdir = None
            link_path = None
            
            try:
                # Prepare paths
                subdir = temp_base_dir / safe_component
                link_name = f"link_{safe_component}"
                link_path = temp_base_dir / link_name
                
                subdir.mkdir(exist_ok=True)
                
                try:
                    link_path.symlink_to(subdir)
                except OSError:
                    # Skip if symlinks not supported
                    pytest.skip("Symlinks not supported on this system")
                
                # safe_join should resolve the symlink and validate
                result = handler.safe_join(link_name)
                
                # Result should be within base directory
                assert handler.validate_path(result) is True, (
                    f"Symlink resolution should stay within base directory"
                )
            
            finally:
                # Cleanup
                if link_path is not None:
                    try:
                        if link_path.is_symlink() or link_path.exists():
                            link_path.unlink()
                    except OSError:
                        pass
                
                if subdir is not None and subdir.exists():
                    try:
                        subdir.rmdir()
                    except OSError:
                        pass

    @settings(max_examples=100)
    @given(path_parts=st.lists(safe_path_component(), min_size=1, max_size=3))
    def test_property_9_validate_path_consistent_with_safe_join(self, path_parts):
        """
        **Feature: agent-security-hardening, Property 9: Path Containment**
        
        For any path returned by safe_join(), validate_path() SHALL
        return True (consistency between methods).
        
        **Validates: Requirements 5.1, 5.2**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)
            
            # Get path from safe_join
            result = handler.safe_join(*path_parts)
            
            # validate_path should confirm it's valid
            assert handler.validate_path(result) is True, (
                f"validate_path() should return True for path returned by "
                f"safe_join(). Path: {result}"
            )

    @settings(max_examples=100)
    @given(component=safe_path_component())
    def test_property_9_path_outside_base_rejected_by_validate(self, component):
        """
        **Feature: agent-security-hardening, Property 9: Path Containment**
        
        For any path outside the base directory, validate_path() SHALL
        return False.
        
        **Validates: Requirements 5.1, 5.2**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)
            
            # Create a path outside base directory
            outside_path = Path(tempfile.gettempdir()) / component
            
            # Skip if /tmp happens to be our base (unlikely but possible)
            assume(not str(outside_path.resolve()).startswith(str(temp_base_dir.resolve())))
            
            # validate_path should return False
            assert handler.validate_path(outside_path) is False, (
                f"validate_path() should return False for path outside base. "
                f"Path: {outside_path}, Base: {temp_base_dir}"
            )

    @settings(max_examples=50)
    @given(component=safe_path_component())
    def test_property_9_case_sensitivity_preserved(self, component):
        """
        **Feature: agent-security-hardening, Property 9: Path Containment**
        
        For any path component, safe_join() SHALL preserve the case
        of the component in the returned path.
        
        **Validates: Requirements 5.1**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)
            
            # Test with mixed case
            mixed_case = component[:len(component)//2].upper() + component[len(component)//2:].lower()
            
            result = handler.safe_join(mixed_case)
            
            # The component should appear in the result with preserved case
            assert mixed_case in str(result), (
                f"Case should be preserved. Expected '{mixed_case}' in '{result}'"
            )


class TestPathContainmentEdgeCases:
    """
    Edge case tests for path containment.
    
    These tests verify behavior at boundary conditions.
    """

    def test_property_9_dot_component_handled(self):
        """
        **Feature: agent-security-hardening, Property 9: Path Containment**
        
        A single dot (.) component SHALL be handled safely (current directory).
        
        **Validates: Requirements 5.1**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)
            
            # Single dot should resolve to base directory
            result = handler.safe_join(".")
            
            # Should be within base
            assert handler.validate_path(result) is True

    def test_property_9_double_dot_rejected(self):
        """
        **Feature: agent-security-hardening, Property 9: Path Containment**
        
        A double dot (..) component SHALL be rejected.
        
        **Validates: Requirements 5.1, 5.2**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)
            
            with pytest.raises(PathTraversalError):
                handler.safe_join("..")

    def test_property_9_mixed_separators_handled(self):
        """
        **Feature: agent-security-hardening, Property 9: Path Containment**
        
        Path components with mixed separators SHALL be handled safely.
        
        **Validates: Requirements 5.1, 5.2**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)
            
            # Component with embedded separator should be rejected or handled
            # This depends on implementation - either sanitize or reject
            try:
                result = handler.safe_join("safe/path")
                # If it succeeds, verify containment
                assert handler.validate_path(result) is True
            except PathTraversalError:
                # Also acceptable - rejecting embedded separators
                pass

    def test_property_9_unicode_components_handled(self):
        """
        **Feature: agent-security-hardening, Property 9: Path Containment**
        
        Unicode path components SHALL be handled safely.
        
        **Validates: Requirements 5.1**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)
            
            # Unicode component
            unicode_component = "文件夹"
            
            try:
                result = handler.safe_join(unicode_component)
                # If it succeeds, verify containment
                assert handler.validate_path(result) is True
            except (PathTraversalError, OSError):
                # Some systems may not support unicode paths
                pytest.skip("Unicode paths not supported")

    def test_property_9_very_long_component_handled(self):
        """
        **Feature: agent-security-hardening, Property 9: Path Containment**
        
        Very long path components SHALL be handled (may fail gracefully).
        
        **Validates: Requirements 5.1**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)
            
            # Very long component (300 chars)
            long_component = "a" * 300
            
            try:
                result = handler.safe_join(long_component)
                # If it succeeds, verify containment
                assert handler.validate_path(result) is True
            except (PathTraversalError, OSError):
                # Acceptable - filesystem may reject long names
                pass

    def test_property_9_whitespace_only_component_handled(self):
        """
        **Feature: agent-security-hardening, Property 9: Path Containment**
        
        Whitespace-only components SHALL be handled safely.
        
        **Validates: Requirements 5.1**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)
            
            # Whitespace component
            try:
                result = handler.safe_join("   ")
                # If it succeeds, verify containment
                assert handler.validate_path(result) is True
            except (PathTraversalError, OSError, ValueError):
                # Acceptable - may reject whitespace-only names
                pass


# =============================================================================
# Property 10: Filename Sanitization Safety
# =============================================================================


def filename_with_traversal():
    """Generate filenames containing path traversal sequences."""
    return st.sampled_from([
        "../file.txt",
        "..\\file.txt",
        "file/../other.txt",
        "file\\..\\other.txt",
        "..%2f..%2ffile.txt",
        "....//file.txt",
        "file../.txt",
        "..",
        "../",
        "..\\",
        "a/../b",
        "a\\..\\b",
    ])


def filename_with_null_bytes():
    """Generate filenames containing null bytes."""
    return st.sampled_from([
        "file\x00.txt",
        "\x00file.txt",
        "file.txt\x00",
        "file\x00name\x00.txt",
        "../\x00file.txt",
        "file\x00/../other.txt",
    ])


@st.composite
def arbitrary_filename_with_traversal(draw):
    """Generate arbitrary filenames that include traversal sequences."""
    prefix = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-",
        min_size=0,
        max_size=10,
    ))
    traversal = draw(st.sampled_from(["../", "..\\", "..", "/../", "\\..\\", "/.."]))
    suffix = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-.",
        min_size=0,
        max_size=10,
    ))
    return prefix + traversal + suffix


@st.composite
def arbitrary_filename_with_null(draw):
    """Generate arbitrary filenames that include null bytes."""
    prefix = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-.",
        min_size=0,
        max_size=10,
    ))
    suffix = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-.",
        min_size=1,
        max_size=10,
    ))
    return prefix + "\x00" + suffix


class TestFilenameSanitizationSafety:
    """
    Property tests for filename sanitization safety.

    **Feature: agent-security-hardening, Property 10: Filename Sanitization Safety**

    For any filename after sanitization, it SHALL NOT contain path traversal
    sequences (../, ..\\) or null bytes.

    **Validates: Requirements 5.3**
    """

    @settings(max_examples=100)
    @given(filename=filename_with_traversal())
    def test_property_10_traversal_removed_from_known_patterns(self, filename):
        """
        **Feature: agent-security-hardening, Property 10: Filename Sanitization Safety**

        For any filename containing known traversal patterns, sanitize_filename()
        SHALL return a filename without those traversal sequences.

        **Validates: Requirements 5.3**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)

            try:
                sanitized = handler.sanitize_filename(filename)

                # Sanitized filename must not contain traversal sequences
                assert ".." not in sanitized, (
                    f"Sanitized filename should not contain '..'. "
                    f"Input: {repr(filename)}, Output: {repr(sanitized)}"
                )
                assert "../" not in sanitized, (
                    f"Sanitized filename should not contain '../'. "
                    f"Input: {repr(filename)}, Output: {repr(sanitized)}"
                )
                assert "..\\" not in sanitized, (
                    f"Sanitized filename should not contain '..\\'. "
                    f"Input: {repr(filename)}, Output: {repr(sanitized)}"
                )

            except InvalidFilenameError:
                # Acceptable - filename may be entirely unsafe characters
                pass

    @settings(max_examples=100)
    @given(filename=arbitrary_filename_with_traversal())
    def test_property_10_traversal_removed_from_arbitrary(self, filename):
        """
        **Feature: agent-security-hardening, Property 10: Filename Sanitization Safety**

        For any arbitrary filename containing traversal sequences,
        sanitize_filename() SHALL return a filename without those sequences.

        **Validates: Requirements 5.3**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)

            try:
                sanitized = handler.sanitize_filename(filename)

                # Sanitized filename must not contain traversal sequences
                assert ".." not in sanitized, (
                    f"Sanitized filename should not contain '..'. "
                    f"Input: {repr(filename)}, Output: {repr(sanitized)}"
                )

            except InvalidFilenameError:
                # Acceptable - filename may be entirely unsafe characters
                pass

    @settings(max_examples=100)
    @given(filename=filename_with_null_bytes())
    def test_property_10_null_bytes_removed_from_known_patterns(self, filename):
        """
        **Feature: agent-security-hardening, Property 10: Filename Sanitization Safety**

        For any filename containing null bytes, sanitize_filename()
        SHALL return a filename without null bytes.

        **Validates: Requirements 5.3**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)

            try:
                sanitized = handler.sanitize_filename(filename)

                # Sanitized filename must not contain null bytes
                assert "\x00" not in sanitized, (
                    f"Sanitized filename should not contain null bytes. "
                    f"Input: {repr(filename)}, Output: {repr(sanitized)}"
                )

            except InvalidFilenameError:
                # Acceptable - filename may be entirely unsafe characters
                pass

    @settings(max_examples=100)
    @given(filename=arbitrary_filename_with_null())
    def test_property_10_null_bytes_removed_from_arbitrary(self, filename):
        """
        **Feature: agent-security-hardening, Property 10: Filename Sanitization Safety**

        For any arbitrary filename containing null bytes,
        sanitize_filename() SHALL return a filename without null bytes.

        **Validates: Requirements 5.3**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)

            try:
                sanitized = handler.sanitize_filename(filename)

                # Sanitized filename must not contain null bytes
                assert "\x00" not in sanitized, (
                    f"Sanitized filename should not contain null bytes. "
                    f"Input: {repr(filename)}, Output: {repr(sanitized)}"
                )

            except InvalidFilenameError:
                # Acceptable - filename may be entirely unsafe characters
                pass

    @settings(max_examples=100)
    @given(
        prefix=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
            min_size=1,
            max_size=10,
        ),
        suffix=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
            min_size=1,
            max_size=10,
        ),
        num_dots=st.integers(min_value=2, max_value=10),
    )
    def test_property_10_multiple_dots_handled(self, prefix, suffix, num_dots):
        """
        **Feature: agent-security-hardening, Property 10: Filename Sanitization Safety**

        For any filename with multiple consecutive dots (which could form
        traversal sequences), sanitize_filename() SHALL ensure no '..'
        remains in the output.

        **Validates: Requirements 5.3**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)

            # Create filename with multiple dots
            filename = prefix + ("." * num_dots) + suffix

            try:
                sanitized = handler.sanitize_filename(filename)

                # Sanitized filename must not contain '..'
                assert ".." not in sanitized, (
                    f"Sanitized filename should not contain '..'. "
                    f"Input: {repr(filename)}, Output: {repr(sanitized)}"
                )

            except InvalidFilenameError:
                # Acceptable - filename may become empty after sanitization
                pass

    @settings(max_examples=100)
    @given(
        base_name=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-",
            min_size=1,
            max_size=20,
        ),
        extension=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
            min_size=1,
            max_size=5,
        ),
    )
    def test_property_10_safe_filenames_preserved(self, base_name, extension):
        """
        **Feature: agent-security-hardening, Property 10: Filename Sanitization Safety**

        For any safe filename (no traversal or null bytes),
        sanitize_filename() SHALL preserve the essential content.

        **Validates: Requirements 5.3**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)

            # Create a safe filename
            filename = f"{base_name}.{extension}"

            sanitized = handler.sanitize_filename(filename)

            # Safe filename should be preserved (possibly with minor changes)
            # The sanitized version should not be empty
            assert len(sanitized) > 0, (
                f"Safe filename should not become empty. "
                f"Input: {repr(filename)}, Output: {repr(sanitized)}"
            )

            # Should not contain traversal or null bytes
            assert ".." not in sanitized
            assert "\x00" not in sanitized

    @settings(max_examples=100)
    @given(filename=st.text(min_size=1, max_size=100))
    def test_property_10_universal_no_traversal_or_null(self, filename):
        """
        **Feature: agent-security-hardening, Property 10: Filename Sanitization Safety**

        For ANY filename input, if sanitize_filename() succeeds,
        the output SHALL NOT contain path traversal sequences or null bytes.

        **Validates: Requirements 5.3**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)

            try:
                sanitized = handler.sanitize_filename(filename)

                # Universal property: no traversal sequences
                assert ".." not in sanitized, (
                    f"Sanitized filename should never contain '..'. "
                    f"Input: {repr(filename)}, Output: {repr(sanitized)}"
                )

                # Universal property: no null bytes
                assert "\x00" not in sanitized, (
                    f"Sanitized filename should never contain null bytes. "
                    f"Input: {repr(filename)}, Output: {repr(sanitized)}"
                )

                # Universal property: no path separators
                assert "/" not in sanitized, (
                    f"Sanitized filename should not contain '/'. "
                    f"Input: {repr(filename)}, Output: {repr(sanitized)}"
                )
                assert "\\" not in sanitized, (
                    f"Sanitized filename should not contain '\\'. "
                    f"Input: {repr(filename)}, Output: {repr(sanitized)}"
                )

            except InvalidFilenameError:
                # Acceptable - some filenames cannot be sanitized
                pass

    @settings(max_examples=100)
    @given(
        traversal=st.sampled_from(["../", "..\\", "..", "/../", "\\..\\", "/.."]),
        repeat=st.integers(min_value=1, max_value=5),
    )
    def test_property_10_repeated_traversal_removed(self, traversal, repeat):
        """
        **Feature: agent-security-hardening, Property 10: Filename Sanitization Safety**

        For any filename with repeated traversal sequences,
        sanitize_filename() SHALL remove all of them.

        **Validates: Requirements 5.3**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)

            # Create filename with repeated traversal
            filename = (traversal * repeat) + "file.txt"

            try:
                sanitized = handler.sanitize_filename(filename)

                # All traversal sequences must be removed
                assert ".." not in sanitized, (
                    f"Sanitized filename should not contain '..'. "
                    f"Input: {repr(filename)}, Output: {repr(sanitized)}"
                )

            except InvalidFilenameError:
                # Acceptable - filename may become empty
                pass

    @settings(max_examples=50)
    @given(
        safe_part=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
            min_size=1,
            max_size=10,
        ),
    )
    def test_property_10_combined_traversal_and_null(self, safe_part):
        """
        **Feature: agent-security-hardening, Property 10: Filename Sanitization Safety**

        For any filename containing both traversal sequences AND null bytes,
        sanitize_filename() SHALL remove both.

        **Validates: Requirements 5.3**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)

            # Create filename with both traversal and null bytes
            filename = f"../{safe_part}\x00.txt"

            try:
                sanitized = handler.sanitize_filename(filename)

                # Both must be removed
                assert ".." not in sanitized, (
                    f"Sanitized filename should not contain '..'. "
                    f"Input: {repr(filename)}, Output: {repr(sanitized)}"
                )
                assert "\x00" not in sanitized, (
                    f"Sanitized filename should not contain null bytes. "
                    f"Input: {repr(filename)}, Output: {repr(sanitized)}"
                )

            except InvalidFilenameError:
                # Acceptable
                pass

    def test_property_10_edge_case_only_traversal(self):
        """
        **Feature: agent-security-hardening, Property 10: Filename Sanitization Safety**

        A filename consisting only of traversal sequences SHALL either
        raise InvalidFilenameError or return a safe non-empty string.

        **Validates: Requirements 5.3**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)

            traversal_only = ["../", "..\\", "..", "../..", "../../.."]

            for filename in traversal_only:
                try:
                    sanitized = handler.sanitize_filename(filename)
                    # If it succeeds, must not contain traversal
                    assert ".." not in sanitized
                    assert len(sanitized) > 0
                except InvalidFilenameError:
                    # Expected for traversal-only filenames
                    pass

    def test_property_10_edge_case_only_null_bytes(self):
        """
        **Feature: agent-security-hardening, Property 10: Filename Sanitization Safety**

        A filename consisting only of null bytes SHALL raise
        InvalidFilenameError.

        **Validates: Requirements 5.3**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)

            null_only = ["\x00", "\x00\x00", "\x00\x00\x00"]

            for filename in null_only:
                try:
                    sanitized = handler.sanitize_filename(filename)
                    # If it succeeds, must not contain null bytes
                    assert "\x00" not in sanitized
                    assert len(sanitized) > 0
                except InvalidFilenameError:
                    # Expected for null-only filenames
                    pass


# =============================================================================
# Unit Tests for Atomic Write Operations
# =============================================================================


class TestAtomicWriteOperations:
    """
    Unit tests for atomic write operations.

    These tests verify the atomic_write() method of SafePathHandler
    correctly writes files atomically and handles failures gracefully.

    **Validates: Requirements 5.5**
    """

    def test_atomic_write_successful_write(self):
        """
        Test that atomic_write() successfully writes content to a file.

        **Validates: Requirements 5.5**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)

            target_file = temp_base_dir / "test_file.txt"
            content = b"Hello, World! This is test content."

            handler.atomic_write(target_file, content)

            assert target_file.exists(), "File should exist after atomic write"
            assert target_file.read_bytes() == content, "File content should match"

    def test_atomic_write_overwrites_existing_file(self):
        """
        Test that atomic_write() correctly overwrites an existing file.

        **Validates: Requirements 5.5**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)

            target_file = temp_base_dir / "existing_file.txt"
            original_content = b"Original content"
            new_content = b"New content that replaces the original"

            target_file.write_bytes(original_content)
            assert target_file.read_bytes() == original_content

            handler.atomic_write(target_file, new_content)

            assert target_file.read_bytes() == new_content, (
                "File should contain new content after atomic write"
            )

    @settings(max_examples=20)
    @given(data=st.data())
    def test_verify_outside_base_generator(self, data):
        """
        Verify that absolute_path_outside_base generator produces valid outside paths.
        """
        with temp_base_directory() as temp_base_dir:
            # We must pass base_dir as a string because of the type hint/usage in generator
            path = data.draw(absolute_path_outside_base(base_dir=str(temp_base_dir)))
            
            path_obj = Path(path)
            assert path_obj.is_absolute()
            
            # Should NOT be relative to temp_base_dir
            try:
                path_obj.relative_to(temp_base_dir)
                pytest.fail(f"Generated path {path} should be outside {temp_base_dir}")
            except ValueError:
                pass

    def test_atomic_write_creates_parent_directories(self):
        """
        Test that atomic_write() creates parent directories if needed.

        **Validates: Requirements 5.5**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)

            nested_path = temp_base_dir / "subdir1" / "subdir2" / "file.txt"
            content = b"Content in nested directory"

            assert not nested_path.parent.exists()

            handler.atomic_write(nested_path, content)

            assert nested_path.exists(), "File should exist"
            assert nested_path.read_bytes() == content, "Content should match"

    def test_atomic_write_no_temp_file_left_on_success(self):
        """
        Test that no temporary files are left behind after successful write.

        **Validates: Requirements 5.5**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)

            target_file = temp_base_dir / "clean_write.txt"
            content = b"Test content"

            handler.atomic_write(target_file, content)

            temp_files = list(temp_base_dir.glob(".tmp_*"))
            assert len(temp_files) == 0, (
                f"No temp files should remain. Found: {temp_files}"
            )

    def test_atomic_write_rejects_path_outside_base(self):
        """
        Test that atomic_write() rejects paths outside the base directory.

        **Validates: Requirements 5.5, 5.1, 5.2**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)

            # Use parent directory to ensure it is outside base
            outside_path = temp_base_dir.parent / "outside_base_test.txt"
            content = b"Content should not be written"

            with pytest.raises(PathTraversalError):
                handler.atomic_write(outside_path, content)

    def test_atomic_write_handles_relative_path(self):
        """
        Test that atomic_write() correctly handles relative paths.

        **Validates: Requirements 5.5**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)

            relative_path = "relative_file.txt"
            content = b"Content via relative path"

            handler.atomic_write(relative_path, content)

            expected_path = temp_base_dir / relative_path
            assert expected_path.exists()
            assert expected_path.read_bytes() == content

    def test_atomic_write_handles_empty_content(self):
        """
        Test that atomic_write() correctly handles empty content.

        **Validates: Requirements 5.5**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)

            target_file = temp_base_dir / "empty_file.txt"
            content = b""

            handler.atomic_write(target_file, content)

            assert target_file.exists(), "Empty file should exist"
            assert target_file.read_bytes() == b"", "File should be empty"

    def test_atomic_write_handles_binary_content(self):
        """
        Test that atomic_write() correctly handles binary content.

        **Validates: Requirements 5.5**
        """
        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)

            target_file = temp_base_dir / "binary_file.bin"
            content = bytes(range(256))

            handler.atomic_write(target_file, content)

            assert target_file.read_bytes() == content

    def test_atomic_write_rollback_on_write_failure(self):
        """
        Test that atomic_write() cleans up temp files on write failure.

        **Validates: Requirements 5.5**
        """


        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)

            target_file = temp_base_dir / "failed_write.txt"
            content = b"This should not be written"

            with mock.patch("os.write", side_effect=OSError("Simulated write failure")):
                with pytest.raises(OSError) as exc_info:
                    handler.atomic_write(target_file, content)

                assert "Simulated write failure" in str(exc_info.value)

            assert not target_file.exists()

            temp_files = list(temp_base_dir.glob(".tmp_*"))
            assert len(temp_files) == 0, (
                f"No temp files should remain after failed write. Found: {temp_files}"
            )

    def test_atomic_write_rollback_on_rename_failure(self):
        """
        Test that atomic_write() cleans up temp files on rename failure.

        **Validates: Requirements 5.5**
        """


        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)

            target_file = temp_base_dir / "rename_failed.txt"
            content = b"Content that won't be renamed"

            with mock.patch("os.replace", side_effect=OSError("Simulated rename failure")):
                with pytest.raises(OSError) as exc_info:
                    handler.atomic_write(target_file, content)

                assert "Simulated rename failure" in str(exc_info.value)

            assert not target_file.exists()

            temp_files = list(temp_base_dir.glob(".tmp_*"))
            assert len(temp_files) == 0, (
                f"No temp files should remain after failed rename. Found: {temp_files}"
            )

    def test_atomic_write_preserves_original_on_failure(self):
        """
        Test that atomic_write() preserves the original file if write fails.

        **Validates: Requirements 5.5**
        """


        with temp_base_directory() as temp_base_dir:
            handler = SafePathHandler(temp_base_dir)

            target_file = temp_base_dir / "preserve_original.txt"
            original_content = b"Original content that should be preserved"
            new_content = b"New content that fails to write"

            target_file.write_bytes(original_content)

            with mock.patch("os.replace", side_effect=OSError("Simulated failure")):
                with pytest.raises(OSError):
                    handler.atomic_write(target_file, new_content)

            assert target_file.exists(), "Original file should still exist"
            assert target_file.read_bytes() == original_content, (
                "Original content should be preserved after failed write"
            )
