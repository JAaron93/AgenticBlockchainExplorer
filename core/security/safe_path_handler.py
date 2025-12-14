"""Safe file path handling for secure file operations.

This module provides the SafePathHandler class that ensures file paths
stay within allowed directories and prevents path traversal attacks.

Requirements: 5.1, 5.2, 5.3, 5.5
"""

import os
import re
import tempfile
from pathlib import Path
from typing import Optional, Union


class PathTraversalError(Exception):
    """Raised when path traversal is detected.

    This error indicates an attempt to access a path outside the
    configured base directory, which could be a security attack.
    """

    def __init__(self, message: str, attempted_path: Optional[str] = None):
        """Initialize the error.

        Args:
            message: Human-readable error message.
            attempted_path: The path attempted (sanitized for logging).
        """
        super().__init__(message)
        self.message = message
        self.attempted_path = attempted_path


class InvalidFilenameError(Exception):
    """Raised when a filename contains invalid characters.

    This error indicates a filename that cannot be safely used
    in the filesystem.
    """

    def __init__(self, message: str):
        """Initialize the error.

        Args:
            message: Human-readable error message.
        """
        super().__init__(message)
        self.message = message


class SafePathHandler:
    """Handles file paths safely, preventing directory traversal.

    This class provides methods to safely construct file paths,
    sanitize filenames, and perform atomic file writes while
    ensuring all operations stay within a configured base directory.

    Requirements: 5.1, 5.2, 5.3, 5.5
    """

    # Pattern for unsafe characters in filenames
    # Includes: < > : " / \ | ? * and control characters (0x00-0x1f)
    UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

    # Pattern for path traversal sequences
    PATH_TRAVERSAL_PATTERN = re.compile(r"(?:^|[/\\])\.\.(?:[/\\]|$)")

    # Maximum filename length (common filesystem limit)
    MAX_FILENAME_LENGTH = 255

    def __init__(self, base_directory: Union[str, Path]):
        """Initialize with base directory constraint.

        Args:
            base_directory: The root directory that all paths must be within.
                           This directory must exist and be writable.

        Raises:
            ValueError: If base_directory doesn't exist or isn't a directory.
            PermissionError: If base_directory is not writable.

        Requirements: 5.4
        """
        self._base_directory = Path(base_directory).resolve()

        # Verify base directory exists and is a directory
        if not self._base_directory.exists():
            raise ValueError(
                f"Base directory does not exist: {self._base_directory}"
            )
        if not self._base_directory.is_dir():
            raise ValueError(
                f"Base path is not a directory: {self._base_directory}"
            )

        # Verify base directory is writable
        if not os.access(self._base_directory, os.W_OK):
            raise PermissionError(
                f"Base directory is not writable: {self._base_directory}"
            )

    @property
    def base_directory(self) -> Path:
        """Get the base directory.

        Returns:
            The resolved base directory path.
        """
        return self._base_directory

    def safe_join(self, *parts: str) -> Path:
        """Safely join path parts, raising error if result escapes base.

        Constructs a path from the given parts and verifies the resulting
        path is within the base directory. This prevents path traversal
        attacks using sequences like '../'.

        Args:
            *parts: Path components to join with the base directory.

        Returns:
            The resolved path within the base directory.

        Raises:
            PathTraversalError: If resulting path escapes base directory.

        Requirements: 5.1, 5.2
        """
        if not parts:
            return self._base_directory

        # Check each part for path traversal sequences
        for part in parts:
            if self._contains_traversal(part):
                raise PathTraversalError(
                    "Path traversal sequence detected in path component",
                    attempted_path="[REDACTED]",
                )

        # Construct the full path
        try:
            # Join parts with base directory
            joined_path = self._base_directory.joinpath(*parts)

            # Resolve to absolute path (resolves symlinks and ..)
            resolved_path = joined_path.resolve()

            # Verify the resolved path is within base directory
            if not self.validate_path(resolved_path):
                raise PathTraversalError(
                    "Constructed path escapes base directory",
                    attempted_path="[REDACTED]",
                )

            return resolved_path

        except (ValueError, OSError) as e:
            raise PathTraversalError(
                f"Invalid path construction: {type(e).__name__}",
                attempted_path="[REDACTED]",
            )

    def validate_path(self, path: Union[str, Path]) -> bool:
        """Check if path is within base directory.

        Resolves the path to an absolute path and verifies it starts
        with the base directory path.

        Args:
            path: The path to validate.

        Returns:
            True if the path is within the base directory, False otherwise.

        Requirements: 5.1, 5.2
        """
        try:
            resolved = Path(path).resolve()

            # Check if resolved path starts with base directory
            # Using is_relative_to for Python 3.9+ compatibility
            try:
                resolved.relative_to(self._base_directory)
                return True
            except ValueError:
                return False

        except (ValueError, OSError):
            return False

    def sanitize_filename(self, filename: str) -> str:
        """Remove unsafe characters from filename.

        Sanitizes a filename by:
        1. Removing path traversal sequences (../, ..\\)
        2. Removing unsafe characters (< > : " / \\ | ? * and control chars)
        3. Removing leading/trailing whitespace and dots
        4. Truncating to maximum filename length
        5. Ensuring the result is not empty

        Args:
            filename: The filename to sanitize.

        Returns:
            A sanitized filename safe for filesystem use.

        Raises:
            InvalidFilenameError: If filename can't be sanitized to valid name.

        Requirements: 5.3
        """
        if not filename:
            raise InvalidFilenameError("Filename cannot be empty")

        # Remove null bytes first (critical security issue)
        sanitized = filename.replace("\x00", "")

        # Remove path traversal sequences before other processing
        # This handles ../ and ..\ patterns
        while ".." in sanitized:
            sanitized = sanitized.replace("..", "")

        # Remove path separators
        # Replace path separators with underscores to preserve some meaning
        sanitized = sanitized.replace("/", "_").replace("\\", "_")

        # Remove remaining unsafe characters
        sanitized = self.UNSAFE_CHARS.sub("", sanitized)

        # Remove leading/trailing whitespace and dots
        # Leading dots can hide files on Unix systems
        sanitized = sanitized.strip().strip(".")

        # Truncate to maximum length
        if len(sanitized) > self.MAX_FILENAME_LENGTH:
            # Try to preserve file extension
            if "." in sanitized:
                name, ext = sanitized.rsplit(".", 1)
                max_name_len = self.MAX_FILENAME_LENGTH - len(ext) - 1
                if max_name_len > 0:
                    sanitized = f"{name[:max_name_len]}.{ext}"
                else:
                    sanitized = sanitized[: self.MAX_FILENAME_LENGTH]
            else:
                sanitized = sanitized[: self.MAX_FILENAME_LENGTH]

        # Ensure result is not empty
        if not sanitized:
            raise InvalidFilenameError(
                "Filename contains only unsafe characters"
            )

        return sanitized

    def atomic_write(
        self, path: Union[str, Path], content: bytes, mode: int = 0o644
    ) -> None:
        """Write content atomically using temp file and rename.

        Performs an atomic write operation by:
        1. Writing content to a temporary file in the same directory
        2. Syncing the file to disk
        3. Atomically renaming the temp file to the target path

        This ensures that the target file is never in a partial state,
        even if the process is interrupted during the write.

        Args:
            path: The target file path (must be within base directory).
            content: The bytes content to write.
            mode: File permission mode (default: 0o644).

        Raises:
            PathTraversalError: If the path is outside the base directory.
            OSError: If the write operation fails.

        Requirements: 5.5
        """
        target_path = Path(path)

        # Resolve and validate the target path
        if not target_path.is_absolute():
            target_path = self._base_directory / target_path

        resolved_target = target_path.resolve()

        if not self.validate_path(resolved_target):
            raise PathTraversalError(
                "Target path escapes base directory",
                attempted_path="[REDACTED]"
            )

        # Ensure parent directory exists
        parent_dir = resolved_target.parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        # Create temp file in the same directory for atomic rename
        # Using the same directory ensures we're on the same filesystem
        fd = None
        temp_path = None

        try:
            # Create temporary file
            fd, temp_path = tempfile.mkstemp(
                dir=str(parent_dir), prefix=".tmp_", suffix="_atomic"
            )

            # Write content
            os.write(fd, content)

            # Sync to disk
            os.fsync(fd)

            # Close the file descriptor
            os.close(fd)
            fd = None

            # Set permissions
            os.chmod(temp_path, mode)

            # Atomic rename
            os.replace(temp_path, resolved_target)
            temp_path = None

        except Exception:
            # Clean up on failure
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            raise

    def _contains_traversal(self, path_part: str) -> bool:
        """Check if a path part contains traversal sequences.

        Args:
            path_part: A single path component to check.

        Returns:
            True if the path part contains traversal sequences.
        """
        # Check for explicit '..' component
        if path_part == "..":
            return True

        # Check for traversal patterns
        if self.PATH_TRAVERSAL_PATTERN.search(path_part):
            return True

        # Check for null bytes (can be used to bypass checks)
        if "\x00" in path_part:
            return True

        return False
