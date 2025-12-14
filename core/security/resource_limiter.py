"""Resource consumption limiting for protection against exhaustion attacks.

This module provides the ResourceLimiter class that monitors and enforces
limits on response sizes, file sizes, memory usage, and CPU time.

Requirements: 3.1, 3.2, 3.5, 3.6
"""

import re
import resource
from typing import Optional, Pattern

import psutil

from config.models import ResourceLimitConfig


class ResourceLimitError(Exception):
    """Base class for resource limit errors."""

    pass


class ResponseTooLargeError(ResourceLimitError):
    """Raised when response exceeds size limit."""

    def __init__(self, size: int, limit: int):
        self.size = size
        self.limit = limit
        super().__init__(
            f"Response size {size} bytes exceeds limit of {limit} bytes"
        )


class FileTooLargeError(ResourceLimitError):
    """Raised when file exceeds size limit."""

    def __init__(self, size: int, limit: int):
        self.size = size
        self.limit = limit
        super().__init__(
            f"File size {size} bytes exceeds limit of {limit} bytes"
        )


class MemoryLimitExceededError(ResourceLimitError):
    """Raised when memory usage exceeds limit."""

    def __init__(self, usage_mb: float, limit_mb: int):
        self.usage_mb = usage_mb
        self.limit_mb = limit_mb
        super().__init__(
            f"Memory usage {usage_mb:.1f}MB exceeds limit of {limit_mb}MB"
        )


class CPUTimeLimitExceededError(ResourceLimitError):
    """Raised when CPU time exceeds limit."""

    def __init__(self, usage_seconds: float, limit_seconds: int):
        self.usage_seconds = usage_seconds
        self.limit_seconds = limit_seconds
        super().__init__(
            f"CPU time {usage_seconds:.1f}s exceeds limit of {limit_seconds}s"
        )


class ResourceLimiter:
    """Monitors and enforces resource consumption limits.

    This class provides methods to check various resource limits including
    response sizes, file sizes, memory usage, and CPU time. It also provides
    safe regex matching with input size limits to prevent ReDoS attacks.

    Requirements: 3.1, 3.2, 3.5, 3.6
    """

    # Maximum input size for regex operations to prevent ReDoS
    MAX_REGEX_INPUT_SIZE = 10000  # characters

    def __init__(self, config: Optional[ResourceLimitConfig] = None):
        """Initialize with configured limits.

        Args:
            config: Configuration for resource limits.
                   If None, uses default configuration.
        """
        self._config = config or ResourceLimitConfig()

    @property
    def max_response_size(self) -> int:
        """Get maximum response size in bytes."""
        return self._config.max_response_size_bytes

    @property
    def max_file_size(self) -> int:
        """Get maximum file size in bytes."""
        return self._config.max_output_file_size_bytes

    @property
    def max_memory_mb(self) -> int:
        """Get maximum memory usage in MB."""
        return self._config.max_memory_usage_mb

    @property
    def max_cpu_seconds(self) -> int:
        """Get maximum CPU time in seconds."""
        return self._config.max_cpu_time_seconds

    def check_response_size(self, size: int) -> None:
        """Check if response size is within limit.

        Args:
            size: Response size in bytes.

        Raises:
            ResponseTooLargeError: If response exceeds size limit.

        Requirements: 3.1, 3.2
        """
        if size > self._config.max_response_size_bytes:
            raise ResponseTooLargeError(
                size=size,
                limit=self._config.max_response_size_bytes,
            )

    def check_file_size(self, size: int) -> None:
        """Check if file size is within limit.

        Args:
            size: File size in bytes.

        Raises:
            FileTooLargeError: If file exceeds size limit.

        Requirements: 3.5
        """
        if size > self._config.max_output_file_size_bytes:
            raise FileTooLargeError(
                size=size,
                limit=self._config.max_output_file_size_bytes,
            )

    def check_memory_usage(self) -> None:
        """Check if current memory usage is within limit.

        Raises:
            MemoryLimitExceededError: If memory usage exceeds limit.

        Requirements: 3.6
        """
        current_mb = self.get_current_memory_mb()
        if current_mb > self._config.max_memory_usage_mb:
            raise MemoryLimitExceededError(
                usage_mb=current_mb,
                limit_mb=self._config.max_memory_usage_mb,
            )

    def check_cpu_usage(self) -> None:
        """Check if CPU time is within limit.

        Uses resource.getrusage() on Unix to check process CPU time.
        Fails fast if CPU time exceeds configured threshold.

        Raises:
            CPUTimeLimitExceededError: If CPU time exceeds limit.
        """
        current_seconds = self.get_current_cpu_seconds()
        if current_seconds > self._config.max_cpu_time_seconds:
            raise CPUTimeLimitExceededError(
                usage_seconds=current_seconds,
                limit_seconds=self._config.max_cpu_time_seconds,
            )

    def get_current_memory_mb(self) -> float:
        """Get current process memory usage in MB.

        Returns:
            Current memory usage in megabytes.
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)

    def get_current_cpu_seconds(self) -> float:
        """Get current process CPU time in seconds.

        Returns:
            Total CPU time (user + system) in seconds.
        """
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_utime + usage.ru_stime

    def is_approaching_memory_limit(self, threshold_percent: float = 0.9) -> bool:
        """Check if memory usage is approaching the limit.

        Args:
            threshold_percent: Percentage of limit to consider "approaching".
                             Default is 90%.

        Returns:
            True if memory usage exceeds threshold percentage of limit.
        """
        current_mb = self.get_current_memory_mb()
        threshold_mb = self._config.max_memory_usage_mb * threshold_percent
        return current_mb >= threshold_mb

    def is_approaching_cpu_limit(self, threshold_percent: float = 0.9) -> bool:
        """Check if CPU time is approaching the limit.

        Args:
            threshold_percent: Percentage of limit to consider "approaching".
                             Default is 90%.

        Returns:
            True if CPU time exceeds threshold percentage of limit.
        """
        current_seconds = self.get_current_cpu_seconds()
        threshold_seconds = self._config.max_cpu_time_seconds * threshold_percent
        return current_seconds >= threshold_seconds

    @staticmethod
    def safe_regex_match(
        pattern: Pattern[str],
        text: str,
        max_input_size: int = MAX_REGEX_INPUT_SIZE,
    ) -> Optional[re.Match[str]]:
        """Safely match regex with input size limit to prevent ReDoS.

        This method limits the input size before applying regex to prevent
        Regular Expression Denial of Service (ReDoS) attacks.

        Args:
            pattern: Pre-compiled regex pattern. Must be:
                    - Pre-compiled at module load time
                    - Anchored (^ and $) where appropriate
                    - Free of catastrophic backtracking patterns
            text: Input text to match.
            max_input_size: Maximum input size to process.

        Returns:
            Match object or None if no match or input too large.

        Note:
            All regex patterns used in security components MUST be:
            - Pre-compiled at module load time
            - Anchored (^ and $) where appropriate
            - Free of catastrophic backtracking patterns (nested quantifiers)
            - Tested against ReDoS attack patterns
        """
        if not text:
            return None

        # Limit input size to prevent ReDoS
        if len(text) > max_input_size:
            return None

        return pattern.match(text)

    @staticmethod
    def safe_regex_fullmatch(
        pattern: Pattern[str],
        text: str,
        max_input_size: int = MAX_REGEX_INPUT_SIZE,
    ) -> Optional[re.Match[str]]:
        """Safely fullmatch regex with input size limit to prevent ReDoS.

        Similar to safe_regex_match but uses fullmatch instead of match.

        Args:
            pattern: Pre-compiled regex pattern.
            text: Input text to match.
            max_input_size: Maximum input size to process.

        Returns:
            Match object or None if no match or input too large.
        """
        if not text:
            return None

        # Limit input size to prevent ReDoS
        if len(text) > max_input_size:
            return None

        return pattern.fullmatch(text)
