"""Secure logging wrapper with automatic credential sanitization.

This module provides the SecureLogger class that wraps standard Python
loggers to automatically sanitize credentials from all log output.

Requirements: 1.3, 1.4
"""

import logging
import traceback
from typing import Any, Optional

from core.security.credential_sanitizer import CredentialSanitizer


class SecureLogger:
    """Logger wrapper that sanitizes credentials from all output.

    This class wraps a standard Python logger and automatically sanitizes
    all log messages, arguments, and exception stack traces to prevent
    credential leakage through logs.

    Requirements: 1.3, 1.4
    """

    def __init__(
        self,
        logger: logging.Logger,
        sanitizer: CredentialSanitizer,
    ):
        """Wrap an existing logger with sanitization.

        Args:
            logger: The underlying Python logger to wrap.
            sanitizer: CredentialSanitizer instance for redacting credentials.
        """
        self._logger = logger
        self._sanitizer = sanitizer

    @property
    def name(self) -> str:
        """Get the name of the underlying logger."""
        return self._logger.name

    @property
    def level(self) -> int:
        """Get the logging level of the underlying logger."""
        return self._logger.level

    def setLevel(self, level: int) -> None:
        """Set the logging level of the underlying logger."""
        self._logger.setLevel(level)

    def _sanitize_message(self, msg: str) -> str:
        """Sanitize a log message.

        Args:
            msg: The log message to sanitize.

        Returns:
            Sanitized message with credentials redacted.
        """
        return self._sanitizer.sanitize(str(msg))

    def _sanitize_args(self, args: tuple) -> tuple:
        """Sanitize log message arguments.

        Args:
            args: Tuple of arguments for string formatting.

        Returns:
            Tuple with sanitized arguments.
        """
        sanitized: list[Any] = []
        for arg in args:
            if isinstance(arg, str):
                sanitized.append(self._sanitizer.sanitize(arg))
            elif isinstance(arg, dict):
                sanitized.append(self._sanitizer.sanitize_dict(arg))
            else:
                # Convert to string and sanitize
                sanitized.append(self._sanitizer.sanitize(str(arg)))
        return tuple(sanitized)

    def _sanitize_kwargs(self, kwargs: dict) -> dict:
        """Sanitize keyword arguments for logging.

        Args:
            kwargs: Keyword arguments that may contain credentials.

        Returns:
            Dictionary with sanitized values.
        """
        result: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key == "extra" and isinstance(value, dict):
                # Sanitize extra dict for structured logging
                result[key] = self._sanitizer.sanitize_dict(value)
            elif key == "exc_info":
                # Pass through exc_info as-is (handled separately)
                result[key] = value
            elif key == "stack_info":
                # Sanitize stack info if it's a string
                if isinstance(value, str):
                    result[key] = self._sanitize_stack_trace(value)
                else:
                    result[key] = value
            elif isinstance(value, str):
                result[key] = self._sanitizer.sanitize(value)
            elif isinstance(value, dict):
                result[key] = self._sanitizer.sanitize_dict(value)
            else:
                result[key] = value
        return result

    def _sanitize_stack_trace(self, stack_trace: str) -> str:
        """Sanitize a stack trace to remove credential values.

        This method filters local variables from stack traces that may
        contain credential values.

        Args:
            stack_trace: The stack trace string to sanitize.

        Returns:
            Sanitized stack trace with credentials redacted.

        Requirements: 1.4
        """
        if not stack_trace:
            return stack_trace

        # Apply general sanitization to catch credential patterns
        return self._sanitizer.sanitize(stack_trace)

    def _format_exception_info(
        self,
        exc_info: Optional[tuple] = None,
    ) -> Optional[str]:
        """Format and sanitize exception information.

        Args:
            exc_info: Exception info tuple (type, value, traceback).

        Returns:
            Sanitized exception string or None.

        Requirements: 1.4
        """
        if not exc_info:
            return None

        try:
            # Format the exception
            formatted = "".join(traceback.format_exception(*exc_info))
            # Sanitize the formatted traceback
            return self._sanitize_stack_trace(formatted)
        except Exception:
            return "[Error formatting exception]"

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log debug message with sanitization.

        Args:
            msg: Log message (may contain format placeholders).
            *args: Arguments for string formatting.
            **kwargs: Additional keyword arguments for logging.
        """
        sanitized_msg = self._sanitize_message(msg)
        sanitized_args = self._sanitize_args(args)
        sanitized_kwargs = self._sanitize_kwargs(kwargs)
        self._logger.debug(sanitized_msg, *sanitized_args, **sanitized_kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log info message with sanitization.

        Args:
            msg: Log message (may contain format placeholders).
            *args: Arguments for string formatting.
            **kwargs: Additional keyword arguments for logging.

        Requirements: 1.3
        """
        sanitized_msg = self._sanitize_message(msg)
        sanitized_args = self._sanitize_args(args)
        sanitized_kwargs = self._sanitize_kwargs(kwargs)
        self._logger.info(sanitized_msg, *sanitized_args, **sanitized_kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log warning message with sanitization.

        Args:
            msg: Log message (may contain format placeholders).
            *args: Arguments for string formatting.
            **kwargs: Additional keyword arguments for logging.

        Requirements: 1.3
        """
        sanitized_msg = self._sanitize_message(msg)
        sanitized_args = self._sanitize_args(args)
        sanitized_kwargs = self._sanitize_kwargs(kwargs)
        self._logger.warning(sanitized_msg, *sanitized_args, **sanitized_kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log error message with sanitization.

        Args:
            msg: Log message (may contain format placeholders).
            *args: Arguments for string formatting.
            **kwargs: Additional keyword arguments for logging.

        Requirements: 1.3
        """
        sanitized_msg = self._sanitize_message(msg)
        sanitized_args = self._sanitize_args(args)
        sanitized_kwargs = self._sanitize_kwargs(kwargs)
        self._logger.error(sanitized_msg, *sanitized_args, **sanitized_kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log critical message with sanitization.

        Args:
            msg: Log message (may contain format placeholders).
            *args: Arguments for string formatting.
            **kwargs: Additional keyword arguments for logging.
        """
        sanitized_msg = self._sanitize_message(msg)
        sanitized_args = self._sanitize_args(args)
        sanitized_kwargs = self._sanitize_kwargs(kwargs)
        self._logger.critical(sanitized_msg, *sanitized_args, **sanitized_kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log exception with stack trace sanitization.

        This method logs an error message along with the current exception
        information. The stack trace is sanitized to remove any credential
        values that may appear in local variables.

        Args:
            msg: Log message (may contain format placeholders).
            *args: Arguments for string formatting.
            **kwargs: Additional keyword arguments for logging.

        Requirements: 1.4
        """
        sanitized_msg = self._sanitize_message(msg)
        sanitized_args = self._sanitize_args(args)
        sanitized_kwargs = self._sanitize_kwargs(kwargs)

        import sys
        exc_info = kwargs.get("exc_info", True)
        if exc_info is True:
            exc_info = sys.exc_info()
        
        # Format and sanitize the exception
        sanitized_exc = self._format_exception_info(exc_info) if exc_info else None
        
        # Log error with sanitized traceback in message
        if sanitized_exc:
            full_msg = f"{sanitized_msg}\n{sanitized_exc}"
            self._logger.error(
                full_msg,
                *sanitized_args,
                **{k: v for k, v in sanitized_kwargs.items() if k != "exc_info"},
            )
        else:
            self._logger.error(
                sanitized_msg,
                *sanitized_args,
                **sanitized_kwargs,
            )

    def log(
        self,
        level: int,
        msg: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Log at specified level with sanitization.

        Args:
            level: Logging level (e.g., logging.INFO).
            msg: Log message (may contain format placeholders).
            *args: Arguments for string formatting.
            **kwargs: Additional keyword arguments for logging.
        """
        sanitized_msg = self._sanitize_message(msg)
        sanitized_args = self._sanitize_args(args)
        sanitized_kwargs = self._sanitize_kwargs(kwargs)
        self._logger.log(level, sanitized_msg, *sanitized_args, **sanitized_kwargs)

    def isEnabledFor(self, level: int) -> bool:
        """Check if logging is enabled for the specified level.

        Args:
            level: Logging level to check.

        Returns:
            True if logging is enabled for the level.
        """
        return self._logger.isEnabledFor(level)

    def getEffectiveLevel(self) -> int:
        """Get the effective logging level.

        Returns:
            The effective logging level.
        """
        return self._logger.getEffectiveLevel()

    def addHandler(self, handler: logging.Handler) -> None:
        """Add a handler to the underlying logger.

        Args:
            handler: Handler to add.
        """
        self._logger.addHandler(handler)

    def removeHandler(self, handler: logging.Handler) -> None:
        """Remove a handler from the underlying logger.

        Args:
            handler: Handler to remove.
        """
        self._logger.removeHandler(handler)
