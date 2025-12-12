"""Structured logging configuration for the blockchain stablecoin explorer.

Provides JSON-formatted logging with correlation IDs (run_id) for
request tracing across components.
"""

import logging
import sys
from contextvars import ContextVar
from typing import Any, Optional

from pythonjsonlogger import jsonlogger


# Context variable for storing the current run_id across async operations
_run_id_context: ContextVar[Optional[str]] = ContextVar("run_id", default=None)


def get_run_id() -> Optional[str]:
    """Get the current run_id from context.
    
    Returns:
        The current run_id or None if not set.
    """
    return _run_id_context.get()


def set_run_id(run_id: Optional[str]) -> None:
    """Set the run_id in context for the current async task.
    
    Args:
        run_id: The run_id to set, or None to clear.
    """
    _run_id_context.set(run_id)


class CorrelationIdFilter(logging.Filter):
    """Logging filter that adds correlation ID (run_id) to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add run_id to the log record if available.
        
        Args:
            record: The log record to modify.
            
        Returns:
            Always True to allow the record through.
        """
        # Get run_id from context or from record's extra dict
        run_id = get_run_id()
        if run_id is None:
            run_id = getattr(record, "run_id", None)
        
        record.run_id = run_id or "N/A"
        return True


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields for structured logging."""
    
    def __init__(
        self,
        *args: Any,
        service_name: str = "blockchain-stablecoin-explorer",
        **kwargs: Any
    ):
        """Initialize the formatter.
        
        Args:
            service_name: Name of the service for log identification.
            *args: Positional arguments for parent class.
            **kwargs: Keyword arguments for parent class.
        """
        super().__init__(*args, **kwargs)
        self.service_name = service_name
    
    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any]
    ) -> None:
        """Add custom fields to the log record.
        
        Args:
            log_record: The dict that will be serialized to JSON.
            record: The original LogRecord.
            message_dict: Dict from the log message if it was a dict.
        """
        super().add_fields(log_record, record, message_dict)
        
        # Rename and restructure fields
        log_record["time"] = log_record.pop("asctime", None) or self.formatTime(record)
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["service"] = self.service_name
        
        # Add run_id if available
        run_id = getattr(record, "run_id", None)
        if run_id and run_id != "N/A":
            log_record["run_id"] = run_id
        
        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        
        # Remove redundant fields
        log_record.pop("levelname", None)
        log_record.pop("name", None)



class TextFormatter(logging.Formatter):
    """Text formatter with correlation ID support for development."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with run_id if available.
        
        Args:
            record: The log record to format.
            
        Returns:
            Formatted log string.
        """
        run_id = getattr(record, "run_id", None)
        if run_id and run_id != "N/A":
            # Include run_id in the format
            original_msg = record.msg
            record.msg = f"[run_id={run_id}] {original_msg}"
            result = super().format(record)
            record.msg = original_msg
            return result
        return super().format(record)


def configure_logging(
    level: str = "INFO",
    fmt: str = "json",
    service_name: str = "blockchain-stablecoin-explorer"
) -> logging.Logger:
    """Configure structured logging for the application.
    
    Sets up logging with either JSON or text format, including
    correlation ID support for request tracing.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        fmt: Log format ('json' or 'text').
        service_name: Service name for log identification.
        
    Returns:
        The root logger configured with the specified settings.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Get root logger and clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    
    # Add correlation ID filter
    correlation_filter = CorrelationIdFilter()
    handler.addFilter(correlation_filter)
    
    # Set formatter based on format type
    if fmt == "json":
        formatter = CustomJsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            service_name=service_name
        )
    else:
        formatter = TextFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    This is a convenience function that returns a logger
    configured to work with the structured logging setup.
    
    Args:
        name: The logger name (typically __name__).
        
    Returns:
        A configured logger instance.
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for setting run_id during a block of code.
    
    Usage:
        with LogContext(run_id="abc-123"):
            logger.info("This log will include run_id")
    """
    
    def __init__(self, run_id: Optional[str] = None, **extra: Any):
        """Initialize the log context.
        
        Args:
            run_id: The run_id to set for this context.
            **extra: Additional context values (reserved for future use).
        """
        self.run_id = run_id
        self.extra = extra
        self._token: Any = None
    
    def __enter__(self) -> "LogContext":
        """Enter the context and set the run_id."""
        if self.run_id:
            self._token = _run_id_context.set(self.run_id)
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context and restore the previous run_id."""
        if self._token is not None:
            _run_id_context.reset(self._token)


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    run_id: Optional[str] = None,
    **extra: Any
) -> None:
    """Log a message with additional context.
    
    This is a helper function for logging with extra fields
    that will be included in the JSON output.
    
    Args:
        logger: The logger to use.
        level: The log level (e.g., logging.INFO).
        message: The log message.
        run_id: Optional run_id for correlation.
        **extra: Additional fields to include in the log.
    """
    if run_id:
        extra["run_id"] = run_id
    logger.log(level, message, extra=extra)
