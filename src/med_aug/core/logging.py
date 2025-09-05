"""Centralized logging configuration for the medication augmentation system."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import structlog
from structlog import get_logger
from rich.console import Console
from rich.logging import RichHandler

# Create logs directory if it doesn't exist
LOGS_DIR = Path.home() / ".med_aug" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Console for rich output
console = Console()


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    json_logs: bool = False,
    include_timestamp: bool = True,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Configure application-wide logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        json_logs: Whether to output JSON formatted logs
        include_timestamp: Whether to include timestamps in logs
        context: Optional context to bind to all logs
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure timestamp
    timestamper = (
        structlog.processors.TimeStamper(fmt="iso") if include_timestamp else None
    )

    # Base processors
    processors = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if timestamper:
        processors.insert(0, timestamper)

    # Add context if provided
    if context:
        processors.append(
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                ]
            )
        )

    # Configure output format
    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    handlers = []

    # Console handler with Rich
    console_handler = RichHandler(
        console=console, rich_tracebacks=True, tracebacks_show_locals=level == "DEBUG"
    )
    console_handler.setLevel(numeric_level)
    handlers.append(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        if json_logs:
            formatter = logging.Formatter("%(message)s")
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(level=numeric_level, handlers=handlers, force=True)

    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Bind global context if provided
    if context:
        structlog.contextvars.bind_contextvars(**context)


def get_logger(name: Optional[str] = None, **context) -> structlog.BoundLogger:
    """
    Get a logger instance with optional context.

    Args:
        name: Logger name (defaults to module name)
        **context: Additional context to bind to this logger

    Returns:
        Configured logger instance
    """
    logger = structlog.get_logger(name)
    if context:
        logger = logger.bind(**context)
    return logger


class LogContext:
    """Context manager for temporary log context."""

    def __init__(self, **kwargs):
        """Initialize with context variables."""
        self.context = kwargs
        self.tokens = []

    def __enter__(self):
        """Bind context on entry."""
        self.tokens = structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Unbind context on exit."""
        structlog.contextvars.unbind_contextvars(*self.tokens)


class PerformanceLogger:
    """Logger for performance metrics."""

    def __init__(self, logger: Optional[structlog.BoundLogger] = None):
        """Initialize performance logger."""
        self.logger = logger or get_logger("performance")
        self.timings = {}

    def start_operation(self, operation: str) -> float:
        """
        Start timing an operation.

        Args:
            operation: Operation name

        Returns:
            Start time
        """
        start_time = datetime.now().timestamp()
        self.timings[operation] = start_time
        self.logger.debug("operation_started", operation=operation)
        return start_time

    def end_operation(self, operation: str, **context) -> float:
        """
        End timing an operation and log the duration.

        Args:
            operation: Operation name
            **context: Additional context to log

        Returns:
            Duration in seconds
        """
        if operation not in self.timings:
            self.logger.warning("operation_not_started", operation=operation)
            return 0.0

        start_time = self.timings.pop(operation)
        end_time = datetime.now().timestamp()
        duration = end_time - start_time

        self.logger.info(
            "operation_completed",
            operation=operation,
            duration_seconds=duration,
            **context,
        )
        return duration

    def log_metrics(self, metrics: Dict[str, Any], **context):
        """
        Log performance metrics.

        Args:
            metrics: Dictionary of metrics
            **context: Additional context
        """
        self.logger.info("performance_metrics", metrics=metrics, **context)


class AuditLogger:
    """Logger for audit trail of data modifications."""

    def __init__(self, logger: Optional[structlog.BoundLogger] = None):
        """Initialize audit logger."""
        self.logger = logger or get_logger("audit")

    def log_data_access(
        self, resource: str, action: str, user: Optional[str] = None, **details
    ):
        """
        Log data access event.

        Args:
            resource: Resource being accessed
            action: Action performed
            user: User performing action
            **details: Additional details
        """
        self.logger.info(
            "data_access",
            resource=resource,
            action=action,
            user=user or "system",
            timestamp=datetime.now().isoformat(),
            **details,
        )

    def log_modification(
        self,
        resource: str,
        before: Any,
        after: Any,
        user: Optional[str] = None,
        **details,
    ):
        """
        Log data modification.

        Args:
            resource: Resource being modified
            before: State before modification
            after: State after modification
            user: User performing modification
            **details: Additional details
        """
        self.logger.info(
            "data_modification",
            resource=resource,
            before=before,
            after=after,
            user=user or "system",
            timestamp=datetime.now().isoformat(),
            **details,
        )


class ErrorLogger:
    """Enhanced error logging with context."""

    def __init__(self, logger: Optional[structlog.BoundLogger] = None):
        """Initialize error logger."""
        self.logger = logger or get_logger("errors")

    def log_error(self, error: Exception, operation: str, **context):
        """
        Log an error with full context.

        Args:
            error: Exception that occurred
            operation: Operation during which error occurred
            **context: Additional context
        """
        self.logger.error(
            "error_occurred",
            operation=operation,
            error_type=type(error).__name__,
            error_message=str(error),
            **context,
            exc_info=True,
        )

    def log_warning(self, message: str, operation: str, **context):
        """
        Log a warning.

        Args:
            message: Warning message
            operation: Operation during which warning occurred
            **context: Additional context
        """
        self.logger.warning("warning", message=message, operation=operation, **context)


def setup_file_rotation(
    log_file: Path,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,  # 10MB
) -> logging.Handler:
    """
    Setup rotating file handler.

    Args:
        log_file: Path to log file
        max_bytes: Maximum size before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured rotating file handler
    """
    from logging.handlers import RotatingFileHandler

    handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    return handler


# Convenience function for quick setup
def quick_setup(debug: bool = False, log_to_file: bool = True):
    """
    Quick logging setup with sensible defaults.

    Args:
        debug: Whether to enable debug logging
        log_to_file: Whether to log to file
    """
    level = "DEBUG" if debug else "INFO"

    log_file = None
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGS_DIR / f"med_aug_{timestamp}.log"

    setup_logging(
        level=level, log_file=log_file, json_logs=False, include_timestamp=True
    )

    logger = get_logger("med_aug")
    logger.info(
        "logging_initialized", level=level, log_file=str(log_file) if log_file else None
    )


# Export commonly used functions
__all__ = [
    "setup_logging",
    "get_logger",
    "LogContext",
    "PerformanceLogger",
    "AuditLogger",
    "ErrorLogger",
    "quick_setup",
    "console",
]
