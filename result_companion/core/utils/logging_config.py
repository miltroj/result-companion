import json
import logging
import logging.config
import os
import sys
import tempfile
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional


class LoggerRegistry:
    """
    A registry for managing loggers consistently across the application.
    This provides a single point for managing log levels and configuration.
    It leverages the existing setup_logging function for consistent logger setup.
    """

    def __init__(self, default_log_level: int = logging.INFO):
        """Initialize the registry with a default log level."""
        self.loggers: Dict[str, logging.Logger] = {}
        self.default_log_level: int = default_log_level
        self.custom_handlers: Dict[str, logging.Handler] = {}
        self.log_filename: str = "result_companion.log"
        self.log_dir: Optional[str] = None
        self.max_log_size: int = 5 * 1024 * 1024
        self.backup_count: int = 3
        self.enable_json: bool = False

    def register_handler(self, handler_name: str, handler: logging.Handler) -> None:
        """Register a custom handler that can be applied to loggers."""
        self.custom_handlers[handler_name] = handler

    def get_logger(
        self, name: str, use_handlers: Optional[list[str]] = None
    ) -> logging.Logger:
        """
        Get a logger from the registry, creating it if it doesn't exist.

        Args:
            name: Name of the logger
            use_handlers: List of handler names to apply to the logger

        Returns:
            The configured logger
        """
        if name in self.loggers:
            return self.loggers[name]

        logger = setup_logging(
            name,
            log_level=self.default_log_level,
            log_filename=self.log_filename,
            log_dir=self.log_dir,
            max_log_size=self.max_log_size,
            backup_count=self.backup_count,
            enable_json=self.enable_json,
        )

        # Apply any additional custom handlers
        if use_handlers:
            for handler_name in use_handlers:
                if handler_name in self.custom_handlers:
                    # Check if a similar handler is already attached
                    if not any(
                        isinstance(h, type(self.custom_handlers[handler_name]))
                        for h in logger.handlers
                    ):
                        logger.addHandler(self.custom_handlers[handler_name])

        self.loggers[name] = logger
        return logger

    def set_log_level(self, level: str | int) -> None:
        """
        Set the log level for all loggers in the registry.

        Args:
            level: Log level to set (can be string or int level)
        """
        # Convert string to log level if needed
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)

        self.default_log_level = level

        # Update all registered loggers
        for logger_name, logger_instance in self.loggers.items():
            logger_instance.setLevel(level)
            for handler in logger_instance.handlers:
                handler.setLevel(level)

    def configure_file_logging(
        self,
        log_filename: Optional[str] = None,
        log_dir: Optional[str] = None,
        max_log_size: Optional[int] = None,
        backup_count: Optional[int] = None,
        enable_json: Optional[bool] = None,
    ) -> None:
        """
        Configure file logging parameters for future loggers.

        Args:
            log_filename: The name of the log file
            log_dir: Directory where log files will be saved
            max_log_size: Max size of a log file before rotating
            backup_count: Number of rotated log files to keep
            enable_json: Whether to use JSON formatting
        """
        if log_filename is not None:
            self.log_filename = log_filename
        if log_dir is not None:
            self.log_dir = log_dir
        if max_log_size is not None:
            self.max_log_size = max_log_size
        if backup_count is not None:
            self.backup_count = backup_count
        if enable_json is not None:
            self.enable_json = enable_json


# Create a single instance of the registry
logger_registry = LoggerRegistry()


def setup_logging(
    name,
    log_level=logging.INFO,
    log_filename="result_companion.log",
    log_dir=None,
    max_log_size=5 * 1024 * 1024,
    backup_count=3,
    enable_json=False,
):
    """
    Configures advanced logging for CLI applications.
    Outputs to both stdout and a log file, with optional JSON formatting.

    Parameters:
    - name: logger name
    - log_level: The logging level (e.g., logging.DEBUG, logging.INFO).
    - log_filename: The name of the log file.
    - log_dir: Directory where the log file will be saved. Defaults to the system's temp dir.
    - max_log_size: Max size of a single log file before rotating (in bytes).
    - backup_count: Number of rotated log files to keep.
    - enable_json: If True, logs will be formatted as JSON.
    """
    # Determine log directory
    if log_dir is None:
        log_dir = tempfile.gettempdir()
    log_file_path = os.path.join(log_dir, log_filename)

    # Define formatters
    plain_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    json_formatter = logging.Formatter(
        json.dumps(
            {
                "timestamp": "%(asctime)s",
                "logger": "%(name)s",
                "level": "%(levelname)s",
                "message": "%(message)s",
            }
        )
    )

    # Choose formatter based on user preference
    formatter = json_formatter if enable_json else plain_formatter

    # Create custom logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate handlers
    if logger.hasHandlers():
        return logger

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating File Handler
    try:
        file_handler = RotatingFileHandler(
            log_file_path, maxBytes=max_log_size, backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)  # TODO: not working
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)
    except (OSError, IOError) as e:
        logger.warning(f"Failed to write to log file {log_file_path}: {e}")

    # Log setup details
    logger.debug(f"Logging initialized. Log file: {log_file_path}")
    return logger


class TqdmLoggingHandler(logging.Handler):
    """
    Custom logging handler that ensures logs are displayed above the tqdm progress bar.
    It writes directly to sys.stdout (bypassing the progress bar).
    """

    def __init__(self, level=logging.NOTSET):
        """Initialize with the given log level."""
        super().__init__(level)
        self.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    def emit(self, record):
        try:
            from tqdm import tqdm

            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


tqdm_handler = TqdmLoggingHandler()
logger_registry.register_handler("tqdm", tqdm_handler)


def log_uncaught_exceptions(logger) -> None:
    """
    Logs any uncaught exceptions globally for better diagnostics.
    """

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical(
            "Uncaught Exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = handle_exception


def set_global_log_level(log_level) -> None:
    """
    Set log level for all loggers.

    Args:
        log_level: The log level to set (can be string or int level)
    """
    # Use the registry to set log level for all loggers
    logger_registry.set_log_level(log_level)


def get_progress_logger(name="RC") -> logging.Logger:
    """
    Get a logger that works with progress bars.

    Args:
        name: Name of the logger

    Returns:
        A logger configured to work with progress bars
    """
    return logger_registry.get_logger(name, use_handlers=["tqdm"])


# Create a default logger - maintains backward compatibility
logger = get_progress_logger("RC")
