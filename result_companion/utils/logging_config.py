import logging
import logging.config
import os
import sys
import tempfile
from logging.handlers import RotatingFileHandler
import json


def setup_logging(
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
        json.dumps({
            "timestamp": "%(asctime)s",
            "logger": "%(name)s",
            "level": "%(levelname)s",
            "message": "%(message)s",
        })
    )

    # Choose formatter based on user preference
    formatter = json_formatter if enable_json else plain_formatter

    # Create custom logger
    logger = logging.getLogger("RC")
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
        file_handler.setLevel(logging.DEBUG) # TODO: not working
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)
    except (OSError, IOError) as e:
        logger.warning(f"Failed to write to log file {log_file_path}: {e}")

    # Log setup details
    logger.info(f"Logging initialized. Log file: {log_file_path}")
    return logger


def log_uncaught_exceptions(logger) -> None:
    """
    Logs any uncaught exceptions globally for better diagnostics.
    """
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical("Uncaught Exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception
