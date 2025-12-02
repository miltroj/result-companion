"""Logging configuration utilities."""

import logging
import os
import sys
import tempfile
from logging.handlers import RotatingFileHandler
from typing import Dict


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that displays logs above tqdm progress bars."""

    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level)
        self.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from tqdm import tqdm

            tqdm.write(self.format(record))
            self.flush()
        except Exception:
            self.handleError(record)


class LoggerRegistry:
    """Registry for managing loggers with tqdm-compatible output."""

    def __init__(self, default_log_level: int = logging.INFO):
        self.loggers: Dict[str, logging.Logger] = {}
        self.default_log_level: int = default_log_level
        self._tqdm_handler = TqdmLoggingHandler()

    def get_logger(self, name: str, use_tqdm: bool = True) -> logging.Logger:
        """Get or create a logger by name."""
        if name in self.loggers:
            return self.loggers[name]

        logger = _setup_logging(name, log_level=self.default_log_level)

        if use_tqdm and not any(
            isinstance(h, TqdmLoggingHandler) for h in logger.handlers
        ):
            logger.addHandler(self._tqdm_handler)

        self.loggers[name] = logger
        return logger

    def set_log_level(self, level: str | int) -> None:
        """Set log level for all registered loggers."""
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)

        self.default_log_level = level
        for logger_instance in self.loggers.values():
            logger_instance.setLevel(level)
            for handler in logger_instance.handlers:
                handler.setLevel(level)


def _setup_logging(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """Create a logger with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file_path = os.path.join(tempfile.gettempdir(), "result_companion.log")
    try:
        file_handler = RotatingFileHandler(
            log_file_path, maxBytes=5 * 1024 * 1024, backupCount=3
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (OSError, IOError) as e:
        logger.warning(f"Failed to write to log file {log_file_path}: {e}")

    return logger


# Module-level singleton and helpers
logger_registry = LoggerRegistry()


def set_global_log_level(log_level: str | int) -> None:
    """Set log level for all loggers."""
    logger_registry.set_log_level(log_level)


def get_progress_logger(name: str = "RC") -> logging.Logger:
    """Get a logger that works with progress bars."""
    return logger_registry.get_logger(name)


def log_uncaught_exceptions(target_logger: logging.Logger) -> None:
    """Log uncaught exceptions globally."""

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        target_logger.critical(
            "Uncaught Exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = handle_exception


# Default logger for backward compatibility
logger = get_progress_logger("RC")
