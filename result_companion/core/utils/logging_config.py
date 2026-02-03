import json
import logging
import os
import tempfile
from logging.handlers import RotatingFileHandler
from typing import Dict

from tqdm import tqdm


class JsonFormatter(logging.Formatter):
    """Formats log records as JSON."""

    STANDARD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)

    def format(self, record: logging.LogRecord) -> str:
        """Formats a log record as JSON, including extra fields."""
        log_data = {
            "timestamp": self.formatTime(record),
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key not in self.STANDARD_ATTRS:
                log_data[key] = value

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that displays logs above tqdm progress bars."""

    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level)
        self.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
            self.flush()
        except Exception:
            self.handleError(record)


class LoggerRegistry:
    """Registry for managing loggers with tqdm-compatible output."""

    def __init__(self, default_log_level: int = logging.INFO):
        self.loggers: Dict[str, logging.Logger] = {}
        self.default_log_level: int = default_log_level
        self._tqdm_handler = TqdmLoggingHandler(level=default_log_level)

    def get_logger(self, name: str, use_tqdm: bool = True) -> logging.Logger:
        """Get or create a logger by name."""
        if name in self.loggers:
            return self.loggers[name]

        logger = _add_file_handler(name)

        if use_tqdm and not any(
            isinstance(h, TqdmLoggingHandler) for h in logger.handlers
        ):
            logger.addHandler(self._tqdm_handler)

        self.loggers[name] = logger
        return logger

    def set_log_level(self, level: str | int) -> None:
        """Set log level for console output only. File logging always captures DEBUG."""
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)

        self.default_log_level = level
        self._tqdm_handler.setLevel(level)


def _add_file_handler(name: str) -> logging.Logger:
    """Adds JSON file handler to logger. Returns logger set to DEBUG level."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        return logger

    log_file_path = os.path.join(tempfile.gettempdir(), "result_companion.log")
    try:
        file_handler = RotatingFileHandler(
            log_file_path, maxBytes=5 * 1024 * 1024, backupCount=3
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JsonFormatter())
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


# Default logger
logger = get_progress_logger("RC")
