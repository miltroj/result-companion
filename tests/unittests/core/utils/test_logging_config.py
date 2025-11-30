"""Tests for logging configuration utilities."""

import logging

from result_companion.core.utils.logging_config import (
    LoggerRegistry,
    TqdmLoggingHandler,
    get_progress_logger,
    set_global_log_level,
    setup_logging,
)

# --- LoggerRegistry Tests ---


def test_logger_registry_creates_logger_with_specified_level():
    registry = LoggerRegistry(default_log_level=logging.WARNING)

    logger = registry.get_logger("test_reg_1")

    assert logger.level == logging.WARNING


def test_logger_registry_caches_loggers():
    registry = LoggerRegistry()

    logger1 = registry.get_logger("test_reg_2")
    logger2 = registry.get_logger("test_reg_2")

    assert logger1 is logger2


def test_logger_registry_set_log_level_updates_all_loggers():
    registry = LoggerRegistry(default_log_level=logging.INFO)
    logger1 = registry.get_logger("test_reg_3")
    logger2 = registry.get_logger("test_reg_4")

    registry.set_log_level(logging.DEBUG)

    assert logger1.level == logging.DEBUG
    assert logger2.level == logging.DEBUG


def test_logger_registry_set_log_level_accepts_string():
    registry = LoggerRegistry()

    registry.set_log_level("ERROR")

    assert registry.default_log_level == logging.ERROR


def test_logger_registry_applies_custom_handlers():
    registry = LoggerRegistry()
    custom_handler = logging.StreamHandler()
    registry.register_handler("custom", custom_handler)

    logger = registry.get_logger("test_reg_5", use_handlers=["custom"])

    assert custom_handler in logger.handlers


# --- TqdmLoggingHandler Tests ---


def test_tqdm_handler_emits_message(monkeypatch):
    """Handler should use tqdm.write for output."""
    written_messages = []

    import tqdm

    monkeypatch.setattr(
        tqdm.tqdm, "write", staticmethod(lambda m: written_messages.append(m))
    )

    handler = TqdmLoggingHandler()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Hello from tqdm",
        args=(),
        exc_info=None,
    )

    handler.emit(record)

    assert len(written_messages) == 1
    assert "Hello from tqdm" in written_messages[0]


# --- setup_logging Tests ---


def test_setup_logging_creates_logger_with_correct_level():
    logger = setup_logging("test_setup_1", log_level=logging.DEBUG)

    assert logger.name == "test_setup_1"
    assert logger.level == logging.DEBUG


def test_setup_logging_prevents_duplicate_handlers():
    name = "test_setup_2"

    logger1 = setup_logging(name)
    count_before = len(logger1.handlers)

    setup_logging(name)  # Call again
    count_after = len(logger1.handlers)

    assert count_before == count_after


# --- get_progress_logger Tests ---


def test_get_progress_logger_includes_tqdm_handler():
    logger = get_progress_logger("test_progress")

    handler_types = [type(h).__name__ for h in logger.handlers]

    assert "TqdmLoggingHandler" in handler_types


# --- set_global_log_level Tests ---


def test_set_global_log_level_updates_registry():
    from result_companion.core.utils.logging_config import logger_registry

    original = logger_registry.default_log_level

    set_global_log_level("CRITICAL")

    assert logger_registry.default_log_level == logging.CRITICAL

    # Cleanup
    logger_registry.set_log_level(original)
