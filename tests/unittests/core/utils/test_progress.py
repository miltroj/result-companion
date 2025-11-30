"""Tests for progress bar utilities."""

import asyncio
import logging

import pytest

from result_companion.core.utils.progress import (
    AsyncTaskProgress,
    ProgressLogger,
    get_current_progress_bar,
    run_tasks_with_progress,
)


class FakeLogger:
    """Simple fake logger that records all calls."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.handlers = []
        self.level = logging.INFO

    def debug(self, msg, *args, **kwargs):
        self.calls.append(("debug", msg))

    def info(self, msg, *args, **kwargs):
        self.calls.append(("info", msg))

    def warning(self, msg, *args, **kwargs):
        self.calls.append(("warning", msg))

    def error(self, msg, *args, **kwargs):
        self.calls.append(("error", msg))

    def critical(self, msg, *args, **kwargs):
        self.calls.append(("critical", msg))

    def setLevel(self, level):
        self.level = level


# --- ProgressLogger Tests ---


def test_progress_logger_delegates_to_underlying_logger():
    """ProgressLogger should forward all log calls to the underlying logger."""
    progress_logger = ProgressLogger("test_delegate")
    fake = FakeLogger()
    progress_logger.logger = fake

    progress_logger.debug("debug msg")
    progress_logger.info("info msg")
    progress_logger.warning("warning msg")
    progress_logger.error("error msg")
    progress_logger.critical("critical msg")

    assert ("debug", "debug msg") in fake.calls
    assert ("info", "info msg") in fake.calls
    assert ("warning", "warning msg") in fake.calls
    assert ("error", "error msg") in fake.calls
    assert ("critical", "critical msg") in fake.calls
    assert len(fake.calls) == 5


def test_progress_logger_set_level_accepts_string():
    progress_logger = ProgressLogger("test_level")

    progress_logger.set_level("DEBUG")

    assert progress_logger.logger.level == logging.DEBUG


def test_progress_logger_set_level_accepts_int():
    progress_logger = ProgressLogger("test_level_int")

    progress_logger.set_level(logging.WARNING)

    assert progress_logger.logger.level == logging.WARNING


# --- AsyncTaskProgress Tests ---


@pytest.mark.asyncio
async def test_async_task_progress_tracks_completed_count():
    async with AsyncTaskProgress(total=5, desc="Test") as progress:
        assert progress.completed == 0

        await progress.update(1)
        await progress.update(2)

        assert progress.completed == 3


@pytest.mark.asyncio
async def test_async_task_progress_sets_and_clears_global_bar():
    # Before - no active bar
    assert get_current_progress_bar() is None

    async with AsyncTaskProgress(total=3, desc="Test"):
        # During - bar is active
        assert get_current_progress_bar() is not None

    # After - bar is cleared
    assert get_current_progress_bar() is None


# --- run_tasks_with_progress Tests ---


@pytest.mark.asyncio
async def test_run_tasks_with_progress_executes_coroutines():
    async def double(x: int) -> int:
        return x * 2

    results = await run_tasks_with_progress(
        [double(1), double(2), double(3)],
        desc="Test",
    )

    assert sorted(results) == [2, 4, 6]


@pytest.mark.asyncio
async def test_run_tasks_with_progress_with_empty_list():
    results = await run_tasks_with_progress([], desc="Empty")

    assert results == []


@pytest.mark.asyncio
async def test_run_tasks_with_progress_respects_semaphore():
    """With semaphore=1, tasks should run sequentially."""
    execution_log = []

    async def tracked(task_id: int) -> int:
        execution_log.append(f"start_{task_id}")
        await asyncio.sleep(0.01)
        execution_log.append(f"end_{task_id}")
        return task_id

    await run_tasks_with_progress(
        [tracked(1), tracked(2)],
        semaphore=asyncio.Semaphore(1),
        desc="Sequential",
    )

    # First task must complete before second starts
    assert execution_log.index("end_1") < execution_log.index("start_2")
