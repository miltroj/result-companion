"""Tests for progress bar utilities."""

import asyncio

import pytest

from result_companion.core.utils.progress import run_tasks_with_progress


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


@pytest.mark.asyncio
async def test_run_tasks_with_progress_preserves_result_order():
    """Results should be in the same order as input coroutines."""

    async def delayed(value: str, delay: float) -> str:
        await asyncio.sleep(delay)
        return value

    # Second task finishes first, but results should preserve input order
    results = await run_tasks_with_progress(
        [delayed("first", 0.02), delayed("second", 0.01)],
        semaphore=asyncio.Semaphore(2),
        desc="Order test",
    )

    assert results == ["first", "second"]
