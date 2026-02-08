import asyncio

import pytest

from result_companion.core.analizers.common import run_with_semaphore


class TestRunWithSemaphore:
    """Tests for run_with_semaphore function."""

    @pytest.mark.asyncio
    async def test_runs_coroutine_with_semaphore(self):
        """Test that coroutine runs under semaphore control."""
        semaphore = asyncio.Semaphore(1)

        async def simple_coroutine():
            return "result"

        result = await run_with_semaphore(semaphore, simple_coroutine())

        assert result == "result"

    @pytest.mark.asyncio
    async def test_limits_concurrency(self):
        """Test that semaphore limits concurrent executions."""
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(2)

        async def tracking_coroutine():
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.01)
            async with lock:
                current_concurrent -= 1
            return "done"

        # Run 5 coroutines with semaphore limit of 2
        tasks = [run_with_semaphore(semaphore, tracking_coroutine()) for _ in range(5)]
        await asyncio.gather(*tasks)

        assert max_concurrent <= 2
