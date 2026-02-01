import asyncio
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from result_companion.core.analizers.common import (
    run_with_semaphore,
    simple_llm_call,
)


@dataclass
class FakeLiteLLMResponse:
    """Fake LiteLLM response for testing."""

    content: str

    @property
    def choices(self):
        """Returns fake choices list."""
        msg = type("Message", (), {"content": self.content})()
        choice = type("Choice", (), {"message": msg})()
        return [choice]


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


class TestSimpleLLMCall:
    """Tests for simple_llm_call function."""

    @pytest.mark.asyncio
    async def test_calls_acompletion_with_prompt(self):
        """Test that simple_llm_call formats messages correctly."""
        captured_kwargs = {}

        async def capture_acompletion(**kwargs):
            captured_kwargs.update(kwargs)
            return FakeLiteLLMResponse(content="LLM response")

        with patch(
            "result_companion.core.analizers.common.acompletion",
            capture_acompletion,
        ):
            result = await simple_llm_call(
                prompt="What is the answer?",
                llm_params={"model": "test-model"},
            )

        assert result == "LLM response"
        assert captured_kwargs["model"] == "test-model"
        assert captured_kwargs["messages"][0]["content"] == "What is the answer?"
        assert captured_kwargs["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_passes_llm_params(self):
        """Test that additional LLM params are passed through."""
        captured_kwargs = {}

        async def capture_acompletion(**kwargs):
            captured_kwargs.update(kwargs)
            return FakeLiteLLMResponse(content="response")

        with patch(
            "result_companion.core.analizers.common.acompletion",
            capture_acompletion,
        ):
            await simple_llm_call(
                prompt="test",
                llm_params={
                    "model": "openai/gpt-4",
                    "api_key": "sk-test",
                    "temperature": 0.5,
                },
            )

        assert captured_kwargs["model"] == "openai/gpt-4"
        assert captured_kwargs["api_key"] == "sk-test"
        assert captured_kwargs["temperature"] == 0.5
