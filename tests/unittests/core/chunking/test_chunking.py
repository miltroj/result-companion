import asyncio
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from result_companion.core.chunking.chunking import (
    accumulate_llm_results_for_summarization,
    analyze_chunk,
    split_text_into_chunks,
    synthesize_summaries,
)
from result_companion.core.chunking.utils import Chunking


@dataclass
class FakeLiteLLMChoice:
    """Fake LiteLLM choice for testing."""

    message: object


@dataclass
class FakeLiteLLMResponse:
    """Fake LiteLLM response for testing."""

    content: str

    @property
    def choices(self):
        """Returns fake choices list."""
        msg = type("Message", (), {"content": self.content})()
        choice = FakeLiteLLMChoice(message=msg)
        return [choice]


class FakeACompletionSequence:
    """Fake acompletion that returns responses in sequence."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0

    async def __call__(self, **kwargs):
        """Returns next response in sequence."""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return FakeLiteLLMResponse(content=response)
        return FakeLiteLLMResponse(content="default response")


class TestSplitTextIntoChunks:
    """Tests for split_text_into_chunks function."""

    def test_splits_text_into_chunks(self):
        """Test basic text splitting."""
        text = "abcdefghij"
        chunks = split_text_into_chunks(text, chunk_size=4, overlap=1)

        assert len(chunks) == 4
        assert chunks[0] == "abcd"
        assert chunks[1] == "defg"
        assert chunks[2] == "ghij"

    def test_handles_text_shorter_than_chunk_size(self):
        """Test with text shorter than chunk size."""
        text = "abc"
        chunks = split_text_into_chunks(text, chunk_size=10, overlap=2)

        assert chunks == ["abc"]

    def test_handles_empty_text(self):
        """Test with empty text."""
        chunks = split_text_into_chunks("", chunk_size=10, overlap=2)

        assert chunks == []

    def test_handles_zero_chunk_size(self):
        """Test with zero chunk size returns original text."""
        text = "test"
        chunks = split_text_into_chunks(text, chunk_size=0, overlap=0)

        assert chunks == ["test"]

    def test_overlap_larger_than_chunk_size_uses_default(self):
        """Test that overlap >= chunk_size falls back to 10%."""
        text = "abcdefghij"
        chunks = split_text_into_chunks(text, chunk_size=4, overlap=10)

        # Should not infinite loop, overlap reduced to chunk_size // 10
        assert len(chunks) > 0


class TestAnalyzeChunk:
    """Tests for analyze_chunk function."""

    @pytest.mark.asyncio
    async def test_formats_prompt_and_calls_llm(self):
        """Test that analyze_chunk formats prompt correctly."""
        captured_messages = []

        async def capture_acompletion(**kwargs):
            captured_messages.append(kwargs.get("messages"))
            return FakeLiteLLMResponse(content="chunk analysis")

        semaphore = asyncio.Semaphore(1)

        with patch(
            "result_companion.core.chunking.chunking.acompletion",
            capture_acompletion,
        ):
            result = await analyze_chunk(
                chunk="test chunk content",
                chunk_idx=0,
                total_chunks=3,
                test_name="my_test",
                chunk_analysis_prompt="Analyze this: {text}",
                llm_params={"model": "test-model"},
                semaphore=semaphore,
            )

        assert result == "chunk analysis"
        assert len(captured_messages) == 1
        assert "Analyze this: test chunk content" in captured_messages[0][0]["content"]


class TestSynthesizeSummaries:
    """Tests for synthesize_summaries function."""

    @pytest.mark.asyncio
    async def test_synthesizes_summaries(self):
        """Test that synthesize_summaries formats and calls LLM."""
        captured_messages = []

        async def capture_acompletion(**kwargs):
            captured_messages.append(kwargs.get("messages"))
            return FakeLiteLLMResponse(content="final synthesis")

        with patch(
            "result_companion.core.chunking.chunking.acompletion",
            capture_acompletion,
        ):
            result = await synthesize_summaries(
                aggregated_summary="chunk1 summary\nchunk2 summary",
                final_synthesis_prompt="Combine: {summary}",
                llm_params={"model": "test-model"},
            )

        assert result == "final synthesis"
        assert "Combine:" in captured_messages[0][0]["content"]
        assert "chunk1 summary" in captured_messages[0][0]["content"]


class TestAccumulateLLMResultsForSummarization:
    """Tests for accumulate_llm_results_for_summarization function."""

    @pytest.mark.asyncio
    async def test_splits_and_summarizes(self):
        """Test full chunking and summarization flow."""
        # str(test_case) creates ~155 chars, chunk_size=50 produces 4 chunks + 1 synthesis
        fake_acompletion = FakeACompletionSequence(
            responses=["chunk1", "chunk2", "chunk3", "chunk4", "final summary"]
        )

        test_case = {"name": "chunking_test", "content": "x" * 100}
        chunking_strategy = Chunking(
            chunk_size=50,
            number_of_chunks=2,
            raw_text_len=100,
            tokens_from_raw_text=25,
            tokenized_chunks=2,
        )

        with patch(
            "result_companion.core.chunking.chunking.acompletion",
            fake_acompletion,
        ):
            result, name, chunks = await accumulate_llm_results_for_summarization(
                test_case=test_case,
                chunk_analysis_prompt="Analyze: {text}",
                final_synthesis_prompt="Synthesize: {summary}",
                chunking_strategy=chunking_strategy,
                llm_params={"model": "test-model"},
                chunk_concurrency=1,
            )

        assert result == "final summary"
        assert name == "chunking_test"
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_respects_chunk_concurrency(self):
        """Test that chunk_concurrency limits parallel processing."""
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def tracking_acompletion(**kwargs):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.01)  # Simulate work
            async with lock:
                current_concurrent -= 1
            return FakeLiteLLMResponse(content="result")

        test_case = {"name": "concurrency_test", "content": "x" * 200}
        chunking_strategy = Chunking(
            chunk_size=50,
            number_of_chunks=4,
            raw_text_len=200,
            tokens_from_raw_text=50,
            tokenized_chunks=4,
        )

        with patch(
            "result_companion.core.chunking.chunking.acompletion",
            tracking_acompletion,
        ):
            await accumulate_llm_results_for_summarization(
                test_case=test_case,
                chunk_analysis_prompt="Analyze: {text}",
                final_synthesis_prompt="Synthesize: {summary}",
                chunking_strategy=chunking_strategy,
                llm_params={"model": "test-model"},
                chunk_concurrency=2,
            )

        # Max concurrent should be limited to 2 (not counting final synthesis)
        assert max_concurrent <= 2
