import asyncio
from dataclasses import dataclass

import pytest

from result_companion.core.chunking.chunking import (
    _collect_ancestor_context_at,
    _split_long_line,
    accumulate_llm_results_for_summarization,
    analyze_chunk,
    chunk_rf_test_lines,
    split_text_into_chunks,
    synthesize_summaries,
)


@pytest.fixture
def patch_smart_acompletion(monkeypatch):
    """Patches chunking._smart_acompletion with provided async callable."""

    def _apply(fake_acompletion):
        monkeypatch.setattr(
            "result_companion.core.chunking.chunking._smart_acompletion",
            fake_acompletion,
        )

    return _apply


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

    async def __call__(self, messages: list[dict], **kwargs):
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
    async def test_formats_prompt_and_calls_llm(self, patch_smart_acompletion):
        """Test that analyze_chunk formats prompt correctly."""
        captured_messages = []

        async def capture_acompletion(messages: list[dict], **kwargs):
            captured_messages.append(messages)
            return FakeLiteLLMResponse(content="chunk analysis")

        semaphore = asyncio.Semaphore(1)
        patch_smart_acompletion(capture_acompletion)

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
    async def test_synthesizes_summaries(self, patch_smart_acompletion):
        """Test that synthesize_summaries formats and calls LLM."""
        captured_messages = []

        async def capture_acompletion(messages: list[dict], **kwargs):
            captured_messages.append(messages)
            return FakeLiteLLMResponse(content="final synthesis")

        patch_smart_acompletion(capture_acompletion)

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
    async def test_splits_and_summarizes(self, patch_smart_acompletion):
        """Test full chunking and summarization flow."""
        fake_acompletion = FakeACompletionSequence(
            responses=["analysis1", "analysis2", "final summary"]
        )
        chunks = ["chunk one content", "chunk two content"]

        patch_smart_acompletion(fake_acompletion)

        result, name, returned_chunks = await accumulate_llm_results_for_summarization(
            test_name="chunking_test",
            chunks=chunks,
            chunk_analysis_prompt="Analyze: {text}",
            final_synthesis_prompt="Synthesize: {summary}",
            llm_params={"model": "test-model"},
            chunk_concurrency=1,
        )

        assert result == "final summary"
        assert name == "chunking_test"
        assert returned_chunks == chunks

    @pytest.mark.asyncio
    async def test_respects_chunk_concurrency(self, patch_smart_acompletion):
        """Test that chunk_concurrency limits parallel processing."""
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def tracking_acompletion(messages: list[dict], **kwargs):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.01)
            async with lock:
                current_concurrent -= 1
            return FakeLiteLLMResponse(content="result")

        chunks = [f"chunk{i}" for i in range(8)]

        patch_smart_acompletion(tracking_acompletion)

        await accumulate_llm_results_for_summarization(
            test_name="concurrency_test",
            chunks=chunks,
            chunk_analysis_prompt="Analyze: {text}",
            final_synthesis_prompt="Synthesize: {summary}",
            llm_params={"model": "test-model"},
            chunk_concurrency=2,
        )

        assert max_concurrent <= 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Depth-0 suite line, depth-1 test line, depth-2 keyword line used across tests.
_SUITE = (0, "Suite: S")
_TEST = (1, "Test: T - PASS")
_KW1 = (2, "Keyword: K1 - PASS")
_KW2 = (2, "Keyword: K2 - PASS")


class TestCollectAncestorContext:
    """Tests for _collect_ancestor_context_at."""

    def test_collects_suite_and_test_for_depth_2_line(self):
        lines = [_SUITE, _TEST, _KW1]

        result = _collect_ancestor_context_at(lines, at_idx=2)

        assert len(result) == 2
        assert "Suite: S" in result[0]
        assert "Test: T - PASS" in result[1]

    def test_collects_only_suite_for_depth_1_line(self):
        lines = [_SUITE, _TEST]

        result = _collect_ancestor_context_at(lines, at_idx=1)

        assert len(result) == 1
        assert "Suite: S" in result[0]

    def test_returns_empty_list_for_depth_0_line(self):
        lines = [_SUITE, (0, "Suite: S2")]

        result = _collect_ancestor_context_at(lines, at_idx=1)

        assert result == []


class TestSplitLongLine:
    """Tests for _split_long_line."""

    def test_splits_into_breadcrumb_prefixed_chunks(self):
        text = "A" * 100
        breadcrumbs = ["Suite: S", "    Test: T"]

        result = _split_long_line(text, depth=2, breadcrumbs=breadcrumbs, chunk_size=60)

        assert len(result) > 1
        for chunk in result:
            assert "Suite: S" in chunk
            assert "{...}" in chunk
            assert "A" in chunk

    def test_uses_minimum_piece_size_when_breadcrumbs_consume_most_budget(self):
        # Breadcrumbs larger than chunk_size force the chunk_size//3 floor.
        large_breadcrumbs = ["X" * 200]
        text = "A" * 50

        result = _split_long_line(
            text, depth=0, breadcrumbs=large_breadcrumbs, chunk_size=30
        )

        assert len(result) > 0
        assert all("A" in chunk for chunk in result)


class TestChunkRfTestLines:
    """Tests for chunk_rf_test_lines – all code paths."""

    def test_empty_input_returns_empty_list(self):
        assert chunk_rf_test_lines([], chunk_size=1000) == []

    def test_fits_in_single_chunk_when_total_under_limit(self):
        lines = [_SUITE, _TEST, _KW1]

        result = chunk_rf_test_lines(lines, chunk_size=10_000)

        assert len(result) == 1
        assert "Suite: S" in result[0]
        assert "Keyword: K1 - PASS" in result[0]

    def test_normal_overflow_flushes_and_starts_continuation_with_breadcrumbs(self):
        # chunk_size=60: lines 0-2 fit (55 chars), line 3 overflows → 2 chunks.
        lines = [_SUITE, _TEST, _KW1, _KW2]

        result = chunk_rf_test_lines(lines, chunk_size=60)

        assert len(result) == 2
        # First chunk has all three initial lines.
        assert "Suite: S" in result[0]
        assert "Keyword: K1 - PASS" in result[0]
        # Second chunk carries ancestor context + continuation marker.
        assert "Suite: S" in result[1]
        assert "Test: T - PASS" in result[1]
        assert "{...}" in result[1]
        assert "Keyword: K2 - PASS" in result[1]

    def test_long_line_with_no_current_splits_into_sub_chunks(self):
        # Single oversized line → no current to flush, split directly.
        lines = [(0, "X" * 100)]

        result = chunk_rf_test_lines(lines, chunk_size=30)

        assert len(result) > 1
        assert all("{...}" in chunk for chunk in result)

    def test_long_line_with_current_fills_chunk_before_flushing(self):
        lines = [_SUITE, (0, "X" * 100)]

        result = chunk_rf_test_lines(lines, chunk_size=50)

        assert len(result) >= 2
        assert "Suite: S" in result[0]
        assert "X" in result[0]  # current chunk filled to capacity before flush
        assert len(result[0]) <= 50
        assert "X" in result[1]

    def test_initial_chunks_are_fuller_than_last_when_lines_do_not_divide_evenly(self):
        # 8 depth-0 lines of 10 chars each with chunk_size=35:
        # chunk1 fits 3 lines (32 chars), continuations fit 2 (27 chars each),
        # last chunk has 1 line (16 chars) → clearly smaller than all predecessors.
        lines = [(0, "A" * 10)] * 8

        result = chunk_rf_test_lines(lines, chunk_size=35)

        assert len(result) > 1
        assert len(result[-1]) < len(result[-2])
