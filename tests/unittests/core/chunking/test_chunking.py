import asyncio
from dataclasses import dataclass

import pytest

from result_companion.core.chunking.chunking import (
    ChunkingStrategy,
    _collect_ancestor_context_at,
    _split_long_line,
    accumulate_llm_results_for_summarization,
    analyze_chunk,
    chunk_rf_test_lines,
    deduplicate_consecutive_lines,
    synthesize_summaries,
)
from result_companion.core.parsers.config import TokenizerModel, TokenizerTypes


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
        patch_smart_acompletion(fake_acompletion)

        result, name, chunks = await accumulate_llm_results_for_summarization(
            test_name="chunking_test",
            chunks=["chunk one", "chunk two"],
            chunk_analysis_prompt="Analyze: {text}",
            final_synthesis_prompt="Synthesize: {summary}",
            llm_params={"model": "test-model"},
            chunk_concurrency=1,
        )

        assert result == "final summary"
        assert name == "chunking_test"
        assert len(chunks) == 2

    @pytest.mark.asyncio
    async def test_aggregates_chunk_summaries_with_numbering(
        self, patch_smart_acompletion
    ):
        """Test that chunk summaries are numbered and joined before final synthesis."""
        captured_synthesis_prompt = []

        async def capture_acompletion(messages: list[dict], **kwargs):
            content = messages[0]["content"]
            captured_synthesis_prompt.append(content)
            return FakeLiteLLMResponse(content="final")

        patch_smart_acompletion(capture_acompletion)

        await accumulate_llm_results_for_summarization(
            test_name="agg_test",
            chunks=["chunk_a", "chunk_b"],
            chunk_analysis_prompt="Analyze: {text}",
            final_synthesis_prompt="Synthesize: {summary}",
            llm_params={"model": "test-model"},
        )

        # Last call is the synthesis; its prompt contains numbered chunk summaries.
        synthesis_input = captured_synthesis_prompt[-1]
        assert "Chunk 1/2" in synthesis_input
        assert "Chunk 2/2" in synthesis_input

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
            await asyncio.sleep(0.01)  # Simulate work
            async with lock:
                current_concurrent -= 1
            return FakeLiteLLMResponse(content="result")

        patch_smart_acompletion(tracking_acompletion)

        await accumulate_llm_results_for_summarization(
            test_name="concurrency_test",
            chunks=[f"chunk{i}" for i in range(8)],
            chunk_analysis_prompt="Analyze: {text}",
            final_synthesis_prompt="Synthesize: {summary}",
            llm_params={"model": "test-model"},
            chunk_concurrency=2,
        )

        # Max concurrent should be limited to 2 (not counting final synthesis)
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


class TestDeduplicateConsecutiveLines:
    """Tests for deduplicate_consecutive_lines."""

    def test_unique_lines_unchanged(self):
        lines = [(0, "Suite: A"), (1, "Test: B"), (2, "msg one"), (2, "msg two")]

        result = deduplicate_consecutive_lines(lines)

        assert result == lines

    def test_consecutive_duplicates_collapsed_with_count(self):
        lines = [(2, "same log"), (2, "same log"), (2, "same log")]

        result = deduplicate_consecutive_lines(lines)

        assert result == [(2, "same log (repeats ×3)")]

    def test_non_consecutive_duplicates_not_collapsed(self):
        lines = [(1, "msg"), (1, "other"), (1, "msg")]

        result = deduplicate_consecutive_lines(lines)

        assert result == [(1, "msg"), (1, "other"), (1, "msg")]

    def test_single_line_unchanged(self):
        lines = [(0, "only line")]

        result = deduplicate_consecutive_lines(lines)

        assert result == [(0, "only line")]

    def test_empty_input_returns_empty(self):
        assert deduplicate_consecutive_lines([]) == []

    def test_mixed_run_lengths(self):
        lines = [(0, "a"), (0, "b"), (0, "b"), (0, "c"), (0, "c"), (0, "c")]

        result = deduplicate_consecutive_lines(lines)

        assert result == [(0, "a"), (0, "b (repeats ×2)"), (0, "c (repeats ×3)")]

    def test_same_log_from_different_keywords_not_collapsed(self):
        lines = [
            (1, "Keyword: Step A - PASS"),
            (2, "log: connection established"),
            (1, "Keyword: Step B - PASS"),
            (2, "log: connection established"),
        ]

        result = deduplicate_consecutive_lines(lines)

        assert result == lines

    def test_deduplications_is_not_reordering_keywords(self):
        lines = [
            (1, "Keyword: Step B - PASS"),
            (1, "Keyword: Step A3 - PASS"),
            (1, "Keyword: Step A2 - PASS"),
            (1, "Keyword: Step A1 - PASS"),
        ]

        result = deduplicate_consecutive_lines(lines)

        assert result == lines


class TestChunkingStrategy:
    """Tests for ChunkingStrategy dataclass."""

    def test_build_returns_strategy_with_correct_tokenizer_config(self):
        strategy = ChunkingStrategy.build(
            tokenizer=TokenizerTypes.OPENAI,
            max_content_tokens=4000,
            system_prompt="sys",
        )

        assert isinstance(strategy, ChunkingStrategy)
        assert strategy.tokenizer_config.tokenizer == TokenizerTypes.OPENAI
        assert strategy.tokenizer_config.max_content_tokens == 4000
        assert strategy.system_prompt == "sys"

    def test_build_uses_defaults(self):
        strategy = ChunkingStrategy.build()

        assert strategy.tokenizer_config.tokenizer == TokenizerTypes.OPENAI
        assert strategy.tokenizer_config.max_content_tokens == 8000
        assert strategy.system_prompt == ""

    def test_apply_returns_single_chunk_when_content_fits(self):
        strategy = ChunkingStrategy(
            tokenizer_config=TokenizerModel(
                tokenizer=TokenizerTypes.OPENAI, max_content_tokens=100_000
            )
        )
        lines = [(0, "Suite: S"), (1, "Test: T - FAIL"), (2, "Keyword: K - FAIL")]

        chunks, chunk_info = strategy.apply(lines)

        assert len(chunks) == 1
        assert "Suite: S" in chunks[0]
        assert not chunk_info.requires_chunking

    def test_apply_returns_multiple_chunks_when_content_exceeds_budget(self):
        strategy = ChunkingStrategy(
            tokenizer_config=TokenizerModel(
                tokenizer=TokenizerTypes.OPENAI, max_content_tokens=1
            )
        )
        lines = [(0, "Suite: S"), (1, "Test: T - FAIL"), (2, "Keyword: K - FAIL")]

        chunks, chunk_info = strategy.apply(lines)

        assert len(chunks) > 1
        assert chunk_info.requires_chunking

    def test_apply_empty_lines_returns_empty_chunks(self):
        strategy = ChunkingStrategy(
            tokenizer_config=TokenizerModel(
                tokenizer=TokenizerTypes.OPENAI, max_content_tokens=1000
            )
        )

        chunks, _ = strategy.apply([])

        assert chunks == []
