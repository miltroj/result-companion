import asyncio
from dataclasses import dataclass

import pytest

from result_companion.core.chunking.chunking import (
    _collect_ancestor_context_at,
    _render_rf_keywords,
    _render_rf_test_structure,
    _split_long_line,
    accumulate_llm_results_for_summarization,
    analyze_chunk,
    chunk_rf_test_lines,
    split_text_into_chunks,
    synthesize_summaries,
)
from result_companion.core.chunking.utils import Chunking


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
        # chunk_size=50: test line (27) + Kw1 (24) overflows → 3 chunks + 1 synthesis = 4 calls
        fake_acompletion = FakeACompletionSequence(
            responses=["analysis1", "analysis2", "analysis3", "final summary"]
        )

        test_case = {
            "name": "chunking_test",
            "status": "FAIL",
            "body": [
                {"name": "Kw1", "status": "PASS", "type": "KEYWORD"},
                {"name": "Kw2", "status": "PASS", "type": "KEYWORD"},
            ],
        }
        chunking_strategy = Chunking(
            chunk_size=50,
            number_of_chunks=3,
            raw_text_len=100,
            tokens_from_raw_text=25,
            tokenized_chunks=3,
        )

        patch_smart_acompletion(fake_acompletion)

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

        test_case = {
            "name": "concurrency_test",
            "status": "FAIL",
            "body": [
                {"name": f"Kw{i}", "status": "PASS", "type": "KEYWORD"}
                for i in range(8)
            ],
        }
        chunking_strategy = Chunking(
            chunk_size=50,
            number_of_chunks=4,
            raw_text_len=200,
            tokens_from_raw_text=50,
            tokenized_chunks=4,
        )

        patch_smart_acompletion(tracking_acompletion)

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

    def test_long_line_with_current_flushes_current_then_splits(self):
        lines = [_SUITE, (0, "X" * 100)]

        result = chunk_rf_test_lines(lines, chunk_size=50)

        assert len(result) >= 2
        assert result[0] == "Suite: S"
        assert "X" not in result[0]
        assert "X" in result[1]

    def test_initial_chunks_are_fuller_than_last_when_lines_do_not_divide_evenly(self):
        # 8 depth-0 lines of 10 chars each with chunk_size=35:
        # chunk1 fits 3 lines (32 chars), continuations fit 2 (27 chars each),
        # last chunk has 1 line (16 chars) → clearly smaller than all predecessors.
        lines = [(0, "A" * 10)] * 8

        result = chunk_rf_test_lines(lines, chunk_size=35)

        assert len(result) > 1
        assert len(result[-1]) < len(result[-2])


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_kw(
    name: str = "Kw",
    status: str = "PASS",
    args: list | None = None,
    body: list | None = None,
    kw_type: str = "KEYWORD",
) -> dict:
    item: dict = {"name": name, "status": status, "type": kw_type}
    if args:
        item["args"] = args
    if body:
        item["body"] = body
    return item


def make_msg(message: str = "log line") -> dict:
    return {"type": "MESSAGE", "message": message}


def make_test(
    name: str = "My Test",
    status: str = "PASS",
    body: list | None = None,
    suite_context: list | None = None,
) -> dict:
    tc: dict = {"name": name, "status": status}
    if body:
        tc["body"] = body
    if suite_context:
        tc["suite_context"] = suite_context
    return tc


class TestRenderRfKeywords:
    """Tests for _render_rf_keywords."""

    def test_empty_body_returns_empty_list(self):
        assert _render_rf_keywords([], depth=0) == []

    def test_message_item_uses_message_text_directly(self):
        result = _render_rf_keywords([make_msg("log line")], depth=1)

        assert result == [(1, "log line")]

    def test_keyword_renders_kind_name_and_status(self):
        result = _render_rf_keywords([make_kw("Log", "PASS")], depth=0)

        assert result == [(0, "Keyword: Log - PASS")]

    def test_keyword_with_args_appends_args_line_at_next_depth(self):
        result = _render_rf_keywords(
            [make_kw("Log", "PASS", args=["hello", 42])], depth=0
        )

        assert (0, "Keyword: Log - PASS") in result
        assert (1, "args: hello, 42") in result

    def test_nested_keyword_body_rendered_at_incremented_depth(self):
        inner = make_kw("Inner", "PASS")
        outer = make_kw("Outer", "PASS", body=[inner])

        result = _render_rf_keywords([outer], depth=0)

        assert (0, "Keyword: Outer - PASS") in result
        assert (1, "Keyword: Inner - PASS") in result


class TestRenderRfTestStructure:
    """Tests for _render_rf_test_structure."""

    def test_no_suite_context_renders_single_test_line_at_depth_0(self):
        result = _render_rf_test_structure(make_test("Login Test", "PASS"))

        assert result == [(0, "Test: Login Test - PASS")]

    def test_suite_context_produces_nested_suite_lines_before_test(self):
        suite_ctx = [{"name": "Suite A"}, {"name": "Suite B"}]

        result = _render_rf_test_structure(make_test(suite_context=suite_ctx))

        assert result[0] == (0, "Suite: Suite A")
        assert result[1] == (1, "Suite: Suite B")
        assert result[2] == (2, "Test: My Test - PASS")

    def test_body_keywords_appended_after_test_line(self):
        result = _render_rf_test_structure(make_test(body=[make_kw("Click", "PASS")]))

        assert result[0] == (0, "Test: My Test - PASS")
        assert result[1] == (1, "Keyword: Click - PASS")
