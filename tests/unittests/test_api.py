import pytest

from result_companion._internal.analysis_helpers import apply_concurrency_overrides
from result_companion.api import analyze, run_analysis
from result_companion.core.chunking.rf_chunker import ChunkableResult
from result_companion.core.parsers.config import (
    ChunkingPromptsModel,
    DefaultConfigModel,
    LLMConfigModel,
    LLMFactoryModel,
    TokenizerModel,
)
from result_companion.core.results.analysis_result import AnalysisResult


def make_config() -> DefaultConfigModel:
    """Creates minimal test configuration."""
    return DefaultConfigModel(
        version=1,
        llm_config=LLMConfigModel(
            question_prompt="What failed?",
            prompt_template="Q: {question}\nC: {context}",
            summary_prompt_template="Summary:\n{analyses}",
            chunking=ChunkingPromptsModel(
                chunk_analysis_prompt="Analyze: {text}",
                final_synthesis_prompt="Synthesize: {summary}",
            ),
        ),
        llm_factory=LLMFactoryModel(model="openai/gpt-4", api_key="sk-test"),
        tokenizer=TokenizerModel(tokenizer="openai_tokenizer", max_content_tokens=1000),
    )


def make_fake_chunkable(test_cases: list[tuple[str, str]]) -> ChunkableResult:
    """Creates a fake ChunkableResult that yields given (name, status) pairs."""

    class FakeChunkable:
        def iter_tests(self, include_passing: bool = True):
            for name, status in test_cases:
                if not include_passing and status == "PASS":
                    continue
                yield name, status, [(0, f"Test: {name} - {status}")]

        @property
        def test_count(self):
            return len(test_cases)

        def source_hash(self):
            return "abc123"

    return FakeChunkable()


PATCH_API = "result_companion.api"
PATCH_HELPERS = "result_companion._internal.analysis_helpers"


async def fake_execute(chunkable, config, include_passing=False, **kw):
    return {
        name: f"Analysis of {name}"
        for name, status, _ in chunkable.iter_tests(include_passing=include_passing)
    }


class TestApplyConcurrencyOverrides:
    """Tests for apply_concurrency_overrides."""

    def test_overrides_both_when_provided(self):
        config = make_config()
        apply_concurrency_overrides(
            config, test_case_concurrency=5, chunk_concurrency=3
        )
        assert config.concurrency.test_case == 5
        assert config.concurrency.chunk == 3

    def test_overrides_only_test_case_when_chunk_is_none(self):
        config = make_config()
        original_chunk = config.concurrency.chunk
        apply_concurrency_overrides(
            config, test_case_concurrency=4, chunk_concurrency=None
        )
        assert config.concurrency.test_case == 4
        assert config.concurrency.chunk == original_chunk

    def test_overrides_only_chunk_when_test_case_is_none(self):
        config = make_config()
        original_test_case = config.concurrency.test_case
        apply_concurrency_overrides(
            config, test_case_concurrency=None, chunk_concurrency=2
        )
        assert config.concurrency.test_case == original_test_case
        assert config.concurrency.chunk == 2


class TestRunAnalysis:
    """Tests for run_analysis — the core async object-based API."""

    @pytest.fixture(autouse=True)
    def _patch_deps(self, monkeypatch):
        monkeypatch.setattr(
            f"{PATCH_HELPERS}.run_provider_init_strategies", lambda **kw: None
        )
        monkeypatch.setattr(f"{PATCH_API}.execute_llm_and_get_results", fake_execute)

    @pytest.mark.asyncio
    async def test_returns_analysis_result(self):
        config = make_config()
        chunkable = make_fake_chunkable([("test_fail", "FAIL")])

        result = await run_analysis(config=config, chunkable=chunkable)

        assert isinstance(result, AnalysisResult)
        assert result.llm_results == {"test_fail": "Analysis of test_fail"}
        assert result.test_names == ["test_fail"]
        assert result.summary is None

    @pytest.mark.asyncio
    async def test_summarize_failures(self, monkeypatch):
        async def fake_summarize(**kw):
            return "Root cause: network timeout"

        monkeypatch.setattr(f"{PATCH_API}.summarize_failures_with_llm", fake_summarize)

        result = await run_analysis(
            config=make_config(),
            chunkable=make_fake_chunkable([("test_fail", "FAIL")]),
            summarize_failures=True,
        )

        assert result.summary == "Root cause: network timeout"

    @pytest.mark.asyncio
    async def test_dryrun_skips_summary(self, monkeypatch):
        summary_called = False

        async def fake_summarize(**kw):
            nonlocal summary_called
            summary_called = True
            return "should not appear"

        monkeypatch.setattr(f"{PATCH_API}.summarize_failures_with_llm", fake_summarize)

        result = await run_analysis(
            config=make_config(),
            chunkable=make_fake_chunkable([("t", "FAIL")]),
            summarize_failures=True,
            dryrun=True,
        )

        assert result.summary is None
        assert not summary_called

    @pytest.mark.asyncio
    async def test_filters_passing_when_include_passing_false(self):
        config = make_config()
        chunkable = make_fake_chunkable([("test_pass", "PASS"), ("test_fail", "FAIL")])

        result = await run_analysis(
            config=config, chunkable=chunkable, include_passing=False
        )

        assert "test_fail" in result.test_names
        assert "test_pass" not in result.test_names


class TestAnalyze:
    """Tests for analyze — the main sync programmatic entry point."""

    @pytest.fixture(autouse=True)
    def _patch_deps(self, monkeypatch):
        monkeypatch.setattr(
            f"{PATCH_HELPERS}.run_provider_init_strategies", lambda **kw: None
        )
        monkeypatch.setattr(f"{PATCH_API}.set_global_log_level", lambda _: None)
        monkeypatch.setattr(f"{PATCH_API}.execute_llm_and_get_results", fake_execute)

    def test_with_path_loads_and_filters(self, monkeypatch):
        monkeypatch.setattr(
            f"{PATCH_API}.build_chunkable",
            lambda **kw: make_fake_chunkable(
                [("test_pass", "PASS"), ("test_fail", "FAIL")]
            ),
        )

        result = analyze(output="output.xml", config=make_config())

        assert "test_fail" in result.test_names
        assert "test_pass" not in result.test_names

    def test_with_path_includes_passing_when_requested(self, monkeypatch):
        monkeypatch.setattr(
            f"{PATCH_API}.build_chunkable",
            lambda **kw: make_fake_chunkable(
                [("test_pass", "PASS"), ("test_fail", "FAIL")]
            ),
        )

        result = analyze(
            output="output.xml", config=make_config(), include_passing=True
        )

        assert "test_pass" in result.test_names
        assert "test_fail" in result.test_names
