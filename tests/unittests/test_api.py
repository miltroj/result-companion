from pathlib import Path

import pytest

from result_companion._internal.analysis_helpers import apply_concurrency_overrides
from result_companion.api import analyze, run_analysis
from result_companion.core.chunking.rf_results import ContextAwareRobotResults
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


PATCH_API = "result_companion.api"
PATCH_HELPERS = "result_companion._internal.analysis_helpers"


class FakeContextAwareRobotResults(ContextAwareRobotResults):
    """Pre-built results object for API tests without loading output.xml."""

    def __init__(self, test_names: list[str]) -> None:
        self._fake_test_names = list(test_names)
        self._chunking = True

    @property
    def test_names(self) -> list[str]:
        return self._fake_test_names


async def fake_execute(**kw):
    results = kw["results"]
    return {n: f"Analysis of {n}" for n in results.test_names}


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
        results = FakeContextAwareRobotResults(["test_fail"])

        result = await run_analysis(config=config, results=results)

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
            results=FakeContextAwareRobotResults(["test_fail"]),
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
            results=FakeContextAwareRobotResults(["t"]),
            summarize_failures=True,
            dryrun=True,
        )

        assert result.summary is None
        assert not summary_called

    @pytest.mark.asyncio
    async def test_accepts_pre_built_objects(self):
        """Verifies no file paths are needed — only config + results."""
        config = make_config()
        config.concurrency.test_case = 7

        result = await run_analysis(
            config=config,
            results=FakeContextAwareRobotResults(["test_pass", "test_fail"]),
        )

        assert len(result.test_names) == 2
        assert config.concurrency.test_case == 7


class TestAnalyze:
    """Tests for analyze — the main sync programmatic entry point."""

    @pytest.fixture(autouse=True)
    def _patch_deps(self, monkeypatch):
        monkeypatch.setattr(
            f"{PATCH_HELPERS}.run_provider_init_strategies", lambda **kw: None
        )
        monkeypatch.setattr(f"{PATCH_API}.set_global_log_level", lambda _: None)
        monkeypatch.setattr(f"{PATCH_API}.execute_llm_and_get_results", fake_execute)

    def test_with_context_aware_results_passes_results_directly(self):
        config = make_config()
        results = FakeContextAwareRobotResults(["test_fail"])

        result = analyze(output=results, config=config)

        assert result.llm_results == {"test_fail": "Analysis of test_fail"}
        assert result.test_names == ["test_fail"]

    def test_with_path_loads_and_filters(self, monkeypatch):
        monkeypatch.setattr(
            f"{PATCH_API}.get_rc_robot_results",
            lambda **kw: FakeContextAwareRobotResults(["test_fail"]),
        )

        result = analyze(output="output.xml", config=make_config())

        assert result.test_names == ["test_fail"]
        assert "test_pass" not in result.llm_results

    def test_with_path_includes_passing_when_requested(self, monkeypatch):
        monkeypatch.setattr(
            f"{PATCH_API}.get_rc_robot_results",
            lambda **kw: FakeContextAwareRobotResults(["test_pass", "test_fail"]),
        )

        result = analyze(
            output="output.xml", config=make_config(), include_passing=True
        )

        assert "test_pass" in result.test_names
        assert "test_fail" in result.test_names

    def test_context_aware_results_ignores_tag_filters(self):
        config = make_config()
        results = FakeContextAwareRobotResults(["test_pass", "test_fail"])

        result = analyze(
            output=results,
            config=config,
            include_tags=["smoke"],
            exclude_tags=["wip"],
        )

        assert len(result.test_names) == 2

    def test_quiet_false_skips_log_level_change(self, monkeypatch):
        log_calls: list[str] = []
        monkeypatch.setattr(f"{PATCH_API}.set_global_log_level", log_calls.append)

        analyze(
            output=FakeContextAwareRobotResults(["t"]),
            config=make_config(),
            quiet=False,
        )

        assert log_calls == []

    def test_quiet_true_sets_log_level_to_error(self, monkeypatch):
        log_calls: list[str] = []
        monkeypatch.setattr(f"{PATCH_API}.set_global_log_level", log_calls.append)

        analyze(
            output=FakeContextAwareRobotResults(["t"]), config=make_config(), quiet=True
        )

        assert log_calls == ["ERROR"]

    def test_sets_chunking_when_not_already_set(self, monkeypatch):
        chunking_set: list[object] = []

        class NoChunkingResults(FakeContextAwareRobotResults):
            def __init__(self) -> None:
                super().__init__(["t"])
                self._chunking = None

            def set_chunking(self, strategy: object) -> "NoChunkingResults":
                chunking_set.append(strategy)
                self._chunking = strategy
                return self

        result = analyze(output=NoChunkingResults(), config=make_config())

        assert len(chunking_set) == 1
        assert result.test_names == ["t"]

    def test_path_object_forwards_correct_args_to_loader(self, monkeypatch):
        captured: dict = {}

        def fake_loader(**kw: object) -> FakeContextAwareRobotResults:
            captured.update(kw)
            return FakeContextAwareRobotResults(["t"])

        monkeypatch.setattr(f"{PATCH_API}.get_rc_robot_results", fake_loader)

        analyze(
            output=Path("out.xml"),
            config=make_config(),
            include_tags=["smoke"],
            exclude_tags=["wip"],
            include_passing=True,
        )

        assert captured["file_path"] == Path("out.xml")
        assert captured["include_tags"] == ["smoke"]
        assert captured["exclude_tags"] == ["wip"]
        assert captured["exclude_passing"] is False
