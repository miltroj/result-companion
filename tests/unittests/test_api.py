import pytest

from result_companion.api import analyze, run_analysis
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


def make_test_cases() -> list[dict]:
    """Creates test case dicts for testing."""
    return [
        {"name": "test_pass", "status": "PASS"},
        {"name": "test_fail", "status": "FAIL"},
    ]


PATCH_PREFIX = "result_companion.api"


async def fake_execute(**kw):
    names = [t["name"] for t in kw["test_cases"]]
    return {n: f"Analysis of {n}" for n in names}


class TestRunAnalysis:
    """Tests for run_analysis — the core async object-based API."""

    @pytest.fixture(autouse=True)
    def _patch_deps(self, monkeypatch):
        monkeypatch.setattr(
            f"{PATCH_PREFIX}._run_provider_init_strategies", lambda **kw: None
        )
        monkeypatch.setattr(f"{PATCH_PREFIX}.execute_llm_and_get_results", fake_execute)

    @pytest.mark.asyncio
    async def test_returns_analysis_result(self):
        config = make_config()
        test_cases = [{"name": "test_fail", "status": "FAIL"}]

        result = await run_analysis(config=config, test_cases=test_cases)

        assert isinstance(result, AnalysisResult)
        assert result.llm_results == {"test_fail": "Analysis of test_fail"}
        assert result.test_names == ["test_fail"]
        assert result.summary is None

    @pytest.mark.asyncio
    async def test_summarize_failures(self, monkeypatch):
        async def fake_summarize(**kw):
            return "Root cause: network timeout"

        monkeypatch.setattr(
            f"{PATCH_PREFIX}.summarize_failures_with_llm", fake_summarize
        )

        result = await run_analysis(
            config=make_config(),
            test_cases=[{"name": "test_fail", "status": "FAIL"}],
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

        monkeypatch.setattr(
            f"{PATCH_PREFIX}.summarize_failures_with_llm", fake_summarize
        )

        result = await run_analysis(
            config=make_config(),
            test_cases=[{"name": "t", "status": "FAIL"}],
            summarize_failures=True,
            dryrun=True,
        )

        assert result.summary is None
        assert not summary_called

    @pytest.mark.asyncio
    async def test_accepts_pre_built_objects(self):
        """Verifies no file paths are needed — only config + test_cases."""
        config = make_config()
        config.concurrency.test_case = 7

        result = await run_analysis(config=config, test_cases=make_test_cases())

        assert len(result.test_names) == 2
        assert config.concurrency.test_case == 7


class TestAnalyze:
    """Tests for analyze — the main sync programmatic entry point."""

    @pytest.fixture(autouse=True)
    def _patch_deps(self, monkeypatch):
        monkeypatch.setattr(
            f"{PATCH_PREFIX}._run_provider_init_strategies", lambda **kw: None
        )
        monkeypatch.setattr(f"{PATCH_PREFIX}.set_global_log_level", lambda _: None)
        monkeypatch.setattr(f"{PATCH_PREFIX}.execute_llm_and_get_results", fake_execute)

    def test_with_list_passes_test_cases_directly(self):
        config = make_config()
        test_cases = [{"name": "test_fail", "status": "FAIL"}]

        result = analyze(output=test_cases, config=config)

        assert result.llm_results == {"test_fail": "Analysis of test_fail"}
        assert result.test_names == ["test_fail"]

    def test_with_path_loads_and_filters(self, monkeypatch):
        monkeypatch.setattr(
            f"{PATCH_PREFIX}.get_robot_results_from_file_as_dict",
            lambda **kw: make_test_cases(),
        )

        result = analyze(output="output.xml", config=make_config())

        assert result.test_names == ["test_fail"]
        assert "test_pass" not in result.llm_results

    def test_with_path_includes_passing_when_requested(self, monkeypatch):
        monkeypatch.setattr(
            f"{PATCH_PREFIX}.get_robot_results_from_file_as_dict",
            lambda **kw: make_test_cases(),
        )

        result = analyze(
            output="output.xml", config=make_config(), include_passing=True
        )

        assert "test_pass" in result.test_names
        assert "test_fail" in result.test_names

    def test_with_list_ignores_tag_filters(self):
        config = make_config()
        test_cases = make_test_cases()

        result = analyze(
            output=test_cases,
            config=config,
            include_tags=["smoke"],
            exclude_tags=["wip"],
        )

        assert len(result.test_names) == 2
