from pathlib import Path
from unittest.mock import patch

import pytest

from result_companion.api import _analyze, analyze
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


PATCH_PREFIX = "result_companion.api"


class TestAnalyze:
    """Tests for _analyze async function."""

    @pytest.fixture(autouse=True)
    def _patch_deps(self, monkeypatch):
        self.config = make_config()

        monkeypatch.setattr(f"{PATCH_PREFIX}.load_config", lambda _: self.config)
        monkeypatch.setattr(
            f"{PATCH_PREFIX}._run_provider_init_strategies", lambda **kw: None
        )
        monkeypatch.setattr(f"{PATCH_PREFIX}.set_global_log_level", lambda _: None)

        self.parsed_test_cases = [
            {"name": "test_pass", "status": "PASS"},
            {"name": "test_fail", "status": "FAIL"},
        ]
        monkeypatch.setattr(
            f"{PATCH_PREFIX}.get_robot_results_from_file_as_dict",
            lambda **kw: self.parsed_test_cases,
        )

        async def fake_execute(**kw):
            names = [t["name"] for t in kw["test_cases"]]
            return {n: f"Analysis of {n}" for n in names}

        monkeypatch.setattr(f"{PATCH_PREFIX}.execute_llm_and_get_results", fake_execute)

    @pytest.mark.asyncio
    async def test_filters_passing_tests_by_default(self):
        result = await _analyze(output=Path("output.xml"))

        assert result.test_names == ["test_fail"]
        assert "test_pass" not in result.llm_results

    @pytest.mark.asyncio
    async def test_includes_passing_when_requested(self):
        result = await _analyze(output=Path("output.xml"), include_passing=True)

        assert "test_pass" in result.test_names
        assert "test_fail" in result.test_names

    @pytest.mark.asyncio
    async def test_returns_analysis_result(self):
        result = await _analyze(output=Path("output.xml"))

        assert isinstance(result, AnalysisResult)
        assert result.llm_results == {"test_fail": "Analysis of test_fail"}
        assert result.summary is None

    @pytest.mark.asyncio
    async def test_concurrency_overrides_config(self):
        await _analyze(
            output=Path("output.xml"),
            test_case_concurrency=5,
            chunk_concurrency=3,
        )

        assert self.config.concurrency.test_case == 5
        assert self.config.concurrency.chunk == 3

    @pytest.mark.asyncio
    async def test_summarize_failures_calls_llm(self, monkeypatch):
        async def fake_summarize(**kw):
            return "Root cause: network timeout"

        monkeypatch.setattr(
            f"{PATCH_PREFIX}.summarize_failures_with_llm", fake_summarize
        )

        result = await _analyze(output=Path("output.xml"), summarize_failures=True)

        assert result.summary == "Root cause: network timeout"

    @pytest.mark.asyncio
    async def test_summarize_skipped_on_dryrun(self, monkeypatch):
        summary_called = False

        async def fake_summarize(**kw):
            nonlocal summary_called
            summary_called = True
            return "should not appear"

        monkeypatch.setattr(
            f"{PATCH_PREFIX}.summarize_failures_with_llm", fake_summarize
        )

        result = await _analyze(
            output=Path("output.xml"), summarize_failures=True, dryrun=True
        )

        assert result.summary is None
        assert not summary_called


class TestAnalyzeSync:
    """Tests for analyze sync wrapper."""

    def test_converts_string_paths_and_delegates(self):
        with patch(f"{PATCH_PREFIX}._analyze") as mocked:
            mocked.return_value = AnalysisResult(
                llm_results={"t": "r"}, test_names=["t"]
            )

            result = analyze("output.xml", config="my_config.yaml", dryrun=True)

            call_kwargs = mocked.call_args.kwargs
            assert call_kwargs["output"] == Path("output.xml")
            assert call_kwargs["config"] == Path("my_config.yaml")
            assert call_kwargs["dryrun"] is True
            assert result.test_names == ["t"]

    def test_config_none_stays_none(self):
        with patch(f"{PATCH_PREFIX}._analyze") as mocked:
            mocked.return_value = AnalysisResult()

            analyze("output.xml")

            assert mocked.call_args.kwargs["config"] is None
