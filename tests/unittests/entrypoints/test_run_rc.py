from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from result_companion._internal.analysis_helpers import run_provider_init_strategies
from result_companion.core.parsers.config import DefaultConfigModel
from result_companion.core.results.analysis_result import AnalysisResult
from result_companion.entrypoints.run_rc import _emit_reports, _main, run_rc


def make_fake_results(test_names: list[str], total: int | None = None) -> MagicMock:
    """Builds a stand-in for ContextAwareRobotResults in entrypoint tests."""
    fake = MagicMock()
    fake.test_names = test_names
    fake.total_test_count = total if total is not None else len(test_names)
    fake.source_hash = "abc123def456"
    fake._chunking = True
    fake.set_chunking = MagicMock()
    return fake


class TestRunProviderInitStrategies:
    """Tests for provider init strategy dispatcher."""

    @pytest.fixture(autouse=True)
    def _capture_ollama_init_calls(self, monkeypatch):
        calls = []

        def _fake_ollama_on_init_strategy(*, model_name: str):
            calls.append(model_name)

        monkeypatch.setattr(
            "result_companion._internal.analysis_helpers.ollama_on_init_strategy",
            _fake_ollama_on_init_strategy,
        )
        self.calls = calls

    def test_skips_non_ollama_models(self):
        run_provider_init_strategies(model_name="openai/gpt-4")
        assert self.calls == []

    def test_runs_for_ollama_models(self):
        run_provider_init_strategies(model_name="ollama_chat/llama2")
        assert self.calls == ["llama2"]

    def test_extracts_model_name_from_versioned_identifier(self):
        run_provider_init_strategies(model_name="ollama_chat/deepseek-r1:1.5b")
        assert self.calls == ["deepseek-r1"]


class TestMainE2E:
    """End-to-end tests for _main function."""

    @pytest.mark.asyncio
    async def test_main_executes_analysis(self):
        """Test that _main executes the full analysis flow."""
        with (
            patch(
                "result_companion.entrypoints.run_rc.create_llm_html_log"
            ) as mocked_html,
            patch("result_companion.api.execute_llm_and_get_results") as mocked_execute,
            patch(
                "result_companion.entrypoints.run_rc.get_rc_robot_results"
            ) as mocked_get_results,
            patch("result_companion.entrypoints.run_rc.load_config") as mocked_config,
            patch(
                "result_companion._internal.analysis_helpers.run_provider_init_strategies"
            ),
        ):
            fake_results = make_fake_results(["test2"], total=2)
            mocked_get_results.return_value = fake_results

            mocked_config.return_value = DefaultConfigModel(
                version=1.0,
                llm_config={
                    "question_prompt": "question prompt",
                    "prompt_template": "my_template {question} {context}",
                    "chunking": {
                        "chunk_analysis_prompt": "Analyze: {text}",
                        "final_synthesis_prompt": "Synthesize: {summary}",
                    },
                    "summary_prompt_template": "CI summary:\n{analyses}",
                },
                llm_factory={
                    "model": "openai/gpt-4",
                    "api_key": "sk-test",
                },
                tokenizer={
                    "tokenizer": "openai_tokenizer",
                    "max_content_tokens": 1000,
                },
                test_filter={
                    "include_tags": [],
                    "exclude_tags": [],
                    "include_passing": False,
                },
            )

            mocked_execute.return_value = {"test2": "llm_result_2"}

            result = await _main(
                output=Path("output.xml"),
                log_level="DEBUG",
                config=None,
                report="/tmp/report.html",
                html_report=True,
                text_report=None,
                print_text_report=False,
                summarize_failures=False,
                include_passing=False,
                test_case_concurrency=None,
                chunk_concurrency=None,
                include_tags=None,
                exclude_tags=None,
            )

            mocked_get_results.assert_called_once_with(
                file_path=Path("output.xml"),
                include_tags=None,
                exclude_tags=None,
                exclude_fields=None,
                exclude_passing=True,
            )
            mocked_config.assert_called_once_with(None)

            mocked_execute.assert_called_once()
            assert mocked_execute.call_args.kwargs["results"] is fake_results

            mocked_html.assert_called_once_with(
                input_result_path=Path("output.xml"),
                llm_output_path="/tmp/report.html",
                llm_results={"test2": "llm_result_2"},
                model_info={"model": "openai/gpt-4"},
                overall_summary=None,
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_main_runs_ollama_init_for_ollama_models(self):
        """Test that Ollama init strategy runs for Ollama models."""
        with (
            patch("result_companion.entrypoints.run_rc.create_llm_html_log"),
            patch(
                "result_companion.api.execute_llm_and_get_results",
                return_value={},
            ),
            patch(
                "result_companion.entrypoints.run_rc.get_rc_robot_results",
                return_value=make_fake_results([], total=0),
            ),
            patch("result_companion.entrypoints.run_rc.load_config") as mocked_config,
            patch(
                "result_companion._internal.analysis_helpers.ollama_on_init_strategy"
            ) as mocked_init,
        ):
            mocked_config.return_value = DefaultConfigModel(
                version=1.0,
                llm_config={
                    "question_prompt": "question",
                    "prompt_template": "template {question} {context}",
                    "chunking": {
                        "chunk_analysis_prompt": "Analyze: {text}",
                        "final_synthesis_prompt": "Synthesize: {summary}",
                    },
                    "summary_prompt_template": "CI summary:\n{analyses}",
                },
                llm_factory={
                    "model": "ollama_chat/llama2:123",
                    "api_base": "http://localhost:11434",
                },
                tokenizer={"tokenizer": "ollama_tokenizer", "max_content_tokens": 1000},
            )

            await _main(
                output=Path("output.xml"),
                log_level="DEBUG",
                config=None,
                report=None,
                html_report=True,
                text_report=None,
                print_text_report=False,
                summarize_failures=False,
                include_passing=False,
            )

            mocked_init.assert_called_once_with(model_name="llama2")


class TestRunRC:
    """Tests for run_rc function."""

    def test_successfully_run_rc(self):
        """Test successful run_rc execution."""
        with patch(
            "result_companion.entrypoints.run_rc._main",
            return_value="RESULT",
        ) as mocked_main:
            result = run_rc(
                output=Path("output.xml"),
                log_level="DEBUG",
                config=None,
                report="/tmp/report.html",
                html_report=True,
                text_report=None,
                print_text_report=False,
                summarize_failures=False,
                include_passing=False,
                test_case_concurrency=None,
                chunk_concurrency=None,
                include_tags=None,
                exclude_tags=None,
                dryrun=False,
            )
            mocked_main.assert_called_once_with(
                output=Path("output.xml"),
                log_level="DEBUG",
                config=None,
                report="/tmp/report.html",
                html_report=True,
                text_report=None,
                json_report=None,
                print_text_report=False,
                summarize_failures=False,
                quiet=False,
                include_passing=False,
                test_case_concurrency=None,
                chunk_concurrency=None,
                include_tags=None,
                exclude_tags=None,
                dryrun=False,
                debug_log=None,
            )
            assert result == "RESULT"

    def test_run_rc_passes_tag_filters_to_main(self):
        """Test that tag filters are passed correctly."""
        with patch(
            "result_companion.entrypoints.run_rc._main",
            return_value=True,
        ) as mocked_main:
            run_rc(
                output=Path("output.xml"),
                log_level="DEBUG",
                config=None,
                report=None,
                html_report=True,
                text_report=None,
                print_text_report=False,
                summarize_failures=False,
                include_passing=False,
                include_tags=["smoke*", "critical"],
                exclude_tags=["wip"],
            )

            call_kwargs = mocked_main.call_args.kwargs
            assert call_kwargs["include_tags"] == ["smoke*", "critical"]
            assert call_kwargs["exclude_tags"] == ["wip"]

    def test_run_rc_passes_dryrun_flag(self):
        """Test that dryrun flag is passed correctly."""
        with patch(
            "result_companion.entrypoints.run_rc._main",
            return_value=True,
        ) as mocked_main:
            run_rc(
                output=Path("output.xml"),
                log_level="DEBUG",
                config=None,
                report=None,
                html_report=True,
                text_report=None,
                print_text_report=False,
                summarize_failures=False,
                include_passing=False,
                dryrun=True,
            )

            call_kwargs = mocked_main.call_args.kwargs
            assert call_kwargs["dryrun"] is True

    def test_run_rc_passes_quiet_flag(self):
        """Test that quiet flag is passed correctly."""
        with patch(
            "result_companion.entrypoints.run_rc._main",
            return_value=True,
        ) as mocked_main:
            run_rc(
                output=Path("output.xml"),
                log_level="DEBUG",
                config=None,
                report=None,
                include_passing=False,
                quiet=True,
            )

            call_kwargs = mocked_main.call_args.kwargs
            assert call_kwargs["quiet"] is True

    def test_run_rc_reraises_exception_on_failure(self):
        """Test that exceptions from _main propagate after logging."""
        with patch(
            "result_companion.entrypoints.run_rc._main",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                run_rc(
                    output=Path("output.xml"),
                    log_level="DEBUG",
                    config=None,
                    report=None,
                    include_passing=False,
                )


class TestEmitReports:
    """Tests for _emit_reports function."""

    def test_print_text_report_writes_to_stdout(self, capsys):
        fake_results = make_fake_results(["test_fail"])
        analysis_result = AnalysisResult(
            llm_results={"test_fail": "Root cause: timeout"},
            test_names=["test_fail"],
        )

        _emit_reports(
            output=Path("output.xml"),
            analysis_result=analysis_result,
            config=MagicMock(llm_factory=MagicMock(model="test-model")),
            results=fake_results,
            report=None,
            html_report=False,
            text_report=None,
            json_report=None,
            print_text_report=True,
        )

        captured = capsys.readouterr()
        assert "test_fail" in captured.out
        assert "Root cause: timeout" in captured.out


class TestMainJsonReport:
    """Tests for JSON report generation in _main."""

    @pytest.mark.asyncio
    async def test_main_writes_json_report_with_metadata(self, tmp_path):
        json_path = tmp_path / "report.json"

        with (
            patch("result_companion.entrypoints.run_rc.create_llm_html_log"),
            patch("result_companion.api.execute_llm_and_get_results") as mocked_execute,
            patch(
                "result_companion.entrypoints.run_rc.get_rc_robot_results"
            ) as mocked_get_results,
            patch("result_companion.entrypoints.run_rc.load_config") as mocked_config,
            patch(
                "result_companion._internal.analysis_helpers.run_provider_init_strategies"
            ),
        ):
            mocked_get_results.return_value = make_fake_results(["test_fail"], total=2)
            mocked_config.return_value = DefaultConfigModel(
                version=1.0,
                llm_config={
                    "question_prompt": "question prompt",
                    "prompt_template": "my_template {question} {context}",
                    "chunking": {
                        "chunk_analysis_prompt": "Analyze: {text}",
                        "final_synthesis_prompt": "Synthesize: {summary}",
                    },
                    "summary_prompt_template": "{analyses}",
                },
                llm_factory={"model": "openai/gpt-4", "api_key": "sk-test"},
                tokenizer={
                    "tokenizer": "openai_tokenizer",
                    "max_content_tokens": 1000,
                },
                test_filter={
                    "include_tags": [],
                    "exclude_tags": [],
                    "include_passing": False,
                },
            )
            mocked_execute.return_value = {"test_fail": "Root cause details"}

            await _main(
                output=Path("output.xml"),
                log_level="DEBUG",
                config=None,
                report=None,
                html_report=False,
                text_report=None,
                json_report=str(json_path),
                print_text_report=False,
                summarize_failures=False,
                include_passing=False,
            )

        import json

        data = json.loads(json_path.read_text())
        assert data["failed_test_count"] == 1
        assert data["analyzed_tests"] == ["test_fail"]
        assert data["per_test_results"]["test_fail"] == "Root cause details"
        assert data["model"] == "openai/gpt-4"
        assert data["source_file"] == "output.xml"
        assert data["total_test_count"] == 2
        assert len(data["source_hash"]) == 12
        assert data["timestamp"] is not None


class TestMainTextAndSynthesis:
    """Tests for text output and optional global synthesis."""

    @pytest.mark.asyncio
    async def test_main_writes_text_report_with_overall_summary(self, tmp_path):
        """Writes text file and includes synthesized summary."""
        text_report_path = tmp_path / "rc_summary.txt"

        with (
            patch(
                "result_companion.entrypoints.run_rc.create_llm_html_log"
            ) as mocked_html,
            patch("result_companion.api.execute_llm_and_get_results") as mocked_execute,
            patch(
                "result_companion.entrypoints.run_rc.get_rc_robot_results"
            ) as mocked_get_results,
            patch("result_companion.entrypoints.run_rc.load_config") as mocked_config,
            patch("result_companion.api.summarize_failures_with_llm") as mocked_summary,
            patch(
                "result_companion._internal.analysis_helpers.run_provider_init_strategies"
            ),
        ):
            mocked_get_results.return_value = make_fake_results(["test_fail"], total=1)
            mocked_config.return_value = DefaultConfigModel(
                version=1.0,
                llm_config={
                    "question_prompt": "question prompt",
                    "prompt_template": "my_template {question} {context}",
                    "chunking": {
                        "chunk_analysis_prompt": "Analyze: {text}",
                        "final_synthesis_prompt": "Synthesize: {summary}",
                    },
                    "summary_prompt_template": "CI summary:\n{analyses}",
                },
                llm_factory={
                    "model": "openai/gpt-4",
                    "api_key": "sk-test",
                },
                tokenizer={
                    "tokenizer": "openai_tokenizer",
                    "max_content_tokens": 1000,
                },
                test_filter={
                    "include_tags": [],
                    "exclude_tags": [],
                    "include_passing": False,
                },
            )
            mocked_execute.return_value = {"test_fail": "LLM details"}
            mocked_summary.return_value = "Shared root cause summary."

            result = await _main(
                output=Path("output.xml"),
                log_level="DEBUG",
                config=None,
                report=None,
                html_report=False,
                text_report=str(text_report_path),
                print_text_report=False,
                summarize_failures=True,
                include_passing=False,
            )

            assert result is True
            mocked_html.assert_not_called()
            mocked_summary.assert_called_once()
            text_content = text_report_path.read_text()
            assert "Shared root cause summary." in text_content
            assert "test_fail" in text_content
