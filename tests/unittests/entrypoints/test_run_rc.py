from pathlib import Path
from unittest.mock import patch

import pytest

from result_companion.core.parsers.config import DefaultConfigModel
from result_companion.core.utils.log_levels import LogLevels
from result_companion.entrypoints.run_rc import (
    _main,
    _register_copilot_if_needed,
    _run_ollama_init_strategy,
    _run_provider_init_strategies,
    run_rc,
)


class TestRunOllamaInitStrategy:
    """Tests for _run_ollama_init_strategy function."""

    @pytest.fixture(autouse=True)
    def _capture_ollama_init_calls(self, monkeypatch):
        calls = []

        def _fake_ollama_on_init_strategy(*, model_name: str):
            calls.append(model_name)

        monkeypatch.setattr(
            "result_companion.entrypoints.run_rc.ollama_on_init_strategy",
            _fake_ollama_on_init_strategy,
        )
        self.calls = calls

    def test_skips_non_ollama_models(self):
        """Test that non-Ollama models are skipped."""
        _run_ollama_init_strategy(
            model_name="openai/gpt-4",
        )
        assert self.calls == []

    def test_runs_for_ollama_models(self):
        """Test that Ollama models trigger init strategy."""
        _run_ollama_init_strategy(
            model_name="ollama_chat/llama2",
        )
        assert self.calls == ["llama2"]

    def test_extracts_model_name_from_identifier(self):
        """Test that model name is extracted when not in strategy params."""
        _run_ollama_init_strategy(
            model_name="ollama_chat/deepseek-r1:1.5b",
        )
        assert self.calls == ["deepseek-r1"]


class TestRunProviderInitStrategies:
    """Tests for generic provider strategy dispatcher."""

    @pytest.fixture(autouse=True)
    def _capture_provider_init_calls(self, monkeypatch):
        ollama_calls = []
        copilot_calls = []

        def _fake_run_ollama_init_strategy(model_name: str):
            ollama_calls.append(model_name)

        def _fake_register_copilot_if_needed(model_name: str):
            copilot_calls.append(model_name)

        monkeypatch.setattr(
            "result_companion.entrypoints.run_rc._run_ollama_init_strategy",
            _fake_run_ollama_init_strategy,
        )
        monkeypatch.setattr(
            "result_companion.entrypoints.run_rc._register_copilot_if_needed",
            _fake_register_copilot_if_needed,
        )
        self.ollama_calls = ollama_calls
        self.copilot_calls = copilot_calls

    def test_runs_ollama_strategy_for_ollama_chat(self):
        _run_provider_init_strategies(
            model_name="ollama_chat/deepseek-r1:1.5b",
        )
        assert self.ollama_calls == ["ollama_chat/deepseek-r1:1.5b"]
        assert self.copilot_calls == []

    def test_runs_copilot_provider_init_strategy(self):
        _run_provider_init_strategies(
            model_name="copilot_sdk/gpt-5-mini",
        )
        assert self.copilot_calls == ["copilot_sdk/gpt-5-mini"]
        assert self.ollama_calls == []

    def test_skips_for_unmapped_providers(self):
        _run_provider_init_strategies(
            model_name="openai/gpt-4o",
        )
        assert self.ollama_calls == []
        assert self.copilot_calls == []


class TestRegisterCopilotIfNeeded:
    """Tests for _register_copilot_if_needed function."""

    def test_registers_for_copilot_models(self):
        with patch(
            "result_companion.entrypoints.run_rc.register_copilot_provider"
        ) as mocked_register:
            _register_copilot_if_needed("copilot_sdk/gpt-4.1")

        mocked_register.assert_called_once_with()


class TestMainE2E:
    """End-to-end tests for _main function."""

    @pytest.mark.asyncio
    async def test_main_executes_analysis(self):
        """Test that _main executes the full analysis flow."""
        with (
            patch(
                "result_companion.entrypoints.run_rc.create_llm_html_log"
            ) as mocked_html,
            patch(
                "result_companion.entrypoints.run_rc.execute_llm_and_get_results"
            ) as mocked_execute,
            patch(
                "result_companion.entrypoints.run_rc.get_robot_results_from_file_as_dict"
            ) as mocked_get_results,
            patch("result_companion.entrypoints.run_rc.load_config") as mocked_config,
        ):
            mocked_get_results.return_value = [
                {"name": "test1", "status": "PASS", "tags": []},
                {"name": "test2", "status": "FAIL", "tags": []},
            ]

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
                print_text_summary=False,
                summarize_failures=False,
                include_passing=False,
                test_case_concurrency=None,
                chunk_concurrency=None,
                include_tags=None,
                exclude_tags=None,
            )

            # Verify the flow
            mocked_get_results.assert_called_once_with(
                file_path=Path("output.xml"),
                log_level=LogLevels.DEBUG,
                include_tags=None,
                exclude_tags=None,
            )
            mocked_config.assert_called_once_with(None)

            # Should filter out PASS test
            mocked_execute.assert_called_once()
            call_args = mocked_execute.call_args
            assert call_args.kwargs["test_cases"] == [
                {"name": "test2", "status": "FAIL", "tags": []}
            ]

            mocked_html.assert_called_once_with(
                input_result_path=Path("output.xml"),
                llm_output_path="/tmp/report.html",
                llm_results={"test2": "llm_result_2"},
                model_info={"model": "openai/gpt-4"},
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_main_runs_ollama_init_for_ollama_models(self):
        """Test that Ollama init strategy runs for Ollama models."""
        with (
            patch("result_companion.entrypoints.run_rc.create_llm_html_log"),
            patch(
                "result_companion.entrypoints.run_rc.execute_llm_and_get_results",
                return_value={},
            ),
            patch(
                "result_companion.entrypoints.run_rc.get_robot_results_from_file_as_dict",
                return_value=[],
            ),
            patch("result_companion.entrypoints.run_rc.load_config") as mocked_config,
            patch(
                "result_companion.entrypoints.run_rc.ollama_on_init_strategy"
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
                print_text_summary=False,
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
                print_text_summary=False,
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
                print_text_summary=False,
                summarize_failures=False,
                include_passing=False,
                test_case_concurrency=None,
                chunk_concurrency=None,
                include_tags=None,
                exclude_tags=None,
                dryrun=False,
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
                print_text_summary=False,
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
                print_text_summary=False,
                summarize_failures=False,
                include_passing=False,
                dryrun=True,
            )

            call_kwargs = mocked_main.call_args.kwargs
            assert call_kwargs["dryrun"] is True


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
            patch(
                "result_companion.entrypoints.run_rc.execute_llm_and_get_results"
            ) as mocked_execute,
            patch(
                "result_companion.entrypoints.run_rc.get_robot_results_from_file_as_dict"
            ) as mocked_get_results,
            patch("result_companion.entrypoints.run_rc.load_config") as mocked_config,
            patch(
                "result_companion.entrypoints.run_rc.summarize_failures_with_llm"
            ) as mocked_summary,
        ):
            mocked_get_results.return_value = [
                {"name": "test_fail", "status": "FAIL", "tags": []},
            ]
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
                print_text_summary=False,
                summarize_failures=True,
                include_passing=False,
            )

            assert result is True
            mocked_html.assert_not_called()
            mocked_summary.assert_called_once()
            text_content = text_report_path.read_text()
            assert "Shared root cause summary." in text_content
            assert "test_fail" in text_content
