from pathlib import Path
from unittest.mock import patch

import pytest

from result_companion.core.parsers.config import DefaultConfigModel
from result_companion.core.utils.log_levels import LogLevels
from result_companion.entrypoints.run_rc import (
    _main,
    _register_copilot_if_needed,
    _run_ollama_init_strategy,
    run_rc,
)


class TestRunOllamaInitStrategy:
    """Tests for _run_ollama_init_strategy function."""

    def test_skips_non_ollama_models(self):
        """Test that non-Ollama models are skipped."""
        with patch(
            "result_companion.entrypoints.run_rc.ollama_on_init_strategy"
        ) as mock_init:
            _run_ollama_init_strategy(
                model_name="openai/gpt-4",
                strategy_params={},
            )
            mock_init.assert_not_called()

    def test_runs_for_ollama_models(self):
        """Test that Ollama models trigger init strategy."""
        with patch(
            "result_companion.entrypoints.run_rc.ollama_on_init_strategy"
        ) as mock_init:
            _run_ollama_init_strategy(
                model_name="ollama_chat/llama2",
                strategy_params={"model_name": "llama2"},
            )
            mock_init.assert_called_once_with(model_name="llama2")

    def test_extracts_model_name_from_identifier(self):
        """Test that model name is extracted when not in strategy params."""
        with patch(
            "result_companion.entrypoints.run_rc.ollama_on_init_strategy"
        ) as mock_init:
            _run_ollama_init_strategy(
                model_name="ollama_chat/deepseek-r1:1.5b",
                strategy_params={},
            )
            mock_init.assert_called_once_with(model_name="deepseek-r1")


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
                "result_companion.entrypoints.run_rc._run_ollama_init_strategy"
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
                },
                llm_factory={
                    "model": "ollama_chat/llama2",
                    "api_base": "http://localhost:11434",
                    "strategy": {"parameters": {"model_name": "llama2"}},
                },
                tokenizer={"tokenizer": "ollama_tokenizer", "max_content_tokens": 1000},
            )

            await _main(
                output=Path("output.xml"),
                log_level="DEBUG",
                config=None,
                report=None,
                include_passing=False,
            )

            mocked_init.assert_called_once_with(
                model_name="ollama_chat/llama2",
                strategy_params={"model_name": "llama2"},
            )


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
                include_passing=False,
                dryrun=True,
            )

            call_kwargs = mocked_main.call_args.kwargs
            assert call_kwargs["dryrun"] is True
