import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.prompts import ChatPromptTemplate

from result_companion.core.parsers.config import (
    DefaultConfigModel,
    LLMConfigModel,
    LLMFactoryModel,
    LLMInitStrategyModel,
    ModelType,
    TokenizerModel,
    TokenizerTypes,
)
from result_companion.core.utils.log_levels import LogLevels
from result_companion.entrypoints.run_rc import (
    _main,
    init_llm_with_strategy_factory,
    run_rc,
)


def test_init_llm_model_without_setup_strategy():
    config = LLMFactoryModel(
        **{
            "model_type": "AzureChatOpenAI",
            "parameters": {
                "azure_deployment": "deployment",
                "azure_endpoint": "endpoint",
                "openai_api_version": "version",
                "openai_api_type": "azure",
                "openai_api_key": "key",
            },
        }
    )
    model, strategy = init_llm_with_strategy_factory(config)
    assert model.__class__.__name__ == "AzureChatOpenAI"
    assert strategy is None


def test_init_llm_model_with_setup_strategy():
    config = LLMFactoryModel(
        **{
            "model_type": "OllamaLLM",
            "parameters": {
                "model": "llama3.2",
            },
        }
    )
    model, strategy = init_llm_with_strategy_factory(config)
    assert model.__class__.__name__ == "OllamaLLM"
    assert strategy.__name__ == "ollama_on_init_strategy"


def test_init_llm_model_for_google_gemini():
    config = LLMFactoryModel(
        **{
            "model_type": "ChatGoogleGenerativeAI",
            "parameters": {
                "model": "gemini-pro",
                "google_api_key": "fake-api-key",
            },
        }
    )
    model, strategy = init_llm_with_strategy_factory(config)
    assert model.__class__.__name__ == "ChatGoogleGenerativeAI"
    assert strategy is None


def test_fail_llm_init_on_unsupported_llm_parameters():
    config = LLMFactoryModel(
        **{
            "model_type": "BedrockLLM",
            "parameters": {
                "unsupported_param": "param",
            },
        }
    )
    with pytest.raises(ValueError) as e:
        init_llm_with_strategy_factory(config)
    assert (
        str(e.value)
        == "Invalid parameters for BedrockLLM: {'unsupported_param': 'param'}, while available parameters are: {'args': typing.Any, 'kwargs': typing.Any, 'return': None}"
    )


def test_fail_llm_init_on_unsupported_google_parameters():
    config = LLMFactoryModel(
        **{
            "model_type": "ChatGoogleGenerativeAI",
            "parameters": {
                "unsupported_param": "param",
                "google_api_key": "fake-api-key",
            },
        }
    )
    with pytest.raises(ValueError) as e:
        init_llm_with_strategy_factory(config)
    assert "Invalid parameters for ChatGoogleGenerativeAI" in str(e.value)


def test_fail_init_llm_model_for_unsupported_model():
    config = LLMFactoryModel(
        **{
            "model_type": "UnsupportedModel",
            "parameters": {
                "model": "crazyModel",
            },
        }
    )
    with pytest.raises(ValueError) as e:
        init_llm_with_strategy_factory(config)
    assert (
        str(e.value)
        == "Unsupported model type: UnsupportedModel not in dict_keys(['OllamaLLM', 'AzureChatOpenAI', 'BedrockLLM', 'ChatGoogleGenerativeAI'])"
    )


def test_fail_init_llm_model_for_case_sensitive_model():
    config = LLMFactoryModel(
        **{
            "model_type": "ollamallm",  # lowercase instead of OllamaLLM
            "parameters": {
                "model": "llama3",
            },
        }
    )
    with pytest.raises(ValueError) as e:
        init_llm_with_strategy_factory(config)
    assert (
        str(e.value)
        == "Unsupported model type: ollamallm not in dict_keys(['OllamaLLM', 'AzureChatOpenAI', 'BedrockLLM', 'ChatGoogleGenerativeAI'])"
    )


@patch(
    "result_companion.entrypoints.run_rc.create_llm_html_log",
    autospec=True,
)
@patch(
    "result_companion.entrypoints.run_rc.AzureChatOpenAI",
    autospec=True,
)
@patch(
    "result_companion.entrypoints.run_rc.execute_llm_and_get_results",
    autospec=True,
)
@patch(
    "result_companion.entrypoints.run_rc.get_robot_results_from_file_as_dict",
    autospec=True,
)
@patch("result_companion.entrypoints.run_rc.load_config", autospec=True)
def test_main_e2e_execution(
    mock_config_loading,
    mocked_get_robot_results,
    mocked_execute_llm_chain,
    mocked_azure_model,
    mocked_html_creation,
):
    mocked_get_robot_results.return_value = {
        "tests": [
            {"name": "test1", "status": "PASS"},
            {"name": "test2", "status": "FAIL"},
        ]
    }

    mock_config_loading.return_value = DefaultConfigModel(
        version=1.0,
        llm_config={
            "question_prompt": "question prompt",
            "prompt_template": "my_template {question}",
            "model_type": "local",
        },
        llm_factory={
            "model_type": "AzureChatOpenAI",
            "parameters": {
                "azure_deployment": "deployment",
                "azure_endpoint": "endpoint",
                "openai_api_version": "version",
                "openai_api_type": "azure",
                "openai_api_key": "key",
            },
        },
        tokenizer={"tokenizer": "azure_openai_tokenizer", "max_content_tokens": 1000},
    )

    mocked_execute_llm_chain.return_value = {
        "test2": "llm_result_2",
    }

    result = asyncio.run(
        _main(
            output=Path("output.xml"),
            log_level="DEBUG",
            config=None,
            report="/tmp/report.html",
            include_passing=False,
        )
    )

    mocked_get_robot_results.assert_called_once_with(
        file_path=Path("output.xml"), log_level=LogLevels.TRACE
    )
    mock_config_loading.assert_called_once_with(None)

    mocked_execute_llm_chain.assert_called_once_with(
        {
            "tests": [
                {"name": "test1", "status": "PASS"},
                {"name": "test2", "status": "FAIL"},
            ]
        },
        DefaultConfigModel(
            version=1.0,
            llm_config=LLMConfigModel(
                question_prompt="question prompt",
                prompt_template="my_template {question}",
                model_type=ModelType.LOCAL,
            ),
            llm_factory=LLMFactoryModel(
                model_type="AzureChatOpenAI",
                parameters={
                    "azure_deployment": "deployment",
                    "azure_endpoint": "endpoint",
                    "openai_api_version": "version",
                    "openai_api_type": "azure",
                    "openai_api_key": "key",
                },
                strategy=LLMInitStrategyModel(parameters={}),
            ),
            tokenizer=TokenizerModel(
                tokenizer=TokenizerTypes.AZURE_OPENAI, max_content_tokens=1000
            ),
        ),
        ChatPromptTemplate.from_template("my_template {question}"),
        mocked_azure_model(),
        include_passing=False,
    )
    mocked_html_creation.assert_called_once_with(
        input_result_path=Path("output.xml"),
        llm_output_path="/tmp/report.html",
        llm_results={
            "test2": "llm_result_2",
        },
    )
    assert result is True


def test_succesfully_run_rc():
    with patch(
        "result_companion.entrypoints.run_rc._main",
        return_value="RESULT",
        autospec=True,
    ) as mocked_main:
        result = run_rc(
            output=Path("output.xml"),
            log_level="DEBUG",
            config=None,
            report="/tmp/report.html",
            include_passing=False,
        )
        mocked_main.assert_called_once_with(
            output=Path("output.xml"),
            log_level="DEBUG",
            config=None,
            report="/tmp/report.html",
            include_passing=False,
        )
        assert result == "RESULT"
