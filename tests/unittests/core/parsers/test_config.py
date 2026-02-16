import os
from pathlib import Path
from unittest import mock
from unittest.mock import mock_open

import pytest
from pydantic import ValidationError
from pytest_mock import MockerFixture

from result_companion.core.parsers.config import (
    ConcurrencyModel,
    ConfigLoader,
    DefaultConfigModel,
    LLMFactoryModel,
    TokenizerModel,
)

prompt_template = {"prompt_template": "{question} {cotext}"}

chunking_prompts = {
    "chunking": {
        "chunk_analysis_prompt": "Analyze: {text}",
        "final_synthesis_prompt": "Synthesize: {summary}",
    }
}


def test_reading_yaml_from_file(mocker: MockerFixture) -> None:
    mocked_data = mocker.mock_open(read_data="something: here")
    mocker.patch("builtins.open", mocked_data)
    assert ConfigLoader._read_yaml_file("mocker_open.yaml") == {"something": "here"}


def test_load_default_config(mocker: MockerFixture) -> None:
    mock_data = mocker.mock_open(
        read_data="version: 1.0\nllm_config:\n  question_prompt: Test prompt message.\n  prompt_template: question context\n  chunking:\n    chunk_analysis_prompt: 'Analyze: {text}'\n    final_synthesis_prompt: 'Synthesize: {summary}'\nllm_factory:\n  model: ollama_chat/llama2\n  parameters: {}\ntokenizer:\n  ollama:\n  tokenizer: ollama_tokenizer\n  max_content_tokens: 1234"
    )
    mocker.patch("builtins.open", mock_data)

    config = ConfigLoader(default_config_file="default_config.yaml").load_config()
    assert config.llm_config.question_prompt == "Test prompt message."
    assert config.llm_config.prompt_template == "question context"
    assert config.version == 1.0


def test_reading_existing_user_config_not_default(mocker: MockerFixture) -> None:
    mock_data = mocker.mock_open(
        read_data="version: 1.0\nllm_config:\n  question_prompt: User config.\n  prompt_template: question context\n  chunking:\n    chunk_analysis_prompt: 'Analyze: {text}'\n    final_synthesis_prompt: 'Synthesize: {summary}'\nllm_factory:\n  model: ollama_chat/llama2\n  parameters: {}\ntokenizer:\n  ollama:\n  tokenizer: ollama_tokenizer\n  max_content_tokens: 1234"
    )
    mocker.patch("builtins.open", mock_data)
    config = ConfigLoader(default_config_file="default_config.yaml").load_config(
        "mocked_user_config.yaml"
    )
    assert config.llm_config.question_prompt == "User config."
    assert config.llm_config.prompt_template == "question context"
    assert config.version == 1.0


def test_default_config_model_loads_parameters() -> None:
    config = DefaultConfigModel(
        version=1.0,
        **{
            "llm_config": {
                "question_prompt": "Test prompt message.",
                **prompt_template,
                **chunking_prompts,
            }
        },
        **{"llm_factory": {"model": "ollama_chat/llama2", "parameters": {}}},
        **{"tokenizer": {"tokenizer": "ollama_tokenizer", "max_content_tokens": 1234}},
    )
    assert config.llm_config.question_prompt == "Test prompt message."
    assert config.llm_config.prompt_template == "{question} {cotext}"
    assert config.llm_factory.model == "ollama_chat/llama2"
    assert config.llm_factory.parameters == {}
    assert config.tokenizer.tokenizer == "ollama_tokenizer"
    assert config.tokenizer.max_content_tokens == 1234
    assert config.version == 1.0


def test_default_config_model_drops_redundant_parameters() -> None:
    config = DefaultConfigModel(
        version=1.0,
        **{
            "llm_config": {
                "question_prompt": "Test prompt message.",
                **prompt_template,
                **chunking_prompts,
            }
        },
        redundant="redundant",
        **{"llm_factory": {"model": "openai/gpt-4", "parameters": {}}},
        **{
            "tokenizer": {
                "tokenizer": "ollama_tokenizer",
                "max_content_tokens": 1234,
                "redundant": "redundant",
            }
        },
    )
    assert config.llm_config.question_prompt == "Test prompt message."
    assert config.llm_config.prompt_template == "{question} {cotext}"
    assert config.llm_factory.model == "openai/gpt-4"
    assert config.llm_factory.parameters == {}
    assert config.tokenizer.tokenizer == "ollama_tokenizer"
    assert config.tokenizer.max_content_tokens == 1234
    assert config.version == 1.0


def test_user_llm_config_takes_precedense_over_default(mocker):
    default_config_content = """
    version: 1.0
    llm_config:
      question_prompt: "Default question prompt"
      prompt_template: "Default prompt template"
      chunking:
        chunk_analysis_prompt: "Analyze: {text}"
        final_synthesis_prompt: "Synthesize: {summary}"
    llm_factory:
      model: "ollama_chat/llama2"
      parameters: {}
    tokenizer:
      tokenizer: "ollama_tokenizer"
      max_content_tokens: 1234
    """
    user_config_content = """
    llm_config:
      question_prompt: "User question prompt"
    llm_factory:
      model: "openai/gpt-4"
      parameters: {"param1": "value1"}
    tokenizer:
      tokenizer: "azure_openai_tokenizer"
      max_content_tokens: 4321
    """

    mock_open_instance = mock_open()
    mock_open_instance.side_effect = [
        mock_open(read_data=default_config_content).return_value,
        mock_open(read_data=user_config_content).return_value,
    ]

    mocker.patch("builtins.open", mock_open_instance)

    config_loader = ConfigLoader(default_config_file=Path("default_config.yaml"))
    config = config_loader.load_config(user_config_file=Path("user_config.yaml"))

    assert config.llm_config.question_prompt == "User question prompt"
    assert config.llm_config.prompt_template == "Default prompt template"
    assert config.version == 1.0
    assert config.llm_factory.model == "openai/gpt-4"
    assert config.llm_factory.parameters == {"param1": "value1"}
    assert config.tokenizer.tokenizer == "azure_openai_tokenizer"
    assert config.tokenizer.max_content_tokens == 4321


def test_init_factory_llm_model_with_defaults() -> None:
    factory = LLMFactoryModel(model="ollama_chat/llama2", parameters={})
    assert factory.model == "ollama_chat/llama2"
    assert factory.parameters == {}


def test_init_factory_llm_model_with_extra_params() -> None:
    factory = LLMFactoryModel(
        model="openai/gpt-4",
        api_key="sk-test123",
        parameters={"param1": "value1"},
    )
    assert factory.model == "openai/gpt-4"
    assert factory.api_key == "sk-test123"
    assert factory.parameters == {"param1": "value1"}


def test_llm_factory_model_dump_masks_api_key() -> None:
    factory = LLMFactoryModel(
        model="openai/gpt-4",
        api_key="super-secret-key",
    )

    result = factory.model_dump()

    assert result["api_key"] == "***REDACTED***"
    assert result["model"] == "openai/gpt-4"


def test_llm_factory_model_is_sensitive_detects_sensitive_keys() -> None:
    factory = LLMFactoryModel(model="openai/gpt-4", parameters={})

    assert factory._is_sensitive("api_key") is True
    assert factory._is_sensitive("TOKEN") is True
    assert factory._is_sensitive("user") is False


def test_llm_factory_repr_masks_api_key() -> None:
    factory = LLMFactoryModel(
        model="openai/gpt-4",
        api_key="super-secret-key",
    )

    result = repr(factory)

    assert "super-secret-key" not in result
    assert "***REDACTED***" in result
    assert "openai/gpt-4" in result


def test_tokenizer_type_model_pass_on_existing_tokenizer() -> None:
    tokenizer = TokenizerModel(tokenizer="ollama_tokenizer", max_content_tokens=1234)
    assert tokenizer.tokenizer == "ollama_tokenizer"
    assert tokenizer.max_content_tokens == 1234


def test_tokenizer_type_model_fail_on_not_existing_tokenizer() -> None:
    with pytest.raises(ValidationError) as err:
        TokenizerModel(tokenizer="not_existing_tokenizer", max_content_tokens=1234)
    assert "1 validation error for TokenizerModel" in str(err.value)


def test_tokenizer_type_model_fail_on_negative_max_content_tokens() -> None:
    with pytest.raises(ValidationError) as err:
        TokenizerModel(tokenizer="ollama_tokenizer", max_content_tokens=-1234)
    assert "1 validation error for TokenizerModel" in str(err.value)


def test_expand_config_no_env_vars():
    """Test expanding environment variables in strings."""
    assert ConfigLoader()._expand_env_vars("simple_string") == "simple_string"


def test_expand_config_with_env_vars():
    config_loader = ConfigLoader()

    with mock.patch.dict(os.environ, {"TEST_VAR": "test_value"}):
        assert config_loader._expand_env_vars("${TEST_VAR}") == "test_value"
        assert (
            config_loader._expand_env_vars("prefix_${TEST_VAR}_suffix")
            == "prefix_test_value_suffix"
        )


def test_expand_config_with_multiple_env_vars():
    config_loader = ConfigLoader()
    with mock.patch.dict(os.environ, {"VAR1": "value1", "VAR2": "value2"}):
        assert config_loader._expand_env_vars("${VAR1}_${VAR2}") == "value1_value2"


def test_expand_config_with_not_existing_env_var():
    config_loader = ConfigLoader()
    with mock.patch.dict(os.environ, {}, clear=True):
        assert config_loader._expand_env_vars("${NON_EXISTENT}") == "${NON_EXISTENT}"


def test_process_env_vars():
    """Test processing environment variables in different data structures."""
    config_loader = ConfigLoader()

    with mock.patch.dict(os.environ, {"API_KEY": "secret_key"}):
        assert config_loader._process_env_vars("${API_KEY}") == "secret_key"

        assert config_loader._process_env_vars(["item1", "${API_KEY}"]) == [
            "item1",
            "secret_key",
        ]

        test_dict = {
            "key1": "value1",
            "key2": "${API_KEY}",
            "nested": {"nested_key": "${API_KEY}_suffix"},
            "list_key": ["item1", "${API_KEY}"],
        }

        processed = config_loader._process_env_vars(test_dict)

        assert processed["key1"] == "value1"
        assert processed["key2"] == "secret_key"
        assert processed["nested"]["nested_key"] == "secret_key_suffix"
        assert processed["list_key"][0] == "item1"
        assert processed["list_key"][1] == "secret_key"


def test_load_config_with_env_vars(mocker):
    """Test loading configuration with environment variables."""
    default_config_content = """
    version: 1.0
    llm_config:
      question_prompt: "Default question prompt"
      prompt_template: "Default prompt template"
      chunking:
        chunk_analysis_prompt: "Analyze: {text}"
        final_synthesis_prompt: "Synthesize: {summary}"
    llm_factory:
      model: "ollama_chat/llama3"
      parameters:
        temperature: 0.7
    tokenizer:
      tokenizer: "ollama_tokenizer"
      max_content_tokens: 1000
    """

    user_config_content = """
    llm_factory:
      model: "gemini/gemini-pro"
      api_key: "${GOOGLE_API_KEY}"
    tokenizer:
      tokenizer: "google_tokenizer"
      max_content_tokens: 4000
    """

    mock_open_instance = mock_open()
    mock_open_instance.side_effect = [
        mock_open(read_data=default_config_content).return_value,
        mock_open(read_data=user_config_content).return_value,
    ]

    mocker.patch("builtins.open", mock_open_instance)

    with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-api-key-12345"}):
        config_loader = ConfigLoader(default_config_file=Path("default_config.yaml"))
        config = config_loader.load_config(user_config_file=Path("user_config.yaml"))

        assert config.llm_factory.api_key == "fake-api-key-12345"
        assert config.llm_factory.model == "gemini/gemini-pro"
        assert config.tokenizer.tokenizer == "google_tokenizer"
        assert config.tokenizer.max_content_tokens == 4000
        assert config.llm_config.question_prompt == "Default question prompt"
        assert config.llm_config.prompt_template == "Default prompt template"


def test_concurrency_model_with_defaults():
    concurrency = ConcurrencyModel()
    assert concurrency.test_case == 1
    assert concurrency.chunk == 1


def test_default_config_model_loads_custom_concurrency():
    config = DefaultConfigModel(
        version=1.0,
        llm_config={
            "question_prompt": "Test prompt.",
            **prompt_template,
            **chunking_prompts,
        },
        llm_factory={"model": "ollama_chat/llama2", "parameters": {}},
        tokenizer={"tokenizer": "ollama_tokenizer", "max_content_tokens": 1000},
        concurrency={"test_case": 5, "chunk": 3},
    )
    assert config.concurrency.test_case == 5
    assert config.concurrency.chunk == 3


def test_default_config_model_loads_chunking_prompts():
    """Test that chunking prompts are properly loaded."""
    config = DefaultConfigModel(
        version=1.0,
        llm_config={
            "question_prompt": "Test prompt.",
            **prompt_template,
            **chunking_prompts,
        },
        llm_factory={"model": "ollama_chat/llama2", "parameters": {}},
        tokenizer={"tokenizer": "ollama_tokenizer", "max_content_tokens": 1000},
    )
    assert config.llm_config.chunking.chunk_analysis_prompt == "Analyze: {text}"
    assert config.llm_config.chunking.final_synthesis_prompt == "Synthesize: {summary}"


def test_user_config_can_override_chunking_prompts(mocker):
    """Test that user config can override chunking prompts."""
    default_config_content = """
    version: 1.0
    llm_config:
      question_prompt: "Default prompt"
      prompt_template: "Default template"
      chunking:
        chunk_analysis_prompt: "Default analyze: {text}"
        final_synthesis_prompt: "Default synthesize: {summary}"
    llm_factory:
      model: "ollama_chat/llama2"
      parameters: {}
    tokenizer:
      tokenizer: "ollama_tokenizer"
      max_content_tokens: 1000
    """

    user_config_content = """
    llm_config:
      chunking:
        chunk_analysis_prompt: "Custom analyze: {text}"
        final_synthesis_prompt: "Custom synthesize: {summary}"
    """

    mock_open_instance = mock_open()
    mock_open_instance.side_effect = [
        mock_open(read_data=default_config_content).return_value,
        mock_open(read_data=user_config_content).return_value,
    ]

    mocker.patch("builtins.open", mock_open_instance)

    config_loader = ConfigLoader(default_config_file=Path("default_config.yaml"))
    config = config_loader.load_config(user_config_file=Path("user_config.yaml"))

    assert config.llm_config.chunking.chunk_analysis_prompt == "Custom analyze: {text}"
    assert (
        config.llm_config.chunking.final_synthesis_prompt
        == "Custom synthesize: {summary}"
    )
