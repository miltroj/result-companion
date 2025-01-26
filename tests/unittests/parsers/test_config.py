from result_companion.parsers.config import (
    ConfigLoader,
    DefaultConfigModel,
    LLMInitStrategyModel,
    LLMFactoryModel,
    TokenizerModel,
)
from pytest_mock import MockerFixture
import pytest
from pydantic import ValidationError
from unittest.mock import mock_open

prompt_template = {"prompt_template": "{question} {cotext}"}


def default_file_exists(*args, **kwargs) -> bool:
    return True


def test_reading_yaml_from_file(mocker: MockerFixture) -> None:
    mocked_data = mocker.mock_open(read_data="something: here")
    mocker.patch("builtins.open", mocked_data)
    assert ConfigLoader._read_yaml_file("mocker_open.yaml") == {"something": "here"}


def test_load_default_config(mocker: MockerFixture) -> None:
    mock_data = mocker.mock_open(
        read_data="version: 1.0\nllm_config:\n  question_prompt: Test prompt message.\n  prompt_template: question context\nllm_factory:\n  model_type: local_model\n  parameters: {}\ntokenizer:\n  ollama:\n  tokenizer: ollama_tokenizer\n  max_content_tokens: 1234"
    )
    mocker.patch("builtins.open", mock_data)

    config = ConfigLoader(
        default_config_file="default_config.yaml", file_exists=default_file_exists
    ).load_config()
    assert config.llm_config.question_prompt == "Test prompt message."
    assert config.llm_config.prompt_template == "question context"
    assert config.version == 1.0


def test_reading_existing_user_config_not_default(mocker: MockerFixture) -> None:
    mock_data = mocker.mock_open(
        read_data="version: 1.0\nllm_config:\n  question_prompt: User config.\n  prompt_template: question context\nllm_factory:\n  model_type: local_model\n  parameters: {}\ntokenizer:\n  ollama:\n  tokenizer: ollama_tokenizer\n  max_content_tokens: 1234"
    )
    mocker.patch("builtins.open", mock_data)
    config = ConfigLoader(
        default_config_file="default_config.yaml", file_exists=default_file_exists
    ).load_config("mocked_user_config.yaml")
    assert config.llm_config.question_prompt == "User config."
    assert config.llm_config.prompt_template == "question context"
    assert config.version == 1.0


def test_default_config_model_loads_parameters() -> None:
    config = DefaultConfigModel(
        version=1.0,
        **{
            "llm_config": {"question_prompt": "Test prompt message.", **prompt_template}
        },
        **{"llm_factory": {"model_type": "local", "parameters": {}}},
        **{"tokenizer": {"tokenizer": "ollama_tokenizer", "max_content_tokens": 1234}}
    )
    assert config.llm_config.question_prompt == "Test prompt message."
    assert config.llm_config.prompt_template == "{question} {cotext}"
    assert config.llm_factory.model_type == "local"
    assert config.llm_factory.parameters == {}
    assert config.llm_factory.strategy.parameters == {}
    assert config.tokenizer.tokenizer == "ollama_tokenizer"
    assert config.tokenizer.max_content_tokens == 1234
    assert config.version == 1.0


def test_default_config_model_drops_redundant_parameters() -> None:
    config = DefaultConfigModel(
        version=1.0,
        **{
            "llm_config": {"question_prompt": "Test prompt message.", **prompt_template}
        },
        redundant="redundant",
        **{"llm_factory": {"model_type": "local", "parameters": {}}},
        **{"tokenizer": {"tokenizer": "ollama_tokenizer", "max_content_tokens": 1234, "redundant": "redundant"}}
    )
    assert config.llm_config.question_prompt == "Test prompt message."
    assert config.llm_config.prompt_template == "{question} {cotext}"
    assert config.llm_factory.model_type == "local"
    assert config.llm_factory.parameters == {}
    assert config.llm_factory.strategy.parameters == {}
    assert config.tokenizer.tokenizer == "ollama_tokenizer"
    assert config.tokenizer.max_content_tokens == 1234
    assert config.version == 1.0


def test_user_llm_config_takes_precedense_over_default(mocker):
    default_config_content = """
    version: 1.0
    llm_config:
      question_prompt: "Default question prompt"
      prompt_template: "Default prompt template"
      model_type: "local"
    llm_factory:
      model_type: "OllamaLLM"
      parameters: {}
      strategy:
        parameters:
            custom: "strategy"
    tokenizer:
      tokenizer: "ollama_tokenizer"
      max_content_tokens: 1234
    """
    user_config_content = """
    llm_config:
      question_prompt: "User question prompt"
      model_type: "remote"
    llm_factory:
      model_type: "OverrideLLM"
      parameters: {"param1": "value1"}
      strategy:
        parameters:
            users: "user_strategy"
    tokenizer:
      tokenizer: "azure_openai_tokenizer"
      max_content_tokens: 4321
    """

    # Set the side_effect of mock_open to return different contents for different calls
    mock_open_instance = mock_open()
    mock_open_instance.side_effect = [
        mock_open(read_data=default_config_content).return_value,
        mock_open(read_data=user_config_content).return_value,
    ]

    # Patch the open function and file_exists function
    mocker.patch("builtins.open", mock_open_instance)

    # Load the config using the mocked file contents
    config_loader = ConfigLoader(
        default_config_file="default_config.yaml", file_exists=default_file_exists
    )
    config = config_loader.load_config(user_config_file="user_config.yaml")

    # Check that the user config takes precedence over the default config
    assert config.llm_config.question_prompt == "User question prompt"
    assert config.llm_config.prompt_template == "Default prompt template"
    assert config.llm_config.model_type == "remote"
    assert config.version == 1.0
    assert config.llm_factory.model_type == "OverrideLLM"
    assert config.llm_factory.parameters == {"param1": "value1"}
    assert config.llm_factory.strategy.parameters == {"users": "user_strategy"}
    assert config.tokenizer.tokenizer == "azure_openai_tokenizer"
    assert config.tokenizer.max_content_tokens == 4321


def test_initing_strategy_model_with_defaults() -> None:
    strategy = LLMInitStrategyModel()
    assert strategy.parameters == {}


def test_initing_strategy_model_with_extra_params() -> None:
    strategy = LLMInitStrategyModel(parameters={"param1": "value1"})
    assert strategy.parameters == {"param1": "value1"}


def test_init_factory_llm_model_with_defaults() -> None:
    factory = LLMFactoryModel(model_type="local", parameters={})
    assert factory.model_type == "local"
    assert factory.parameters == {}
    assert factory.strategy.parameters == {}


def test_init_factory_llm_model_with_extra_params() -> None:
    factory = LLMFactoryModel(
        model_type="local",
        parameters={"param1": "value1"},
        strategy={"parameters": {"custom": "strategy"}},
    )
    assert factory.model_type == "local"
    assert factory.parameters == {"param1": "value1"}
    assert factory.strategy.parameters == {"custom": "strategy"}


def test_default_config_model_loads_default_empty_strategy() -> None:
    config = DefaultConfigModel(
        version=1.0,
        **{
            "llm_config": {"question_prompt": "Test prompt message.", **prompt_template}
        },
        **{"llm_factory": {"model_type": "local", "parameters": {}}},
        **{"tokenizer": {"tokenizer": "ollama_tokenizer", "max_content_tokens": 1234}}
    )
    assert config.llm_config.question_prompt == "Test prompt message."
    assert config.llm_config.prompt_template == "{question} {cotext}"

    assert config.llm_factory.strategy.parameters == {}
    assert config.version == 1.0


def test_default_config_model_loads_custom_strategy() -> None:
    config = DefaultConfigModel(
        version=1.0,
        llm_config={"question_prompt": "Test prompt message.", **prompt_template},
        llm_factory={
            "model_type": "local",
            "parameters": {},
            "strategy": {"parameters": {"custom": "strategy"}},
        },
        tokenizer={"tokenizer": "ollama_tokenizer", "max_content_tokens": 1234},
    )
    assert config.llm_config.question_prompt == "Test prompt message."
    assert config.llm_config.prompt_template == "{question} {cotext}"
    assert config.llm_factory.strategy.parameters == {"custom": "strategy"}
    assert config.tokenizer.tokenizer == "ollama_tokenizer"
    assert config.tokenizer.max_content_tokens == 1234
    assert config.version == 1.0


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
