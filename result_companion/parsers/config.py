import os
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError

from result_companion.utils.logging_config import setup_logging

logger = setup_logging("CONFIG")


class ModelType(str, Enum):
    LOCAL = "local"
    REMOTE = "remote"


class TokenizerTypes(str, Enum):
    AZURE_OPENAI = "azure_openai_tokenizer"
    OLLAMA = "ollama_tokenizer"
    BEDROCK = "bedrock_tokenizer"


class CustomEndpointModel(BaseModel):
    azure_deployment: str = Field(min_length=5, description="Azure deployment URL.")
    azure_endpoint: str
    openai_api_version: str = Field(
        min_length=5, description="OpenAI API version.", default="2023-03-15-preview"
    )
    openai_api_type: str = Field(
        min_length=5, description="OpenAI API type.", default="azure"
    )
    openai_api_key: str = Field(min_length=5, description="OpenAI API key.")


class LLMConfigModel(BaseModel):
    question_prompt: str = Field(min_length=5, description="User prompt.")
    prompt_template: str = Field(
        min_length=5, description="Template for LLM ChatPromptTemplate."
    )
    model_type: ModelType = Field(
        default=ModelType.LOCAL,
        description=f"Which type of llm model runners to use {[el.name for el in ModelType]}",
    )


class LLMInitStrategyModel(BaseModel):
    parameters: dict = Field(default={}, description="Strategy parameters.")


class LLMFactoryModel(BaseModel):
    model_type: str = Field(min_length=5, description="Model type.")
    parameters: dict = Field(default={}, description="Model parameters.")
    strategy: LLMInitStrategyModel = Field(
        description="Strategy to run on init.", default_factory=LLMInitStrategyModel
    )


class TokenizerModel(BaseModel):
    tokenizer: TokenizerTypes
    max_content_tokens: int = Field(ge=0, description="Max content tokens.")


class DefaultConfigModel(BaseModel):
    version: float
    llm_config: LLMConfigModel
    llm_factory: LLMFactoryModel
    tokenizer: TokenizerModel


class CustomModelEndpointConfig(DefaultConfigModel):
    custom_endpoint: CustomEndpointModel


class ConfigLoader:
    def __init__(
        self,
        default_config_file: Path | None = None,
    ):
        self.default_config_file = default_config_file

    @staticmethod
    def _read_yaml_file(file_path: Path) -> dict:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)

    def load_config(self, user_config_file: Path = None) -> DefaultConfigModel:
        """Load and validate the YAML configuration file, with defaults."""
        default_config = self._read_yaml_file(self.default_config_file)
        if user_config_file:
            user_config = self._read_yaml_file(user_config_file)
        else:
            logger.info(
                "User configuration not found or not provided. Using default configuration!"
            )
            logger.debug({self.default_config_file})

        # TODO: improve unpacking
        config_data = (
            {
                "version": default_config.get("version"),
                "llm_config": {
                    **default_config.get("llm_config", {}),
                    **user_config.get("llm_config", {}),
                },
                "llm_factory": {
                    **default_config.get("llm_factory", {}),
                    **user_config.get("llm_factory", {}),
                },
                "tokenizer": {
                    **default_config.get("tokenizer", {}),
                    **user_config.get("tokenizer", {}),
                },
            }
            if user_config_file
            else default_config
        )
        try:
            validated_config = DefaultConfigModel(**config_data)
        except ValidationError as e:
            logger.error(f"Configuration validation failed:\n{e}")
            raise
        return validated_config


def load_config(config_path: Path | None = None) -> DefaultConfigModel:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(current_dir, "..", "configs", "default_config.yaml")

    config_loader = ConfigLoader(default_config_file=config_file_path)
    config = config_loader.load_config(user_config_file=config_path)
    logger.debug(f"{config=}")
    return config


# TODO: remove this code
# Example usage in a CLI application
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    # Define default config path (assumes it's in the package directory)
    PACKAGE_DIR = Path(__file__).resolve().parent
    DEFAULT_CONFIG_PATH = PACKAGE_DIR / "config" / "config.yaml"

    parser = argparse.ArgumentParser(description="CLI Application with Config")
    parser.add_argument(
        "--config", type=str, help="Path to the YAML configuration file (optional)."
    )
    args = parser.parse_args()

    config_loader = ConfigLoader(default_config_file=str(DEFAULT_CONFIG_PATH))
    try:
        config = config_loader.load_config(user_config_file=args.config)
        print(f"Config loaded successfully: {config}")
        # Use the `config.prompt` in your CLI application logic
        print(f"Prompt: {config.prompt}")
    except Exception as e:
        print(f"Failed to load configuration: {e}")
