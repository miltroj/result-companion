import os
import re
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError, model_serializer

from result_companion.core.utils.logging_config import logger


class TokenizerTypes(str, Enum):
    AZURE_OPENAI = "azure_openai_tokenizer"
    OLLAMA = "ollama_tokenizer"
    BEDROCK = "bedrock_tokenizer"
    GOOGLE = "google_tokenizer"
    OPENAI = "openai_tokenizer"
    ANTHROPIC = "anthropic_tokenizer"


class ChunkingPromptsModel(BaseModel):
    chunk_analysis_prompt: str = Field(
        min_length=5, description="Prompt for analyzing individual chunks."
    )
    final_synthesis_prompt: str = Field(
        min_length=5, description="Prompt for synthesizing chunk summaries."
    )


class LLMConfigModel(BaseModel):
    question_prompt: str = Field(min_length=5, description="User prompt.")
    prompt_template: str = Field(
        min_length=5, description="Template for LLM prompt formatting."
    )
    # TODO: change name to summary_prompt_template
    summary_prompt_template: str = Field(
        min_length=5,
        description="Template for overall failed-tests synthesis prompt.",
    )
    chunking: ChunkingPromptsModel


class LLMFactoryModel(BaseModel):
    """LiteLLM model configuration.

    Model naming convention:
    - Ollama: ollama_chat/model-name or ollama/model-name
    - OpenAI: openai/gpt-4o or gpt-4o
    - Azure: azure/deployment-name
    - Anthropic: anthropic/claude-3-sonnet
    - Google: gemini/gemini-2.0-flash
    - Bedrock: bedrock/anthropic.claude-v2
    """

    model: str = Field(
        min_length=3, description="LiteLLM model identifier (e.g., ollama_chat/llama2)"
    )
    api_base: str | None = Field(default=None, description="Optional API base URL.")
    api_key: str | None = Field(default=None, description="Optional API key.")
    parameters: dict = Field(default={}, description="Additional model parameters.")

    def _is_sensitive(self, key: str) -> bool:
        """Checks if a key is sensitive."""
        sensitive_keys = {"api_key", "token", "password", "secret", "auth"}
        return any(s in key.lower() for s in sensitive_keys)

    def __repr__(self) -> str:
        """Returns string representation with masked sensitive fields."""
        api_key_display = "***REDACTED***" if self.api_key else None
        return (
            f"LLMFactoryModel(model={self.model!r}, "
            f"api_base={self.api_base!r}, "
            f"api_key={api_key_display!r})"
        )

    @model_serializer
    def _mask_sensitive_fields(self) -> dict:
        """Masks sensitive fields for serialization."""
        return {
            "model": self.model,
            "api_base": self.api_base,
            "api_key": "***REDACTED***" if self.api_key else None,
            "parameters": self.parameters,
        }


class TokenizerModel(BaseModel):
    tokenizer: TokenizerTypes
    max_content_tokens: int = Field(ge=0, description="Max content tokens.")


class ConcurrencyModel(BaseModel):
    test_case: int = Field(
        default=1, ge=1, description="Test cases processed in parallel."
    )
    chunk: int = Field(
        default=1, ge=1, description="Chunks processed in parallel per test case."
    )


class TestFilterModel(BaseModel):
    """Test filtering config - passed to RF's native result.configure()."""

    include_tags: list[str] = Field(default=[], description="RF --include patterns.")
    exclude_tags: list[str] = Field(default=[], description="RF --exclude patterns.")
    include_passing: bool = Field(default=False, description="Include PASS tests.")


class DefaultConfigModel(BaseModel):
    version: float
    llm_config: LLMConfigModel
    llm_factory: LLMFactoryModel
    tokenizer: TokenizerModel
    concurrency: ConcurrencyModel = Field(default_factory=ConcurrencyModel)
    test_filter: TestFilterModel = Field(default_factory=TestFilterModel)


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

    @staticmethod
    def _expand_env_vars(value):
        """Expand environment variables in a string using ${VAR} syntax."""
        if isinstance(value, str) and "${" in value and "}" in value:
            pattern = re.compile(r"\${([^}^{]+)}")
            matches = pattern.findall(value)
            for match in matches:
                env_var = os.environ.get(match)
                if env_var:
                    value = value.replace(f"${{{match}}}", env_var)
                else:
                    logger.warning(f"Environment variable '{match}' not found")
            return value
        return value

    def _process_env_vars(self, data):
        """Recursively process environment variables in the configuration data."""
        if isinstance(data, dict):
            return {k: self._process_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._process_env_vars(item) for item in data]
        else:
            return self._expand_env_vars(data)

    def load_config(self, user_config_file: Path = None) -> DefaultConfigModel:
        """Load and validate the YAML configuration file, with defaults."""
        default_config = self._read_yaml_file(self.default_config_file)
        # Process environment variables in default config
        default_config = self._process_env_vars(default_config)

        if user_config_file:
            user_config = self._read_yaml_file(user_config_file)
            # Process environment variables in user config
            user_config = self._process_env_vars(user_config)
        else:
            logger.info(
                "User configuration not found or not provided. Using default configuration!"
            )
            logger.debug({self.default_config_file})
            user_config = {}

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
                "concurrency": {
                    **default_config.get("concurrency", {}),
                    **user_config.get("concurrency", {}),
                },
                "test_filter": {
                    **default_config.get("test_filter", {}),
                    **user_config.get("test_filter", {}),
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
