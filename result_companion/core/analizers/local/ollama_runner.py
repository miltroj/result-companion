import subprocess

from result_companion.core.utils.logging_config import logger


def check_ollama_installed(ollama_version: list = ["ollama", "--version"]) -> None:
    try:
        result = subprocess.run(ollama_version, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception("Ollama is not installed.")
        logger.debug(f"Ollama version: {result.stdout.strip()}")
    except FileNotFoundError:
        raise Exception("Ollama is not installed.")


def check_model_installed(
    model_name: str = "llama3.2", ollama_list_cmd: list = ["ollama", "list"]
) -> None:
    try:
        result = subprocess.run(ollama_list_cmd, capture_output=True, text=True)
        if model_name not in result.stdout:
            raise Exception(f"Model {model_name} is not installed.")
        logger.debug(f"Model {model_name} is installed.")
    except Exception as e:
        raise Exception(f"Failed to check if model is installed: {e}")


def ollama_on_init_strategy(model_name: str, *args, **kwargs) -> None:
    check_ollama_installed()
    check_model_installed(model_name=model_name)
