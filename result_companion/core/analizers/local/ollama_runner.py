import subprocess

from result_companion.core.analizers.local.ollama_exceptions import (
    OllamaModelNotAvailable,
    OllamaNotInstalled,
    OllamaServerNotRunning,
)
from result_companion.core.analizers.local.ollama_server_manager import (
    OllamaServerManager,
)
from result_companion.core.utils.logging_config import logger


def check_ollama_installed(ollama_version_cmd: list = ["ollama", "--version"]) -> None:
    logger.debug("Checking if Ollama is installed...")
    try:
        result = subprocess.run(
            ollama_version_cmd, capture_output=True, text=True, check=True
        )
        logger.debug(f"Ollama installed: {result.stdout.strip()}")
    except FileNotFoundError:
        raise OllamaNotInstalled(
            "Ollama command not found. Ensure it is installed and in your PATH."
        )
    except subprocess.CalledProcessError as exc:
        raise OllamaNotInstalled(f"Ollama command failed: {exc}.")
    except Exception as exc:
        raise OllamaNotInstalled(f"Failed to check Ollama installation: {exc}") from exc
    logger.debug("Ollama installation check passed.")


def check_model_installed(
    model_name: str, ollama_list_cmd: list = ["ollama", "list"]
) -> None:
    logger.debug(f"Checking if model '{model_name}' is installed...")
    try:
        result = subprocess.run(
            ollama_list_cmd, capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as exc:
        raise OllamaServerNotRunning(
            f"'ollama list' command failed with error code {exc.returncode}: {exc.stderr}"
        ) from exc
    if not any(
        line.startswith(f"{model_name}:") or line.startswith(f"{model_name} ")
        for line in result.stdout.splitlines()
    ):
        raise OllamaModelNotAvailable(
            f"Model '{model_name}' is not installed in Ollama. Run `ollama pull {model_name}`."
        )
    logger.debug(f"Model '{model_name}' is installed.")


def ollama_on_init_strategy(
    model_name: str,
    server_url: str = "http://localhost:11434",
    start_timeout: int = 30,
    server_manager_class: OllamaServerManager = OllamaServerManager,
) -> None:
    check_ollama_installed()
    # Create a server manager instance.
    server_manager = server_manager_class(
        server_url=server_url, start_timeout=start_timeout
    )
    if not server_manager.is_running():
        server_manager.start()
    else:
        logger.debug("Ollama server is already running.")
    check_model_installed(model_name)


if __name__ == "__main__":
    import logging

    from langchain_ollama.llms import OllamaLLM

    logging.basicConfig(level=logging.INFO)
    test_model = "deepseek-r1"  # Change to a model you might have/not have
    try:
        ollama_on_init_strategy(test_model)
        print(f"Successfully verified Ollama setup for model: {test_model}")
    except (OllamaNotInstalled, OllamaServerNotRunning, OllamaModelNotAvailable) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Initialize the model with your local server endpoint
    model = OllamaLLM(
        model="deepseek-r1:1.5b",
    )

    result = model.invoke("Come up with consise interesting fact")
    print(result)
