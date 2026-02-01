import subprocess
from typing import Optional, Type, Union

from result_companion.core.analizers.local.ollama_exceptions import (
    OllamaModelNotAvailable,
    OllamaNotInstalled,
    OllamaServerNotRunning,
)
from result_companion.core.analizers.local.ollama_server_manager import (
    OllamaServerManager,
    resolve_server_manager,
)
from result_companion.core.utils.logging_config import logger


# TODO: move to different location > ollama_install
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
    server_manager: Union[
        Optional["OllamaServerManager"], Type["OllamaServerManager"]
    ] = OllamaServerManager,
) -> "OllamaServerManager":
    """
    Initialize Ollama by ensuring it is installed, the server is running,
    and the specified model is available.

    Parameters:
        model_name (str): Name of the model to check.
        server_url (str): URL where the server is expected.
        start_timeout (int): Timeout for starting the server.
        server_manager (Union[OllamaServerManager, Type[OllamaServerManager]]):
            Either an instance of OllamaServerManager to use directly,
            or the OllamaServerManager class (or subclass) to instantiate.
            Defaults to the OllamaServerManager class.

    Returns:
        OllamaServerManager: The server manager instance that was used.

    Raises:
        OllamaNotInstalled: If Ollama is not installed.
        OllamaServerNotRunning: If the server fails to start.
        OllamaModelNotInstalled: If the specified model is not installed.
    """
    check_ollama_installed()

    server_manager_instance = resolve_server_manager(
        server_manager, server_url=server_url, start_timeout=start_timeout
    )

    if not server_manager_instance.is_running():
        server_manager_instance.start()
    else:
        logger.debug("Ollama server is already running.")
    logger.debug(f"Ollama server is confirmed running at {server_url}")
    check_model_installed(model_name)
    return server_manager_instance


if __name__ == "__main__":
    # TODO: Transfer to integration tests
    import asyncio
    import logging

    from litellm import acompletion

    logging.basicConfig(level=logging.INFO)
    test_model = "deepseek-r1"  # Change to a model you might have/not have
    try:
        server_mnger = ollama_on_init_strategy(test_model)
        print(f"Successfully verified Ollama setup for model: {test_model}")
    except (OllamaNotInstalled, OllamaServerNotRunning, OllamaModelNotAvailable) as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)

    async def test_ollama():
        response = await acompletion(
            model="ollama_chat/deepseek-r1:1.5b",
            messages=[
                {"role": "user", "content": "Come up with a concise interesting fact"}
            ],
            api_base="http://localhost:11434",
        )
        return response.choices[0].message.content

    result = asyncio.run(test_ollama())
    server_mnger.cleanup()
    print(result)
