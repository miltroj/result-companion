import atexit
import subprocess
import time

import requests

from result_companion.core.utils.logging_config import logger


class OllamaNotInstalled(Exception):
    """Exception raised when Ollama is not installed."""


class OllamaServerNotRunning(Exception):
    """Exception raised when Ollama server is not running."""


class OllamaModelNotAvailable(Exception):
    """Exception raised when the required Ollama model is not available."""


# Global variable to store the Ollama server process
_ollama_server_process = None


def cleanup_ollama_server() -> None:
    global _ollama_server_process
    if _ollama_server_process is not None:
        logger.debug(
            f"Cleaning up Ollama server process with PID: {_ollama_server_process.pid}"
        )
        try:
            _ollama_server_process.terminate()
            _ollama_server_process.wait(timeout=5)
            logger.debug("Ollama server terminated gracefully.")
        except subprocess.TimeoutExpired:
            _ollama_server_process.kill()
            logger.debug("Ollama server killed forcefully.")
        except Exception as exc:
            logger.warning(f"Error during Ollama server cleanup: {exc}")
        _ollama_server_process = None


# Register the cleanup function to be called on program exit
atexit.register(cleanup_ollama_server)


def is_ollama_server_running(url: str = "http://localhost:11434") -> bool:
    """Checks if the Ollama server is running."""
    logger.debug(f"Checking if Ollama server is running at {url}...")
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200 and "Ollama is running" in response.text
    except requests.exceptions.RequestException:
        return False


def check_ollama_installed(ollama_version_cmd: list = ["ollama", "--version"]) -> None:
    logger.debug("Checking if Ollama is installed...")
    try:
        result = subprocess.run(
            ollama_version_cmd, capture_output=True, text=True, check=True
        )
        logger.debug(f"Ollama installed: {result.stdout.strip()}")
    except FileNotFoundError:
        raise OllamaNotInstalled(
            "Ollama command not found. Please ensure Ollama is installed and in your PATH."
        )
    except subprocess.CalledProcessError as exc:
        raise OllamaNotInstalled(
            f"Ollama command failed: {exc}. Please ensure Ollama is installed correctly."
        )
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
        if not any(
            line.startswith(f"{model_name}:") or line.startswith(f"{model_name} ")
            for line in result.stdout.splitlines()
        ):
            raise OllamaModelNotAvailable(
                f"Model '{model_name}' is not installed in Ollama. Please run `ollama pull {model_name}`."
            )
        logger.debug(f"Model '{model_name}' is installed.")
    except FileNotFoundError:
        raise OllamaNotInstalled(
            "Ollama command not found during model check. Ensure Ollama is installed and in PATH."
        )
    except subprocess.CalledProcessError as exc:
        raise OllamaServerNotRunning(
            f"'ollama list' command failed (is the server running?): {exc.stderr}"
        ) from exc
    except OllamaModelNotAvailable:
        raise
    except Exception as exc:
        raise Exception(
            f"Failed to check if model '{model_name}' is installed: {exc}"
        ) from exc
    logger.debug(
        f"Ollama initialization strategy complete. Model '{model_name}' is available."
    )


def start_ollama_server(server_url: str, start_timeout: int) -> None:
    """
    Starts the Ollama server, waits for it to become responsive,
    and handles potential errors during startup.
    """
    global _ollama_server_process
    logger.info("Ollama server is not running. Attempting to start it...")
    try:
        _ollama_server_process = subprocess.Popen(
            ["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logger.info(
            f"Launched 'ollama serve' process with PID: {_ollama_server_process.pid}"
        )

        start_time = time.time()
        server_started = False
        while time.time() - start_time < start_timeout:
            if is_ollama_server_running(server_url):
                logger.info("Ollama server started successfully.")
                server_started = True
                time.sleep(1)
                break
            time.sleep(1)

        if not server_started:
            try:
                _ollama_server_process.terminate()
                _ollama_server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _ollama_server_process.kill()
            except Exception as kill_exc:
                logger.warning(
                    f"Could not terminate 'ollama serve' process: {kill_exc}"
                )

            stderr_output = ""
            try:
                stderr_output = _ollama_server_process.stderr.read().decode(
                    "utf-8", errors="ignore"
                )
            except Exception:
                pass

            raise OllamaServerNotRunning(
                f"Failed to start Ollama server within the {start_timeout}s timeout. Error output from 'ollama serve': {stderr_output.strip()}"
            )
    except FileNotFoundError:
        raise OllamaNotInstalled(
            "Ollama command not found when trying to start the server."
        )
    except Exception as exc:
        raise OllamaServerNotRunning(
            f"An unexpected error occurred while trying to start Ollama server: {exc}"
        ) from exc


def ollama_on_init_strategy(
    model_name: str, server_url: str = "http://localhost:11434", start_timeout: int = 30
) -> None:
    check_ollama_installed()

    if not is_ollama_server_running(server_url):
        start_ollama_server(server_url, start_timeout)
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
