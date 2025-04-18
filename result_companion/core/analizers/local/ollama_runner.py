
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


def is_ollama_server_running(url: str = "http://localhost:11434") -> bool:
    """Checks if the Ollama server is running."""
    try:
        # Use a lightweight endpoint like '/' or '/api/tags'
        response = requests.get(url, timeout=5)
        # Consider any 2xx or 404 (if '/' isn't a valid endpoint but server is up) as running
        # Ollama server returns "Ollama is running" on base url
        return response.status_code == 200 and "Ollama is running" in response.text
    except requests.exceptions.RequestException:
        return False


def check_ollama_installed(ollama_version_cmd: list = ["ollama", "--version"]) -> None:
    """Checks if Ollama is installed by running `ollama --version`."""
    try:
        result = subprocess.run(ollama_version_cmd, capture_output=True, text=True, check=True)
        logger.debug(f"Ollama installed: {result.stdout.strip()}")
    except FileNotFoundError:
        raise OllamaNotInstalled("Ollama command not found. Please ensure Ollama is installed and in your PATH.")
    except subprocess.CalledProcessError as exc:
        raise OllamaNotInstalled(f"Ollama command failed: {exc}. Please ensure Ollama is installed correctly.")
    except Exception as exc:
        raise OllamaNotInstalled(f"Failed to check Ollama installation: {exc}") from exc


def check_model_installed(
    model_name: str, ollama_list_cmd: list = ["ollama", "list"]
) -> None:
    """Checks if the specified model is installed in Ollama."""
    try:
        result = subprocess.run(ollama_list_cmd, capture_output=True, text=True, check=True)
        # Check if the model name appears at the beginning of any line or followed by a tag identifier
        if not any(line.startswith(f"{model_name}:") or line.startswith(f"{model_name} ") for line in result.stdout.splitlines()):
             raise OllamaModelNotAvailable(f"Model '{model_name}' is not installed in Ollama. Please run `ollama pull {model_name}`.")
        logger.debug(f"Model '{model_name}' is installed.")
    except FileNotFoundError:
         raise OllamaNotInstalled("Ollama command not found during model check. Ensure Ollama is installed and in PATH.")
    except subprocess.CalledProcessError as exc:
        # Handle cases where 'ollama list' fails (e.g., server not running, though we check earlier)
        raise OllamaServerNotRunning(f"'ollama list' command failed (is the server running?): {exc.stderr}") from exc
    except OllamaModelNotAvailable:
        raise # Re-raise the specific exception
    except Exception as exc:
        raise Exception(f"Failed to check if model '{model_name}' is installed: {exc}") from exc


def ollama_on_init_strategy(model_name: str, server_url: str = "http://localhost:11434", start_timeout: int = 30) -> None:
    """
    Checks if Ollama is installed, if the server is running (starts it if not),
    and if the specified model is available.
    """
    # 1. Check if Ollama CLI is installed
    logger.debug("Checking if Ollama is installed...")
    check_ollama_installed()
    logger.debug("Ollama installation check passed.")

    # 2. Check if Ollama server is running
    logger.debug(f"Checking if Ollama server is running at {server_url}...")
    if not is_ollama_server_running(server_url):
        logger.info("Ollama server is not running. Attempting to start it...")
        try:
            # Use Popen for non-blocking start, but manage the process
            process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info(f"Launched 'ollama serve' process with PID: {process.pid}")

            # Wait for the server to become responsive
            start_time = time.time()
            server_started = False
            while time.time() - start_time < start_timeout:
                if is_ollama_server_running(server_url):
                    logger.info("Ollama server started successfully.")
                    server_started = True
                    # Give it a tiny bit more time to fully initialize
                    time.sleep(1)
                    break
                time.sleep(1) # Poll every second

            if not server_started:
                # Try to terminate the process if it didn't start successfully
                try:
                    process.terminate()
                    process.wait(timeout=5) # Wait for termination
                except subprocess.TimeoutExpired:
                    process.kill() # Force kill if terminate times out
                except Exception as kill_exc:
                    logger.warning(f"Could not terminate 'ollama serve' process: {kill_exc}")

                # Read stderr for potential errors
                stderr_output = ""
                try:
                     stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
                except Exception:
                     pass # Ignore errors reading stderr if process already dead

                raise OllamaServerNotRunning(
                    f"Failed to start Ollama server within the {start_timeout}s timeout. "
                    f"Error output from 'ollama serve': {stderr_output.strip()}"
                )

        except FileNotFoundError:
             # This case should theoretically be caught by check_ollama_installed, but added for safety
             raise OllamaNotInstalled("Ollama command not found when trying to start the server.")
        except Exception as exc:
            raise OllamaServerNotRunning(f"An unexpected error occurred while trying to start Ollama server: {exc}") from exc
    else:
        logger.debug("Ollama server is already running.")

    # 3. Check if the required model is installed
    logger.debug(f"Checking if model '{model_name}' is installed...")
    check_model_installed(model_name)
    logger.debug(f"Ollama initialization strategy complete. Model '{model_name}' is available.")


# --- Example Usage (Optional, for testing) ---
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    test_model = "deepseek-r11" # Change to a model you might have/not have
    try:
        ollama_on_init_strategy(test_model)
        print(f"Successfully verified Ollama setup for model: {test_model}")
    except (OllamaNotInstalled, OllamaServerNotRunning, OllamaModelNotAvailable) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

