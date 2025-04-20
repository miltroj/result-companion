import atexit
import subprocess
import time

import requests

from result_companion.core.analizers.local.ollama_exceptions import (
    OllamaNotInstalled,
    OllamaServerNotRunning,
)
from result_companion.core.utils.logging_config import logger


class OllamaServerManager:
    """
    Manages the lifecycle of an Ollama server.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:11434",
        start_timeout: int = 30,
        wait_for_start: int = 1,
    ):
        self.server_url = server_url
        self.start_timeout = start_timeout
        self._process = None
        self.wait_for_start = wait_for_start
        atexit.register(self.cleanup)

    def is_running(self, skip_logs: bool = False) -> bool:
        """Checks if the Ollama server is running."""
        if not skip_logs:
            logger.debug(
                f"Checking if Ollama server is running at {self.server_url}..."
            )
        try:
            response = requests.get(self.server_url, timeout=5)
            return response.status_code == 200 and "Ollama is running" in response.text
        except requests.exceptions.RequestException:
            return False

    def start(self) -> None:
        """
        Starts the Ollama server if it is not running.
        Raises:
            OllamaNotInstalled: If the 'ollama' command is not found.
            Exception: If the server fails to start within the timeout.
        """
        if self.is_running(skip_logs=True):
            logger.debug("Ollama server is already running.")
            return

        logger.info("Ollama server is not running. Attempting to start it...")
        try:
            self._process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            raise OllamaNotInstalled(
                "Ollama command not found. Ensure it is installed and in your PATH."
            )

        logger.info(f"Launched 'ollama serve' process with PID: {self._process.pid}")
        start_time = time.time()
        while time.time() - start_time < self.start_timeout:
            if self.is_running(skip_logs=True):
                logger.info("Ollama server started successfully.")
                return
            time.sleep(self.wait_for_start)

        # If the server did not start, clean up and raise an error.
        self.cleanup()
        raise OllamaServerNotRunning(
            f"Failed to start Ollama server within {self.start_timeout}s timeout."
        )

    def cleanup(self) -> None:
        """
        Gracefully terminates the Ollama server, or kills it if necessary.
        """
        if self._process is not None:
            logger.debug(
                f"Cleaning up Ollama server process with PID: {self._process.pid}"
            )
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
                logger.debug("Ollama server terminated gracefully.")
            except subprocess.TimeoutExpired:
                self._process.kill()
                logger.debug("Ollama server killed forcefully.")
            except Exception as exc:
                logger.warning(f"Error during Ollama server cleanup: {exc}")
            self._process = None
