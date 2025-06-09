import atexit
import os
import subprocess
import time
from typing import Optional, Type, TypeVar, Union

import requests

from result_companion.core.analizers.local.ollama_exceptions import (
    OllamaNotInstalled,
    OllamaServerNotRunning,
)
from result_companion.core.utils.logging_config import logger


class OllamaServerManager:
    """
    Manages the lifecycle of an Ollama server.

    Can be used as a context manager to ensure proper server initialization and cleanup:

    with OllamaServerManager() as server:
        # Code that requires the Ollama server to be running
        ...
    # Server will be automatically cleaned up when exiting the with block
    """

    def __init__(
        self,
        server_url: str = "http://localhost:11434",
        start_timeout: int = 30,
        wait_for_start: int = 1,
        start_cmd: list = ["ollama", "serve"],
    ):
        self.server_url = server_url
        self.start_timeout = start_timeout
        self._process: Optional[subprocess.Popen] = None
        self.wait_for_start = wait_for_start
        self.start_cmd = start_cmd
        atexit.register(self.cleanup)
        self._server_started_by_manager = False

    def __enter__(self):
        """
        Context manager entry point. Ensures the server is running before proceeding.
        """
        if not self.is_running(skip_logs=True):
            self.start()
            self._server_started_by_manager = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point. Cleans up the server if it was started by this manager.
        """
        if self._server_started_by_manager:
            self.cleanup()
        return False  # Propagate any exceptions

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

    def _check_process_alive(self) -> None:
        """
        Check if the managed process is still alive and raise an exception if it died.

        Raises:
            OllamaServerNotRunning: If the process has terminated unexpectedly.
        """
        if self._process is None:
            return

        if self._process.poll() is not None:
            try:
                _, stderr = self._process.communicate(timeout=1)
                error_msg = (
                    stderr.decode().strip()
                    if stderr
                    else "Process terminated unexpectedly"
                )
            except subprocess.TimeoutExpired:
                error_msg = "Process terminated unexpectedly"
            except Exception as e:
                error_msg = (
                    f"Process terminated unexpectedly (error reading output: {e})"
                )

            raise OllamaServerNotRunning(f"Ollama server process died: {error_msg}")

    def start(self) -> None:
        """
        Starts the Ollama server if it is not running.
        Raises:
            OllamaNotInstalled: If the 'ollama' command is not found.
            OllamaServerNotRunning: If the server fails to start within the timeout.
        """
        if self.is_running(skip_logs=True):
            logger.debug("Ollama server is already running.")
            return

        logger.info("Ollama server is not running. Attempting to start it...")
        try:
            # Use process group to handle child processes better
            self._process = subprocess.Popen(
                self.start_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if os.name != "nt" else None,  # Unix only
            )
        except FileNotFoundError:
            raise OllamaNotInstalled(
                "Ollama command not found. Ensure it is installed and in your PATH."
            )

        logger.info(f"Launched 'ollama serve' process with PID: {self._process.pid}")

        # Check if process died immediately after launch
        time.sleep(0.1)  # Brief pause to let process initialize
        self._check_process_alive()

        start_time = time.time()
        while time.time() - start_time < self.start_timeout:
            if self.is_running(skip_logs=True):
                logger.info("Ollama server started successfully.")
                return

            # Check if process died during startup wait
            self._check_process_alive()

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
            self._server_started_by_manager = False


T = TypeVar("T", bound="OllamaServerManager")


def resolve_server_manager(server_manager: Union[Optional[T], Type[T]], **kwargs) -> T:
    """
    Resolve a server manager parameter that can be either a class or an instance.

    Args:
        server_manager: Either an OllamaServerManager instance, a subclass of OllamaServerManager,
                       or None (in which case a default OllamaServerManager will be created).
        **kwargs: Additional arguments to pass to the constructor if a new instance is created.

    Returns:
        An instance of OllamaServerManager or one of its subclasses.
    """
    if isinstance(server_manager, type):
        return server_manager(**kwargs)

    return server_manager or OllamaServerManager(**kwargs)
