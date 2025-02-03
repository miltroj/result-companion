import subprocess

import requests

from result_companion.core.utils.logging_config import logger


def check_ollama_installed(ollama_version: list = ["ollama", "--version"]) -> None:
    try:
        result = subprocess.run(ollama_version, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception("Ollama is not installed.")
        logger.debug(f"Ollama version: {result.stdout.strip()}")
    except FileNotFoundError:
        raise Exception("Ollama is not installed.")


def check_http_server_running(url="http://localhost:8000") -> bool:
    # TODO: fix this check
    return True
    try:
        response = requests.get(url)
        if response.status_code == 200:
            logger.debug("HTTP server is running.")
        else:
            raise Exception("HTTP server is not responding correctly.")
    except requests.ConnectionError:
        raise Exception("HTTP server is not running.")


def start_ollama_server(start_cmd: list = ["ollama", "start"]) -> None:
    try:
        subprocess.Popen(start_cmd)
        logger.info("Starting Ollama server...")
    except Exception as e:
        raise Exception(f"Failed to start Ollama server: {e}")


def check_model_installed(model_name: str = "llama3.2"):
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if model_name not in result.stdout:
            raise Exception(f"Model {model_name} is not installed.")
        logger.debug(f"Model {model_name} is installed.")
    except Exception as e:
        raise Exception(f"Failed to check if model is installed: {e}")


def ollama_on_init_strategy(model_name: str, *args, **kwargs) -> None:
    check_ollama_installed()
    if not check_http_server_running():
        start_ollama_server()
    check_model_installed(model_name=model_name)
