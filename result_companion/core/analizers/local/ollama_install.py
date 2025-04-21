import subprocess
from sys import platform

from result_companion.core.analizers.local.ollama_runner import (
    check_model_installed,
    check_ollama_installed,
)
from result_companion.core.utils.logging_config import logger


def auto_install_ollama(
    brew_installation_cmd: list = ["brew", "install", "ollama"],
    linux_update_cmd: list = ["sudo", "apt-get", "update"],
    linux_install_cmd: list = ["sudo", "apt-get", "install", "-y", "ollama"],
    ollama_version: list = ["ollama", "--version"],
) -> bool:
    """
    Check if Ollama is installed; if not, install it automatically.
    Returns None if installation is successful, or raises an Exception.
    """
    try:
        check_ollama_installed(ollama_version=ollama_version)
        logger.info("Ollama is already installed.")
        return True
    except Exception:
        logger.warning("Ollama not found. Attempting to install...")

    if platform == "darwin":
        try:
            logger.info("Ollama not found. Installing via Homebrew...")
            subprocess.run(brew_installation_cmd, check=True)
        except Exception as e:
            raise Exception(
                "Automatic installation via Homebrew failed. Please install Ollama manually."
            ) from e

    elif str(platform).startswith("linux"):
        try:
            logger.info("Ollama not found. Installing via apt-get...")
            subprocess.run(linux_update_cmd, check=True)
            subprocess.run(linux_install_cmd, check=True)
        except Exception as e:
            raise Exception(
                "Automatic installation via apt-get failed. Please install Ollama manually."
            ) from e

    else:
        raise Exception(
            "Automatic installation is not supported on your OS. Please install Ollama manually."
        )

    try:
        check_ollama_installed(ollama_version=ollama_version)
    except Exception as e:
        raise Exception("Ollama installation did not complete successfully.") from e

    logger.info("Ollama installed successfully!")
    return True


def auto_install_model(
    model_name: str,
    installation_cmd: list = ["ollama", "install"],
    ollama_list_cmd: list = ["ollama", "list"],
) -> bool:
    """
    Automatically install the specified model into Ollama.
    Returns True if installation is successful, or raises an Exception.
    """
    try:
        check_model_installed(model_name=model_name, ollama_list_cmd=ollama_list_cmd)
        logger.info(f"Model '{model_name}' is already installed.")
        return True
    except Exception:
        logger.warning(f"Model '{model_name}' not found. Attempting to install...")

    try:
        cmd = installation_cmd + [model_name]
        logger.info(f"Installing model '{model_name}' with command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except Exception as e:
        raise Exception(
            f"Automatic installation of model '{model_name}' failed. Please install it manually."
        ) from e

    try:
        check_model_installed(model_name=model_name, ollama_list_cmd=ollama_list_cmd)
    except Exception as e:
        raise Exception(
            f"Model '{model_name}' installation did not complete successfully."
        ) from e

    logger.info(f"Model '{model_name}' installed successfully!")
    return True
