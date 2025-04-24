import subprocess
from dataclasses import dataclass
from enum import Enum, auto
from sys import platform
from typing import List, Optional, Protocol

from result_companion.core.utils.logging_config import logger


class SubprocessRunner(Protocol):
    """Protocol defining how to run subprocess commands."""

    def run(self, cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a subprocess command."""
        ...


class DefaultSubprocessRunner:
    """Default implementation of SubprocessRunner using subprocess module."""

    def run(self, cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a subprocess command and return the result."""
        return subprocess.run(cmd, check=check, capture_output=True, text=True)


class PlatformType(Enum):
    """Enum representing supported operating system platforms."""

    MACOS = auto()
    LINUX = auto()
    UNSUPPORTED = auto()


@dataclass
class OllamaCommands:
    """Configuration for Ollama commands across platforms."""

    version_cmd: List[str] = None
    list_cmd: List[str] = None
    install_model_cmd: List[str] = None

    def __post_init__(self):
        """Set default values if not provided."""
        self.version_cmd = self.version_cmd or ["ollama", "--version"]
        self.list_cmd = self.list_cmd or ["ollama", "list"]
        self.install_model_cmd = self.install_model_cmd or ["ollama", "install"]


@dataclass
class PlatformCommands:
    """Platform-specific installation commands."""

    mac_install_cmd: List[str] = None
    linux_update_cmd: List[str] = None
    linux_install_cmd: List[str] = None

    def __post_init__(self):
        """Set default values if not provided."""
        self.mac_install_cmd = self.mac_install_cmd or ["brew", "install", "ollama"]
        self.linux_update_cmd = self.linux_update_cmd or ["sudo", "apt-get", "update"]
        self.linux_install_cmd = self.linux_install_cmd or [
            "sudo",
            "apt-get",
            "install",
            "-y",
            "ollama",
        ]


class OllamaInstallationError(Exception):
    """Exception raised for errors during Ollama installation."""

    pass


class ModelInstallationError(Exception):
    """Exception raised for errors during model installation."""

    pass


class OllamaInstallationManager:
    """
    Manages Ollama installation and model management across different platforms.

    This class provides functionality to:
    1. Detect the current platform
    2. Check if Ollama is installed
    3. Install Ollama if needed
    4. Check if specific models are installed
    5. Install models if needed
    """

    def __init__(
        self,
        subprocess_runner: Optional[SubprocessRunner] = None,
        ollama_commands: Optional[OllamaCommands] = None,
        platform_commands: Optional[PlatformCommands] = None,
    ):
        """
        Initialize OllamaInstallationManager with configurable dependencies.

        Args:
            subprocess_runner: Component to run subprocess commands
            ollama_commands: Configuration for Ollama commands
            platform_commands: Platform-specific installation commands
        """
        self.subprocess_runner = subprocess_runner or DefaultSubprocessRunner()
        self.ollama_commands = ollama_commands or OllamaCommands()
        self.platform_commands = platform_commands or PlatformCommands()
        self.platform_type = self._detect_platform()

    def _detect_platform(self) -> PlatformType:
        """
        Detect the current operating system platform.

        Returns:
            PlatformType enum representing the current platform
        """
        if platform == "darwin":
            return PlatformType.MACOS
        elif str(platform).startswith("linux"):
            return PlatformType.LINUX
        else:
            return PlatformType.UNSUPPORTED

    def is_ollama_installed(self) -> bool:
        """
        Check if Ollama is installed on the system.

        Returns:
            True if Ollama is installed, False otherwise
        """
        try:
            self.subprocess_runner.run(self.ollama_commands.version_cmd)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def is_model_installed(self, model_name: str) -> bool:
        """
        Check if a specific model is installed in Ollama.

        Args:
            model_name: Name of the model to check

        Returns:
            True if the model is installed, False otherwise
        """
        try:
            result = self.subprocess_runner.run(self.ollama_commands.list_cmd)
            return model_name in result.stdout
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def install_ollama(self) -> bool:
        """
        Install Ollama if not already installed.

        Returns:
            True if installation was successful

        Raises:
            OllamaInstallationError: If installation fails
        """
        if self.is_ollama_installed():
            logger.info("Ollama is already installed.")
            return True

        logger.info("Ollama not found. Attempting to install...")

        if self.platform_type == PlatformType.MACOS:
            self._install_ollama_mac()
        elif self.platform_type == PlatformType.LINUX:
            self._install_ollama_linux()
        else:
            raise OllamaInstallationError(
                f"Automatic installation is not supported on your OS ({self.platform_type}). "
                "Please install Ollama manually."
            )

        if not self.is_ollama_installed():
            raise OllamaInstallationError(
                f"Ollama installation did not complete successfully for {self.platform_type}."
            )

        logger.info("Ollama installed successfully!")
        return True

    def _install_ollama_mac(self) -> None:
        """
        Install Ollama on macOS using Homebrew.

        Raises:
            OllamaInstallationError: If installation fails
        """
        try:
            logger.info("Installing Ollama via Homebrew...")
            self.subprocess_runner.run(self.platform_commands.mac_install_cmd)
        except Exception as e:
            raise OllamaInstallationError(
                "Automatic installation via Homebrew failed. Please install Ollama manually."
            ) from e

    def _install_ollama_linux(self) -> None:
        """
        Install Ollama on Linux using apt-get.

        Raises:
            OllamaInstallationError: If installation fails
        """
        try:
            logger.info("Installing Ollama via apt-get...")
            self.subprocess_runner.run(self.platform_commands.linux_update_cmd)
            self.subprocess_runner.run(self.platform_commands.linux_install_cmd)
        except Exception as e:
            raise OllamaInstallationError(
                "Automatic installation via apt-get failed. Please install Ollama manually."
            ) from e

    def install_model(self, model_name: str) -> bool:
        """
        Install a model to Ollama if not already installed.

        Args:
            model_name: Name of the model to install

        Returns:
            True if the model is installed successfully

        Raises:
            ModelInstallationError: If model installation fails
        """
        if self.is_model_installed(model_name):
            logger.info(f"Model '{model_name}' is already installed.")
            return True

        logger.info(f"Model '{model_name}' not found. Attempting to install...")

        try:
            cmd = self.ollama_commands.install_model_cmd + [model_name]
            logger.info(
                f"Installing model '{model_name}' with command: {' '.join(cmd)}"
            )
            self.subprocess_runner.run(cmd)
        except Exception as e:
            raise ModelInstallationError(
                f"Automatic installation of model '{model_name}' failed. Please install it manually."
            ) from e

        if not self.is_model_installed(model_name):
            raise ModelInstallationError(
                f"Model '{model_name}' installation did not complete successfully."
            )

        logger.info(f"Model '{model_name}' installed successfully!")
        return True


def auto_install_ollama(
    brew_installation_cmd: List[str] = ["brew", "install", "ollama"],
    linux_update_cmd: List[str] = ["sudo", "apt-get", "update"],
    linux_install_cmd: List[str] = ["sudo", "apt-get", "install", "-y", "ollama"],
    ollama_version: List[str] = ["ollama", "--version"],
) -> bool:
    """
    Check if Ollama is installed; if not, install it automatically.

    Args:
        brew_installation_cmd: Command to install Ollama via Homebrew
        linux_update_cmd: Command to update apt repositories
        linux_install_cmd: Command to install Ollama via apt-get
        ollama_version: Command to check Ollama version

    Returns:
        True if installation is successful

    Raises:
        Exception: If installation fails
    """
    platform_commands = PlatformCommands(
        mac_install_cmd=brew_installation_cmd,
        linux_update_cmd=linux_update_cmd,
        linux_install_cmd=linux_install_cmd,
    )
    ollama_commands = OllamaCommands(version_cmd=ollama_version)

    manager = OllamaInstallationManager(
        platform_commands=platform_commands,
        ollama_commands=ollama_commands,
    )

    try:
        return manager.install_ollama()
    except OllamaInstallationError as e:
        raise Exception(str(e)) from e


def auto_install_model(
    model_name: str,
    installation_cmd: List[str] = ["ollama", "install"],
    ollama_list_cmd: List[str] = ["ollama", "list"],
) -> bool:
    """
    Automatically install the specified model into Ollama.

    Args:
        model_name: Name of the model to install
        installation_cmd: Command to install models
        ollama_list_cmd: Command to list installed models

    Returns:
        True if installation is successful

    Raises:
        Exception: If installation fails
    """
    ollama_commands = OllamaCommands(
        list_cmd=ollama_list_cmd,
        install_model_cmd=installation_cmd,
    )

    manager = OllamaInstallationManager(ollama_commands=ollama_commands)

    try:
        return manager.install_model(model_name)
    except ModelInstallationError as e:
        raise Exception(str(e)) from e
