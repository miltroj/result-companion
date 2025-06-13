import platform
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Protocol

from result_companion.core.utils.logging_config import logger


class SubprocessRunner(Protocol):
    """Protocol defining how to run subprocess commands."""

    def run(self, cmd: list) -> subprocess.CompletedProcess:
        """Run command and return completed process."""
        raise NotImplementedError

    def run_with_streaming(self, cmd: list) -> subprocess.CompletedProcess:
        """Run command with real-time output streaming."""
        raise NotImplementedError


class DefaultSubprocessRunner:
    """Default implementation of SubprocessRunner using subprocess module."""

    def run(self, cmd: list) -> subprocess.CompletedProcess:
        """Run command and return completed process."""
        return subprocess.run(cmd, capture_output=True, text=True, check=True)

    def run_with_streaming(self, cmd: list) -> subprocess.CompletedProcess:
        """Run command with real-time output streaming."""
        logger.info(f"Running command with streaming: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        output_lines = []

        for line in iter(process.stdout.readline, ""):
            line = line.rstrip()
            if line:
                logger.info(f"Ollama: {line}")
                output_lines.append(line)

        return_code = process.wait()

        result = subprocess.CompletedProcess(
            args=cmd, returncode=return_code, stdout="\n".join(output_lines), stderr=""
        )

        if return_code != 0:
            raise subprocess.CalledProcessError(
                return_code, cmd, "\n".join(output_lines)
            )

        return result


class PlatformType(Enum):
    """Enum representing supported operating system platforms."""

    MACOS = auto()
    LINUX_DEBIAN = auto()
    LINUX_RHEL = auto()
    LINUX_ARCH = auto()
    WINDOWS = auto()
    UNSUPPORTED = auto()


class OllamaInstallationError(Exception):
    """Exception raised for errors during Ollama installation."""

    pass


class ModelInstallationError(Exception):
    """Exception raised for errors during model installation."""

    pass


class OllamaInstaller(ABC):
    """Abstract base class for platform-specific Ollama installers."""

    def __init__(self, subprocess_runner: SubprocessRunner):
        self.subprocess_runner = subprocess_runner

    @abstractmethod
    def get_install_commands(self) -> List[List[str]]:
        """Get installation commands for this platform."""
        pass

    @abstractmethod
    def validate_prerequisites(self) -> bool:
        """Validate that prerequisites are installed."""
        pass

    @abstractmethod
    def get_platform_type(self) -> PlatformType:
        """Get the platform type this installer handles."""
        pass

    def install(self) -> bool:
        """Install Ollama on this platform."""
        if not self.validate_prerequisites():
            raise OllamaInstallationError(
                f"Prerequisites not met for {self.get_platform_type().name}"
            )

        commands = self.get_install_commands()

        for cmd in commands:
            try:
                logger.info(f"Executing: {' '.join(cmd)}")
                self.subprocess_runner.run(cmd)
            except Exception as e:
                raise OllamaInstallationError(
                    f"Installation failed at command {' '.join(cmd)}: {str(e)}"
                ) from e

        return True


class MacOSInstaller(OllamaInstaller):
    """macOS-specific Ollama installer using Homebrew."""

    def get_install_commands(self) -> List[List[str]]:
        return [["brew", "install", "ollama"]]

    def validate_prerequisites(self) -> bool:
        """Check if Homebrew is installed."""
        return shutil.which("brew") is not None

    def get_platform_type(self) -> PlatformType:
        return PlatformType.MACOS


class DebianLinuxInstaller(OllamaInstaller):
    """Debian/Ubuntu Linux installer using official script."""

    def get_install_commands(self) -> List[List[str]]:
        return [
            [
                "curl",
                "-fsSL",
                "https://ollama.com/install.sh",
                "-o",
                "/tmp/ollama_install.sh",
            ],
            ["bash", "/tmp/ollama_install.sh"],
        ]

    def validate_prerequisites(self) -> bool:
        """Check if curl is installed."""
        return shutil.which("curl") is not None

    def get_platform_type(self) -> PlatformType:
        return PlatformType.LINUX_DEBIAN


class RHELLinuxInstaller(OllamaInstaller):
    """RHEL/CentOS/Fedora Linux installer using official script."""

    def get_install_commands(self) -> List[List[str]]:
        return [
            [
                "curl",
                "-fsSL",
                "https://ollama.com/install.sh",
                "-o",
                "/tmp/ollama_install.sh",
            ],
            ["bash", "/tmp/ollama_install.sh"],
        ]

    def validate_prerequisites(self) -> bool:
        """Check if curl is installed."""
        return shutil.which("curl") is not None

    def get_platform_type(self) -> PlatformType:
        return PlatformType.LINUX_RHEL


class ArchLinuxInstaller(OllamaInstaller):
    """Arch Linux installer using pacman."""

    def get_install_commands(self) -> List[List[str]]:
        return [["sudo", "pacman", "-Sy", "--noconfirm", "ollama"]]

    def validate_prerequisites(self) -> bool:
        """Check if pacman is installed."""
        return shutil.which("pacman") is not None

    def get_platform_type(self) -> PlatformType:
        return PlatformType.LINUX_ARCH


class WindowsInstaller(OllamaInstaller):
    """Windows installer using PowerShell."""

    def get_install_commands(self) -> List[List[str]]:
        return [
            [
                "powershell",
                "-Command",
                "Invoke-WebRequest -Uri https://ollama.com/download/windows -OutFile $env:TEMP\\ollama-installer.exe",
            ],
            [
                "powershell",
                "-Command",
                "Start-Process -FilePath $env:TEMP\\ollama-installer.exe -ArgumentList '/S' -Wait",
            ],
        ]

    def validate_prerequisites(self) -> bool:
        """Check if PowerShell is available."""
        return shutil.which("powershell") is not None

    def get_platform_type(self) -> PlatformType:
        return PlatformType.WINDOWS


class PlatformDetector:
    """Detects the current platform and Linux distribution."""

    @staticmethod
    def detect_platform() -> PlatformType:
        """Detect the current platform."""
        system = platform.system().lower()

        if system == "darwin":
            return PlatformType.MACOS
        elif system == "linux":
            return PlatformDetector._detect_linux_distro()
        elif system == "windows":
            return PlatformType.WINDOWS
        else:
            return PlatformType.UNSUPPORTED

    @staticmethod
    def _detect_linux_distro() -> PlatformType:
        """Detect Linux distribution."""
        try:
            with open("/etc/os-release", "r") as f:
                content = f.read().lower()
                if "ubuntu" in content or "debian" in content:
                    return PlatformType.LINUX_DEBIAN
                elif any(
                    distro in content
                    for distro in ["rhel", "centos", "fedora", "rocky", "alma"]
                ):
                    return PlatformType.LINUX_RHEL
                elif "arch" in content or "manjaro" in content:
                    return PlatformType.LINUX_ARCH
        except FileNotFoundError:
            pass

        # Check for package managers as fallback
        if shutil.which("apt"):
            return PlatformType.LINUX_DEBIAN
        elif shutil.which("yum") or shutil.which("dnf"):
            return PlatformType.LINUX_RHEL
        elif shutil.which("pacman"):
            return PlatformType.LINUX_ARCH

        # Default to Debian-based
        return PlatformType.LINUX_DEBIAN


class OllamaInstallerFactory:
    """Factory for creating platform-specific Ollama installers."""

    _installers: Dict[PlatformType, type] = {
        PlatformType.MACOS: MacOSInstaller,
        PlatformType.LINUX_DEBIAN: DebianLinuxInstaller,
        PlatformType.LINUX_RHEL: RHELLinuxInstaller,
        PlatformType.LINUX_ARCH: ArchLinuxInstaller,
        PlatformType.WINDOWS: WindowsInstaller,
    }

    @classmethod
    def create_installer(
        self,
        platform_type: Optional[PlatformType] = None,
        subprocess_runner: Optional[SubprocessRunner] = None,
    ) -> OllamaInstaller:
        """Create an installer for the specified or current platform."""
        if platform_type is None:
            platform_type = PlatformDetector.detect_platform()

        if subprocess_runner is None:
            subprocess_runner = DefaultSubprocessRunner()

        installer_class = self._installers.get(platform_type)
        if installer_class is None:
            raise OllamaInstallationError(
                f"No installer available for platform: {platform_type}"
            )

        return installer_class(subprocess_runner)


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
        self.install_model_cmd = self.install_model_cmd or ["ollama", "pull"]


class OllamaInstallationManager:
    """
    Manages Ollama installation and model management across different platforms.
    """

    def __init__(
        self,
        subprocess_runner: Optional[SubprocessRunner] = None,
        ollama_commands: Optional[OllamaCommands] = None,
        installer_factory: Optional[OllamaInstallerFactory] = None,
    ):
        """Initialize OllamaInstallationManager with configurable dependencies."""
        self.subprocess_runner = subprocess_runner or DefaultSubprocessRunner()
        self.ollama_commands = ollama_commands or OllamaCommands()
        self.installer_factory = installer_factory or OllamaInstallerFactory()
        self.platform_type = PlatformDetector.detect_platform()

    def is_ollama_installed(self) -> bool:
        """Check if Ollama is installed on the system."""
        try:
            self.subprocess_runner.run(self.ollama_commands.version_cmd)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def is_model_installed(self, model_name: str) -> bool:
        """Check if a specific model is installed in Ollama."""
        try:
            result = self.subprocess_runner.run(self.ollama_commands.list_cmd)
            return model_name in result.stdout
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def install_ollama(self) -> bool:
        """Install Ollama if not already installed."""
        if self.is_ollama_installed():
            logger.info("Ollama is already installed.")
            return True

        logger.info("Ollama not found. Attempting to install...")

        try:
            installer = self.installer_factory.create_installer(
                self.platform_type, self.subprocess_runner
            )
            installer.install()
        except Exception as e:
            raise OllamaInstallationError(
                f"Failed to install Ollama on {self.platform_type.name}: {str(e)}"
            ) from e

        if not self.is_ollama_installed():
            raise OllamaInstallationError(
                f"Ollama installation verification failed for {self.platform_type.name}."
            )

        logger.info("Ollama installed successfully!")
        return True

    def install_model(self, model_name: str) -> bool:
        """Install a model to Ollama if not already installed."""
        if self.is_model_installed(model_name):
            logger.info(f"Model '{model_name}' is already installed.")
            return True

        logger.info(f"Model '{model_name}' not found. Attempting to install...")

        try:
            cmd = self.ollama_commands.install_model_cmd + [model_name]
            logger.info(
                f"Installing model '{model_name}' with command: {' '.join(cmd)}"
            )
            self.subprocess_runner.run_with_streaming(cmd)
        except Exception as e:
            cli_error = e.stderr if hasattr(e, "stderr") else str(e)
            raise ModelInstallationError(
                f"Automatic installation of model '{model_name}' failed with error \n{cli_error}. "
                "Please install it manually."
            ) from e

        if not self.is_model_installed(model_name):
            raise ModelInstallationError(
                f"Model '{model_name}' installation verification failed."
            )

        logger.info(f"Model '{model_name}' installed successfully!")
        return True


def auto_install_ollama() -> bool:
    """Check if Ollama is installed; if not, install it automatically."""
    manager = OllamaInstallationManager()
    try:
        return manager.install_ollama()
    except OllamaInstallationError as e:
        raise Exception(str(e)) from e


def auto_install_model(model_name: str) -> bool:
    """Automatically install the specified model into Ollama."""
    manager = OllamaInstallationManager()
    return manager.install_model(model_name)
