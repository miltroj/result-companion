import platform
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List

from result_companion.core.utils.logging_config import logger


class PlatformType(Enum):
    MACOS = auto()
    LINUX_DEBIAN = auto()
    LINUX_RHEL = auto()
    LINUX_ARCH = auto()
    WINDOWS = auto()
    UNSUPPORTED = auto()


class OllamaInstallationError(Exception):
    pass


class ModelInstallationError(Exception):
    pass


@dataclass
class InstallConfig:
    """Installation configuration for a platform."""

    commands: List[List[str]]
    prerequisite_check: str  # Command to check if prerequisite exists


class BaseInstaller(ABC):
    """Simplified base installer."""

    def __init__(self):
        self.config = self.get_config()

    @abstractmethod
    def get_config(self) -> InstallConfig:
        """Get installation configuration."""
        pass

    def validate_prerequisites(self) -> bool:
        """Check if prerequisites are available."""
        return shutil.which(self.config.prerequisite_check) is not None

    def install(self) -> None:
        """Install Ollama."""
        if not self.validate_prerequisites():
            raise OllamaInstallationError(
                f"Missing prerequisite: {self.config.prerequisite_check}"
            )

        for cmd in self.config.commands:
            logger.info(f"Executing: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, text=True)


# Platform-specific installers (much simpler)
class MacOSInstaller(BaseInstaller):
    def get_config(self) -> InstallConfig:
        return InstallConfig(
            commands=[["brew", "install", "ollama"]], prerequisite_check="brew"
        )


class DebianInstaller(BaseInstaller):
    def get_config(self) -> InstallConfig:
        return InstallConfig(
            commands=[
                ["curl", "-fsSL", "https://ollama.com/install.sh"],
                ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
            ],
            prerequisite_check="curl",
        )


class RHELInstaller(BaseInstaller):
    def get_config(self) -> InstallConfig:
        return InstallConfig(
            commands=[
                ["curl", "-fsSL", "https://ollama.com/install.sh"],
                ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
            ],
            prerequisite_check="curl",
        )


class ArchInstaller(BaseInstaller):
    def get_config(self) -> InstallConfig:
        return InstallConfig(
            commands=[["sudo", "pacman", "-Sy", "--noconfirm", "ollama"]],
            prerequisite_check="pacman",
        )


class WindowsInstaller(BaseInstaller):
    def get_config(self) -> InstallConfig:
        return InstallConfig(
            commands=[
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
            ],
            prerequisite_check="powershell",
        )


class OllamaManager:
    """Simplified Ollama installation and model manager."""

    # Platform detection and installer mapping
    _INSTALLERS = {
        PlatformType.MACOS: MacOSInstaller,
        PlatformType.LINUX_DEBIAN: DebianInstaller,
        PlatformType.LINUX_RHEL: RHELInstaller,
        PlatformType.LINUX_ARCH: ArchInstaller,
        PlatformType.WINDOWS: WindowsInstaller,
    }

    def __init__(self):
        self.platform = self._detect_platform()

    def _detect_platform(self) -> PlatformType:
        """Detect current platform."""
        system = platform.system().lower()

        if system == "darwin":
            return PlatformType.MACOS
        elif system == "windows":
            return PlatformType.WINDOWS
        elif system == "linux":
            return self._detect_linux_distro()
        else:
            return PlatformType.UNSUPPORTED

    def _detect_linux_distro(self) -> PlatformType:
        """Detect Linux distribution."""
        try:
            with open("/etc/os-release", "r") as f:
                content = f.read().lower()
                if any(x in content for x in ["ubuntu", "debian"]):
                    return PlatformType.LINUX_DEBIAN
                elif any(x in content for x in ["rhel", "centos", "fedora"]):
                    return PlatformType.LINUX_RHEL
                elif "arch" in content:
                    return PlatformType.LINUX_ARCH
        except FileNotFoundError:
            pass

        # Fallback to package manager detection
        if shutil.which("apt"):
            return PlatformType.LINUX_DEBIAN
        if shutil.which("yum") or shutil.which("dnf"):
            return PlatformType.LINUX_RHEL
        if shutil.which("pacman"):
            return PlatformType.LINUX_ARCH

        return PlatformType.LINUX_DEBIAN  # Default

    def _run_command(
        self, cmd: List[str], stream_output: bool = False
    ) -> subprocess.CompletedProcess:
        """Run command with optional streaming."""
        if stream_output:
            return self._run_with_streaming(cmd)
        return subprocess.run(cmd, capture_output=True, text=True, check=True)

    def _run_with_streaming(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run command with real-time output streaming."""
        logger.info(f"Running: {' '.join(cmd)}")

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
            cmd, return_code, "\n".join(output_lines), ""
        )

        if return_code != 0:
            raise subprocess.CalledProcessError(
                return_code, cmd, "\n".join(output_lines)
            )

        return result

    def is_ollama_installed(self) -> bool:
        """Check if Ollama is installed."""
        try:
            self._run_command(["ollama", "--version"])
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def is_model_installed(self, model_name: str) -> bool:
        """Check if model is installed."""
        try:
            result = self._run_command(["ollama", "list"])
            return model_name in result.stdout
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def install_ollama(self) -> bool:
        """Install Ollama if not already installed."""
        if self.is_ollama_installed():
            logger.info("Ollama already installed.")
            return True

        if self.platform == PlatformType.UNSUPPORTED:
            raise OllamaInstallationError(f"Unsupported platform: {platform.system()}")

        logger.info("Installing Ollama...")

        try:
            installer_class = self._INSTALLERS[self.platform]
            installer = installer_class()
            installer.install()
        except Exception as e:
            raise OllamaInstallationError(f"Installation failed: {str(e)}") from e

        if not self.is_ollama_installed():
            raise OllamaInstallationError("Installation verification failed")

        logger.info("Ollama installed successfully!")
        return True

    def install_model(self, model_name: str) -> bool:
        """Install model if not already installed."""
        if self.is_model_installed(model_name):
            logger.info(f"Model '{model_name}' already installed.")
            return True

        logger.info(f"Installing model '{model_name}'...")

        try:
            self._run_command(["ollama", "pull", model_name], stream_output=True)
        except Exception as e:
            raise ModelInstallationError(f"Model installation failed: {str(e)}") from e

        if not self.is_model_installed(model_name):
            raise ModelInstallationError("Model installation verification failed")

        logger.info(f"Model '{model_name}' installed successfully!")
        return True


# Simple convenience functions
def auto_install_ollama() -> bool:
    """Auto-install Ollama."""
    return OllamaManager().install_ollama()


def auto_install_model(model_name: str) -> bool:
    """Auto-install model."""
    return OllamaManager().install_model(model_name)
