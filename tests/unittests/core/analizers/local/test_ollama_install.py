import subprocess
from unittest.mock import MagicMock, mock_open, patch

import pytest

from result_companion.core.analizers.local.ollama_install import (
    ArchInstaller,
    BaseInstaller,
    DebianInstaller,
    InstallConfig,
    MacOSInstaller,
    ModelInstallationError,
    OllamaInstallationError,
    OllamaManager,
    PlatformType,
    RHELInstaller,
    WindowsInstaller,
    auto_install_model,
    auto_install_ollama,
)


class TestPlatformType:
    """Test PlatformType enum."""

    def test_platform_type_values(self):
        """Test that all platform types are defined."""
        assert PlatformType.MACOS
        assert PlatformType.LINUX_DEBIAN
        assert PlatformType.LINUX_RHEL
        assert PlatformType.LINUX_ARCH
        assert PlatformType.WINDOWS
        assert PlatformType.UNSUPPORTED


class TestExceptions:
    """Test custom exceptions."""

    def test_ollama_installation_error(self):
        """Test OllamaInstallationError can be raised."""
        with pytest.raises(OllamaInstallationError) as exc_info:
            raise OllamaInstallationError("Test error")
        assert str(exc_info.value) == "Test error"

    def test_model_installation_error(self):
        """Test ModelInstallationError can be raised."""
        with pytest.raises(ModelInstallationError) as exc_info:
            raise ModelInstallationError("Test error")
        assert str(exc_info.value) == "Test error"


class TestInstallConfig:
    """Test InstallConfig dataclass."""

    def test_install_config_creation(self):
        """Test creating InstallConfig instance."""
        commands = [["brew", "install", "ollama"]]
        config = InstallConfig(commands=commands, prerequisite_check="brew")

        assert config.commands == commands
        assert config.prerequisite_check == "brew"


class TestBaseInstaller:
    """Test BaseInstaller abstract class."""

    class ConcreteInstaller(BaseInstaller):
        """Concrete implementation for testing."""

        def get_config(self) -> InstallConfig:
            return InstallConfig(
                commands=[["test", "command"]], prerequisite_check="test"
            )

    def test_base_installer_initialization(self):
        """Test BaseInstaller initialization."""
        installer = self.ConcreteInstaller()
        assert installer.config.prerequisite_check == "test"
        assert installer.config.commands == [["test", "command"]]

    @patch("shutil.which")
    def test_validate_prerequisites_success(self, mock_which):
        """Test prerequisite validation when command exists."""
        mock_which.return_value = "/usr/bin/test"
        installer = self.ConcreteInstaller()

        assert installer.validate_prerequisites() is True
        mock_which.assert_called_once_with("test")

    @patch("shutil.which")
    def test_validate_prerequisites_failure(self, mock_which):
        """Test prerequisite validation when command doesn't exist."""
        mock_which.return_value = None
        installer = self.ConcreteInstaller()

        assert installer.validate_prerequisites() is False
        mock_which.assert_called_once_with("test")

    @patch("subprocess.run")
    @patch("shutil.which")
    def test_install_success(self, mock_which, mock_run):
        """Test successful installation."""
        mock_which.return_value = "/usr/bin/test"
        mock_run.return_value = MagicMock(returncode=0)

        installer = self.ConcreteInstaller()

        installer.install()

        mock_run.assert_called_once_with(
            ["test", "command"], check=True, capture_output=True, text=True
        )

    @patch("shutil.which")
    def test_install_missing_prerequisite(self, mock_which):
        """Test installation fails when prerequisite is missing."""
        mock_which.return_value = None
        installer = self.ConcreteInstaller()

        with pytest.raises(OllamaInstallationError) as exc_info:
            installer.install()

        assert "Missing prerequisite: test" in str(exc_info.value)

    @patch("subprocess.run")
    @patch("shutil.which")
    def test_install_command_failure(self, mock_which, mock_run):
        """Test installation fails when command fails."""
        mock_which.return_value = "/usr/bin/test"
        mock_run.side_effect = subprocess.CalledProcessError(1, ["test", "command"])

        installer = self.ConcreteInstaller()

        with pytest.raises(subprocess.CalledProcessError):
            installer.install()


class TestPlatformInstallers:
    """Test platform-specific installers."""

    def test_macos_installer_config(self):
        """Test MacOS installer configuration."""
        installer = MacOSInstaller()
        config = installer.get_config()

        assert config.commands == [["brew", "install", "ollama"]]
        assert config.prerequisite_check == "brew"

    def test_debian_installer_config(self):
        """Test Debian installer configuration."""
        installer = DebianInstaller()
        config = installer.get_config()

        assert len(config.commands) == 2
        assert config.commands[0] == ["curl", "-fsSL", "https://ollama.com/install.sh"]
        assert config.prerequisite_check == "curl"

    def test_rhel_installer_config(self):
        """Test RHEL installer configuration."""
        installer = RHELInstaller()
        config = installer.get_config()

        assert len(config.commands) == 2
        assert config.commands[0] == ["curl", "-fsSL", "https://ollama.com/install.sh"]
        assert config.prerequisite_check == "curl"

    def test_arch_installer_config(self):
        """Test Arch installer configuration."""
        installer = ArchInstaller()
        config = installer.get_config()

        assert config.commands == [["sudo", "pacman", "-Sy", "--noconfirm", "ollama"]]
        assert config.prerequisite_check == "pacman"

    def test_windows_installer_config(self):
        """Test Windows installer configuration."""
        installer = WindowsInstaller()
        config = installer.get_config()

        assert len(config.commands) == 2
        assert config.commands[0][0] == "powershell"
        assert config.prerequisite_check == "powershell"


class TestOllamaManager:
    """Test OllamaManager class."""

    @patch("platform.system")
    def test_detect_platform_macos(self, mock_system):
        """Test platform detection for macOS."""
        mock_system.return_value = "Darwin"
        manager = OllamaManager()

        assert manager.platform == PlatformType.MACOS

    @patch("platform.system")
    def test_detect_platform_windows(self, mock_system):
        """Test platform detection for Windows."""
        mock_system.return_value = "Windows"
        manager = OllamaManager()

        assert manager.platform == PlatformType.WINDOWS

    @patch("platform.system")
    def test_detect_platform_unsupported(self, mock_system):
        """Test platform detection for unsupported system."""
        mock_system.return_value = "FreeBSD"
        manager = OllamaManager()

        assert manager.platform == PlatformType.UNSUPPORTED

    @patch("platform.system")
    @patch(
        "builtins.open", new_callable=mock_open, read_data='ID=ubuntu\nVERSION="20.04"'
    )
    def test_detect_linux_distro_ubuntu(self, mock_file, mock_system):
        """Test Linux distribution detection for Ubuntu."""
        mock_system.return_value = "Linux"
        manager = OllamaManager()

        assert manager.platform == PlatformType.LINUX_DEBIAN

    @patch("platform.system")
    @patch("builtins.open", new_callable=mock_open, read_data='ID=debian\nVERSION="11"')
    def test_detect_linux_distro_debian(self, mock_file, mock_system):
        """Test Linux distribution detection for Debian."""
        mock_system.return_value = "Linux"
        manager = OllamaManager()

        assert manager.platform == PlatformType.LINUX_DEBIAN

    @patch("platform.system")
    @patch("builtins.open", new_callable=mock_open, read_data='ID=rhel\nVERSION="8"')
    def test_detect_linux_distro_rhel(self, mock_file, mock_system):
        """Test Linux distribution detection for RHEL."""
        mock_system.return_value = "Linux"
        manager = OllamaManager()

        assert manager.platform == PlatformType.LINUX_RHEL

    @patch("platform.system")
    @patch("builtins.open", new_callable=mock_open, read_data='ID=centos\nVERSION="7"')
    def test_detect_linux_distro_centos(self, mock_file, mock_system):
        """Test Linux distribution detection for CentOS."""
        mock_system.return_value = "Linux"
        manager = OllamaManager()

        assert manager.platform == PlatformType.LINUX_RHEL

    @patch("platform.system")
    @patch("builtins.open", new_callable=mock_open, read_data='ID=fedora\nVERSION="35"')
    def test_detect_linux_distro_fedora(self, mock_file, mock_system):
        """Test Linux distribution detection for Fedora."""
        mock_system.return_value = "Linux"
        manager = OllamaManager()

        assert manager.platform == PlatformType.LINUX_RHEL

    @patch("platform.system")
    @patch(
        "builtins.open", new_callable=mock_open, read_data='ID=arch\nVERSION="rolling"'
    )
    def test_detect_linux_distro_arch(self, mock_file, mock_system):
        """Test Linux distribution detection for Arch."""
        mock_system.return_value = "Linux"
        manager = OllamaManager()

        assert manager.platform == PlatformType.LINUX_ARCH

    @patch("platform.system")
    @patch("builtins.open", side_effect=FileNotFoundError)
    @patch("shutil.which")
    def test_detect_linux_distro_fallback_apt(self, mock_which, mock_file, mock_system):
        """Test Linux distribution detection fallback to apt."""
        mock_system.return_value = "Linux"
        mock_which.side_effect = lambda cmd: "/usr/bin/apt" if cmd == "apt" else None

        manager = OllamaManager()

        assert manager.platform == PlatformType.LINUX_DEBIAN

    @patch("platform.system")
    @patch("builtins.open", side_effect=FileNotFoundError)
    @patch("shutil.which")
    def test_detect_linux_distro_fallback_yum(self, mock_which, mock_file, mock_system):
        """Test Linux distribution detection fallback to yum."""
        mock_system.return_value = "Linux"
        mock_which.side_effect = lambda cmd: "/usr/bin/yum" if cmd == "yum" else None

        manager = OllamaManager()

        assert manager.platform == PlatformType.LINUX_RHEL

    @patch("platform.system")
    @patch("builtins.open", side_effect=FileNotFoundError)
    @patch("shutil.which")
    def test_detect_linux_distro_fallback_dnf(self, mock_which, mock_file, mock_system):
        """Test Linux distribution detection fallback to dnf."""
        mock_system.return_value = "Linux"
        mock_which.side_effect = lambda cmd: "/usr/bin/dnf" if cmd == "dnf" else None

        manager = OllamaManager()

        assert manager.platform == PlatformType.LINUX_RHEL

    @patch("platform.system")
    @patch("builtins.open", side_effect=FileNotFoundError)
    @patch("shutil.which")
    def test_detect_linux_distro_fallback_pacman(
        self, mock_which, mock_file, mock_system
    ):
        """Test Linux distribution detection fallback to pacman."""
        mock_system.return_value = "Linux"
        mock_which.side_effect = lambda cmd: (
            "/usr/bin/pacman" if cmd == "pacman" else None
        )

        manager = OllamaManager()

        assert manager.platform == PlatformType.LINUX_ARCH

    @patch("platform.system")
    @patch("builtins.open", side_effect=FileNotFoundError)
    @patch("shutil.which")
    def test_detect_linux_distro_default(self, mock_which, mock_file, mock_system):
        """Test Linux distribution detection default to Debian."""
        mock_system.return_value = "Linux"
        mock_which.return_value = None

        manager = OllamaManager()

        assert manager.platform == PlatformType.LINUX_DEBIAN

    @patch("subprocess.run")
    def test_run_command_without_streaming(self, mock_run):
        """Test running command without streaming."""
        mock_result = MagicMock(stdout="output", stderr="", returncode=0)
        mock_run.return_value = mock_result

        manager = OllamaManager()
        result = manager._run_command(["test", "command"], stream_output=False)

        assert result == mock_result
        mock_run.assert_called_once_with(
            ["test", "command"], capture_output=True, text=True, check=True
        )

    @patch("subprocess.Popen")
    def test_run_command_with_streaming(self, mock_popen):
        """Test running command with streaming."""
        mock_process = MagicMock()
        mock_process.stdout.readline.side_effect = ["line1\n", "line2\n", ""]
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        manager = OllamaManager()
        result = manager._run_command(["test", "command"], stream_output=True)

        assert result.returncode == 0
        assert "line1" in result.stdout
        assert "line2" in result.stdout

    @patch("subprocess.Popen")
    def test_run_with_streaming_error(self, mock_popen):
        """Test streaming command that fails."""
        mock_process = MagicMock()
        mock_process.stdout.readline.side_effect = ["error line\n", ""]
        mock_process.wait.return_value = 1

        mock_popen.return_value = mock_process

        manager = OllamaManager()

        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            manager._run_command(["test", "command"], stream_output=True)

        assert exc_info.value.returncode == 1
        assert "error line" in exc_info.value.output

    @patch("subprocess.run")
    def test_is_ollama_installed_true(self, mock_run):
        """Test checking if Ollama is installed when it is."""
        mock_run.return_value = MagicMock(stdout="ollama version 0.1.0", returncode=0)

        manager = OllamaManager()
        assert manager.is_ollama_installed() is True

        mock_run.assert_called_once_with(
            ["ollama", "--version"], capture_output=True, text=True, check=True
        )

    @patch("subprocess.run")
    def test_is_ollama_installed_false(self, mock_run):
        """Test checking if Ollama is installed when it's not."""
        mock_run.side_effect = subprocess.CalledProcessError(1, ["ollama", "--version"])

        manager = OllamaManager()
        assert manager.is_ollama_installed() is False

    @patch("subprocess.run")
    def test_is_ollama_installed_file_not_found(self, mock_run):
        """Test checking if Ollama is installed when command not found."""
        mock_run.side_effect = FileNotFoundError()

        manager = OllamaManager()
        assert manager.is_ollama_installed() is False

    @patch("subprocess.run")
    def test_is_model_installed_true(self, mock_run):
        """Test checking if model is installed when it is."""
        mock_run.return_value = MagicMock(
            stdout="llama2:latest\ncodellama:latest", returncode=0
        )

        manager = OllamaManager()
        assert manager.is_model_installed("llama2:latest") is True

        mock_run.assert_called_once_with(
            ["ollama", "list"], capture_output=True, text=True, check=True
        )

    @patch("subprocess.run")
    def test_is_model_installed_false(self, mock_run):
        """Test checking if model is installed when it's not."""
        mock_run.return_value = MagicMock(stdout="codellama:latest", returncode=0)

        manager = OllamaManager()
        assert manager.is_model_installed("llama2:latest") is False

    @patch("subprocess.run")
    def test_is_model_installed_error(self, mock_run):
        """Test checking if model is installed when command fails."""
        mock_run.side_effect = subprocess.CalledProcessError(1, ["ollama", "list"])

        manager = OllamaManager()
        assert manager.is_model_installed("llama2:latest") is False

    @patch.object(OllamaManager, "is_ollama_installed")
    def test_install_ollama_already_installed(self, mock_is_installed):
        """Test installing Ollama when already installed."""
        mock_is_installed.return_value = True

        manager = OllamaManager()
        result = manager.install_ollama()

        assert result is True
        mock_is_installed.assert_called_once()

    @patch("platform.system")
    def test_install_ollama_unsupported_platform(self, mock_system):
        """Test installing Ollama on unsupported platform."""
        mock_system.return_value = "FreeBSD"

        manager = OllamaManager()
        manager.is_ollama_installed = MagicMock(return_value=False)

        with pytest.raises(OllamaInstallationError) as exc_info:
            manager.install_ollama()

        assert "Unsupported platform" in str(exc_info.value)

    @patch("platform.system")
    @patch.object(OllamaManager, "is_ollama_installed")
    @patch.object(MacOSInstaller, "install")
    def test_install_ollama_macos_success(
        self, mock_install, mock_is_installed, mock_system
    ):
        """Test successful Ollama installation on macOS."""
        mock_system.return_value = "Darwin"
        mock_is_installed.side_effect = [False, True]  # Not installed, then installed

        manager = OllamaManager()
        result = manager.install_ollama()

        assert result is True
        mock_install.assert_called_once()
        assert mock_is_installed.call_count == 2

    @patch("platform.system")
    @patch.object(OllamaManager, "is_ollama_installed")
    @patch.object(MacOSInstaller, "install")
    def test_install_ollama_installation_fails(
        self, mock_install, mock_is_installed, mock_system
    ):
        """Test Ollama installation when installer raises exception."""
        mock_system.return_value = "Darwin"
        mock_is_installed.return_value = False
        mock_install.side_effect = Exception("Installation error")

        manager = OllamaManager()

        with pytest.raises(OllamaInstallationError) as exc_info:
            manager.install_ollama()

        assert "Installation failed" in str(exc_info.value)

    @patch("platform.system")
    @patch.object(OllamaManager, "is_ollama_installed")
    @patch.object(MacOSInstaller, "install")
    def test_install_ollama_verification_fails(
        self, mock_install, mock_is_installed, mock_system
    ):
        """Test Ollama installation when verification fails."""
        mock_system.return_value = "Darwin"
        mock_is_installed.side_effect = [False, False]  # Not installed before and after

        manager = OllamaManager()

        with pytest.raises(OllamaInstallationError) as exc_info:
            manager.install_ollama()

        assert "Installation verification failed" in str(exc_info.value)

    @patch.object(OllamaManager, "is_model_installed")
    def test_install_model_already_installed(self, mock_is_installed):
        """Test installing model when already installed."""
        mock_is_installed.return_value = True

        manager = OllamaManager()
        result = manager.install_model("llama2:latest")

        assert result is True
        mock_is_installed.assert_called_once_with("llama2:latest")

    @patch.object(OllamaManager, "is_model_installed")
    @patch.object(OllamaManager, "_run_command")
    def test_install_model_success(self, mock_run, mock_is_installed):
        """Test successful model installation."""
        mock_is_installed.side_effect = [False, True]  # Not installed, then installed
        mock_run.return_value = MagicMock(returncode=0)

        manager = OllamaManager()
        result = manager.install_model("llama2:latest")

        assert result is True
        mock_run.assert_called_once_with(
            ["ollama", "pull", "llama2:latest"], stream_output=True
        )
        assert mock_is_installed.call_count == 2

    @patch.object(OllamaManager, "is_model_installed")
    @patch.object(OllamaManager, "_run_command")
    def test_install_model_command_fails(self, mock_run, mock_is_installed):
        """Test model installation when command fails."""
        mock_is_installed.return_value = False
        mock_run.side_effect = Exception("Pull failed")

        manager = OllamaManager()

        with pytest.raises(ModelInstallationError) as exc_info:
            manager.install_model("llama2:latest")

        assert "Model installation failed" in str(exc_info.value)

    @patch.object(OllamaManager, "is_model_installed")
    @patch.object(OllamaManager, "_run_command")
    def test_install_model_verification_fails(self, mock_run, mock_is_installed):
        """Test model installation when verification fails."""
        mock_is_installed.side_effect = [False, False]  # Not installed before and after
        mock_run.return_value = MagicMock(returncode=0)

        manager = OllamaManager()

        with pytest.raises(ModelInstallationError) as exc_info:
            manager.install_model("llama2:latest")

        assert "Model installation verification failed" in str(exc_info.value)


class TestAutoInstallFunctions:
    """Test auto-install convenience functions."""

    @patch.object(OllamaManager, "install_ollama")
    def test_auto_install_ollama_success(self, mock_install):
        """Test auto_install_ollama function success."""
        mock_install.return_value = True

        result = auto_install_ollama()

        assert result is True
        mock_install.assert_called_once()

    @patch.object(OllamaManager, "install_ollama")
    def test_auto_install_ollama_failure(self, mock_install):
        """Test auto_install_ollama function failure."""
        mock_install.side_effect = OllamaInstallationError("Failed")

        with pytest.raises(OllamaInstallationError):
            auto_install_ollama()

    @patch.object(OllamaManager, "install_model")
    def test_auto_install_model_success(self, mock_install):
        """Test auto_install_model function success."""
        mock_install.return_value = True

        result = auto_install_model("llama2:latest")

        assert result is True
        mock_install.assert_called_once_with("llama2:latest")

    @patch.object(OllamaManager, "install_model")
    def test_auto_install_model_failure(self, mock_install):
        """Test auto_install_model function failure."""
        mock_install.side_effect = ModelInstallationError("Failed")

        with pytest.raises(ModelInstallationError):
            auto_install_model("llama2:latest")


# class TestIntegrationScenarios:
#     """Test integration scenarios combining multiple components."""

#     @patch('platform.system')
#     @patch('builtins.open', new_callable=mock_open, read_data='ID=ubuntu\nVERSION="20.04"')
#     @patch('shutil.which')
#     @patch('subprocess.run')
#     def test_full_installation_flow_ubuntu(self, mock_run, mock_which, mock_file, mock_system):
#         """Test complete installation flow on Ubuntu."""
#         mock_system.return_value = "Linux"
#         mock_which.side_effect = lambda cmd: "/usr/bin/curl" if cmd == "curl" else None

#         # Mock subprocess.run for different commands
#         def run_side_effect(*args, **kwargs):
#             cmd = args[0]
#             if cmd == ["ollama", "--version"]:
#                 # First call: not installed, second call: installed
#                 if not hasattr(run_side_effect, 'ollama_installed'):
#                     run_side_effect.ollama_installed = True
