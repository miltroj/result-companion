import subprocess
from unittest.mock import MagicMock, patch

import pytest

from result_companion.core.analizers.local.ollama_install import (
    DefaultSubprocessRunner,
    ModelInstallationError,
    OllamaCommands,
    OllamaInstallationError,
    OllamaInstallationManager,
    PlatformCommands,
    PlatformType,
    auto_install_model,
    auto_install_ollama,
)

OLLAMA_INSTALL_PATH = "result_companion.core.analizers.local.ollama_install"


class MockSubprocessRunner:
    """Mock implementation of SubprocessRunner for testing."""

    def __init__(self, success=True, stdout="", stderr="", raise_on_commands=None):
        """
        Initialize the mock runner.

        Args:
            success: Whether commands should succeed by default
            stdout: Default stdout output
            stderr: Default stderr output
            raise_on_commands: Dict mapping commands to exceptions to raise
        """
        self.success = success
        self.stdout = stdout
        self.stderr = stderr
        self.raise_on_commands = raise_on_commands or {}
        self.called_commands = []
        self.call_count = 0

    def run(self, cmd: list, check: bool = True) -> subprocess.CompletedProcess:
        """Mock running a subprocess command."""
        self.called_commands.append(cmd)

        # Convert command list to string for matching
        cmd_str = " ".join(cmd)

        # Check if this command should raise an exception
        for pattern, exception in self.raise_on_commands.items():
            if pattern in cmd_str:
                raise exception

        # Create a mock CompletedProcess
        result = MagicMock(spec=subprocess.CompletedProcess)
        result.returncode = 0 if self.success else 1
        result.stdout = self.stdout
        result.stderr = self.stderr

        if not self.success and check:
            raise subprocess.CalledProcessError(1, cmd, self.stdout, self.stderr)

        return result

    def run_with_streaming(self, cmd: list) -> subprocess.CompletedProcess:
        """Mock running a subprocess command with streaming output."""
        self.called_commands.append(cmd)
        self.call_count += 1

        # Convert command list to string for matching
        cmd_str = " ".join(cmd)

        # Check if this command should raise an exception
        for pattern, exception in self.raise_on_commands.items():
            if pattern in cmd_str:
                raise exception

        # Handle stdout as list for sequential responses
        current_stdout = self._get_current_output()

        # Create a CompletedProcess result
        result = subprocess.CompletedProcess(
            args=cmd,
            returncode=0 if self.success else 1,
            stdout=current_stdout,
            stderr="",  # stderr is empty because it's merged with stdout
        )

        if not self.success:
            # When streaming fails, we should raise CalledProcessError with the output
            # Note: stderr is empty because it's merged with stdout in streaming mode
            error = subprocess.CalledProcessError(1, cmd, current_stdout)
            error.stdout = current_stdout
            error.stderr = ""
            raise error

        return result

    def _get_current_output(self):
        """Get the current output based on call count."""
        if isinstance(self.stdout, list):
            # If stdout is a list, return the appropriate item based on call count
            index = min(self.call_count - 1, len(self.stdout) - 1)
            return self.stdout[index]
        return self.stdout


class TestOllamaInstallationManager:
    """Test suite for OllamaInstallationManager."""

    @pytest.fixture
    def successful_runner(self):
        """Fixture for a subprocess runner that succeeds."""
        return MockSubprocessRunner(success=True)

    @pytest.fixture
    def failing_runner(self):
        """Fixture for a subprocess runner that fails."""
        return MockSubprocessRunner(success=False)

    @pytest.fixture
    def model_list_runner(self):
        """Fixture for a subprocess runner that returns a list of models."""
        return MockSubprocessRunner(
            success=True,
            stdout="NAME            ID        SIZE     MODIFIED\nllama2          abcdef    1.2 GB   1 day ago\nmistral         ghijkl    2.3 GB   2 days ago\n",
        )

    @pytest.fixture
    def selective_failing_runner(self):
        """Fixture for a runner that fails on specific commands."""
        return MockSubprocessRunner(
            success=True,
            raise_on_commands={
                "ollama --version": FileNotFoundError("Command not found"),
                "ollama list": subprocess.CalledProcessError(1, ["ollama", "list"]),
                "apt-get": subprocess.CalledProcessError(
                    1, ["apt-get"], "Failed to update"
                ),
                "brew install": subprocess.CalledProcessError(
                    1, ["brew", "install"], "Failed to install"
                ),
            },
        )

    @pytest.fixture
    def commands(self):
        """Fixture for standard Ollama commands."""
        return OllamaCommands()

    @pytest.fixture
    def platform_commands(self):
        """Fixture for platform-specific commands."""
        return PlatformCommands()

    #
    # Platform Detection Tests
    #

    @patch(f"{OLLAMA_INSTALL_PATH}.platform", "darwin")
    def test_detect_platform_macos(self):
        """Test platform detection for macOS."""
        manager = OllamaInstallationManager()
        assert manager.platform_type == PlatformType.MACOS

    @patch(f"{OLLAMA_INSTALL_PATH}.platform", "linux")
    def test_detect_platform_linux(self):
        """Test platform detection for Linux."""
        manager = OllamaInstallationManager()
        assert manager.platform_type == PlatformType.LINUX

    @patch(f"{OLLAMA_INSTALL_PATH}.platform", "linux2")
    def test_detect_platform_linux2(self):
        """Test platform detection for Linux2."""
        manager = OllamaInstallationManager()
        assert manager.platform_type == PlatformType.LINUX

    @patch(f"{OLLAMA_INSTALL_PATH}.platform", "win32")
    def test_detect_platform_windows(self):
        """Test platform detection for Windows."""
        manager = OllamaInstallationManager()
        assert manager.platform_type == PlatformType.UNSUPPORTED

    #
    # Ollama Installation Check Tests
    #

    def test_is_ollama_installed_success(self, successful_runner, commands):
        """Test checking if Ollama is installed when it is."""
        manager = OllamaInstallationManager(
            subprocess_runner=successful_runner, ollama_commands=commands
        )
        assert manager.is_ollama_installed() is True
        assert successful_runner.called_commands == [commands.version_cmd]

    def test_is_ollama_installed_not_found(self, selective_failing_runner, commands):
        """Test checking if Ollama is installed when it's not found."""
        manager = OllamaInstallationManager(
            subprocess_runner=selective_failing_runner, ollama_commands=commands
        )
        assert manager.is_ollama_installed() is False
        assert selective_failing_runner.called_commands == [commands.version_cmd]

    def test_is_ollama_installed_error(self, failing_runner, commands):
        """Test checking if Ollama is installed when command fails."""
        manager = OllamaInstallationManager(
            subprocess_runner=failing_runner, ollama_commands=commands
        )
        assert manager.is_ollama_installed() is False
        assert failing_runner.called_commands == [commands.version_cmd]

    #
    # Model Installation Check Tests
    #

    def test_is_model_installed_success(self, model_list_runner, commands):
        """Test checking if a model is installed when it is."""
        manager = OllamaInstallationManager(
            subprocess_runner=model_list_runner, ollama_commands=commands
        )
        assert manager.is_model_installed("llama2") is True
        assert model_list_runner.called_commands == [commands.list_cmd]

    def test_is_model_installed_not_found(self, model_list_runner, commands):
        """Test checking if a model is installed when it's not."""
        manager = OllamaInstallationManager(
            subprocess_runner=model_list_runner, ollama_commands=commands
        )
        assert manager.is_model_installed("gpt4") is False
        assert model_list_runner.called_commands == [commands.list_cmd]

    def test_is_model_installed_command_error(self, selective_failing_runner, commands):
        """Test checking if a model is installed when command fails."""
        manager = OllamaInstallationManager(
            subprocess_runner=selective_failing_runner, ollama_commands=commands
        )
        assert manager.is_model_installed("llama2") is False
        assert selective_failing_runner.called_commands == [commands.list_cmd]

    #
    # Ollama Installation Tests
    #

    @patch.object(
        OllamaInstallationManager, "_detect_platform", return_value=PlatformType.MACOS
    )
    def test_install_ollama_already_installed(
        self, mock_detect, successful_runner, commands
    ):
        """Test installing Ollama when it's already installed."""
        manager = OllamaInstallationManager(
            subprocess_runner=successful_runner, ollama_commands=commands
        )
        assert manager.install_ollama() is True
        assert successful_runner.called_commands == [commands.version_cmd]

    @patch.object(
        OllamaInstallationManager, "_detect_platform", return_value=PlatformType.MACOS
    )
    def test_install_ollama_macos_success(
        self, mock_detect, selective_failing_runner, commands, platform_commands
    ):
        """Test successful Ollama installation on macOS."""
        # First check fails, then succeeds after installation
        runner = MockSubprocessRunner(
            success=True,
            raise_on_commands={
                "ollama --version": FileNotFoundError("Command not found"),
            },
        )

        manager = OllamaInstallationManager(
            subprocess_runner=runner,
            ollama_commands=commands,
            platform_commands=platform_commands,
        )
        manager.is_ollama_installed = MagicMock(side_effect=[False, True])

        assert manager.install_ollama() is True
        assert runner.called_commands == [
            platform_commands.mac_install_cmd,
        ]
        manager.is_ollama_installed.assert_called()

    @patch.object(
        OllamaInstallationManager, "_detect_platform", return_value=PlatformType.LINUX
    )
    def test_install_ollama_linux_success(self, commands, platform_commands):
        """Test successful Ollama installation on Linux."""
        # First check fails, then succeeds after installation
        runner = MockSubprocessRunner(
            success=True,
            raise_on_commands={
                "ollama --version": FileNotFoundError("Command not found"),
            },
        )

        manager = OllamaInstallationManager(
            subprocess_runner=runner,
            ollama_commands=commands,
            platform_commands=platform_commands,
        )
        manager.is_ollama_installed = MagicMock(side_effect=[False, True])

        assert manager._detect_platform() == PlatformType.LINUX
        assert manager.install_ollama() is True
        assert runner.called_commands == [
            platform_commands.linux_update_cmd,
            platform_commands.linux_install_cmd,
        ]
        manager.is_ollama_installed.assert_called()

    @patch.object(
        OllamaInstallationManager,
        "_detect_platform",
        return_value=PlatformType.UNSUPPORTED,
    )
    def test_install_ollama_unsupported_platform(self, mock_detect, successful_runner):
        """Test installing Ollama on an unsupported platform."""
        manager = OllamaInstallationManager(subprocess_runner=successful_runner)
        manager.is_ollama_installed = MagicMock(return_value=False)

        with pytest.raises(OllamaInstallationError) as exc_info:
            manager.install_ollama()

        assert "not supported on your OS" in str(exc_info.value)

    @patch.object(
        OllamaInstallationManager, "_detect_platform", return_value=PlatformType.MACOS
    )
    def test_install_ollama_macos_failure(
        self, mock_detect, selective_failing_runner, commands, platform_commands
    ):
        """Test failed Ollama installation on macOS."""
        manager = OllamaInstallationManager(
            subprocess_runner=selective_failing_runner,
            ollama_commands=commands,
            platform_commands=platform_commands,
        )

        with pytest.raises(OllamaInstallationError) as exc_info:
            manager.install_ollama()

        assert "Homebrew failed" in str(exc_info.value)
        assert selective_failing_runner.called_commands == [
            commands.version_cmd,
            platform_commands.mac_install_cmd,
        ]

    @patch.object(
        OllamaInstallationManager, "_detect_platform", return_value=PlatformType.LINUX
    )
    def test_install_ollama_linux_update_failure(
        self, mock_detect, selective_failing_runner, commands, platform_commands
    ):
        """Test failed apt-get update during Ollama installation on Linux."""
        manager = OllamaInstallationManager(
            subprocess_runner=selective_failing_runner,
            ollama_commands=commands,
            platform_commands=platform_commands,
        )

        with pytest.raises(OllamaInstallationError) as exc_info:
            manager.install_ollama()

        assert "apt-get failed" in str(exc_info.value)
        assert selective_failing_runner.called_commands == [
            commands.version_cmd,
            platform_commands.linux_update_cmd,
        ]

    @patch.object(
        OllamaInstallationManager, "_detect_platform", return_value=PlatformType.MACOS
    )
    def test_install_ollama_verification_failure(
        self, mock_detect, commands, platform_commands
    ):
        """Test Ollama installation that appears to succeed but verification fails."""
        # First check fails, installation succeeds, but second check still fails
        runner = MockSubprocessRunner(
            success=True,
            raise_on_commands={
                "ollama --version": FileNotFoundError("Command not found")
            },
        )

        manager = OllamaInstallationManager(
            subprocess_runner=runner,
            ollama_commands=commands,
            platform_commands=platform_commands,
        )

        with pytest.raises(OllamaInstallationError) as exc_info:
            manager.install_ollama()

        assert "did not complete successfully" in str(exc_info.value)
        assert runner.called_commands == [
            commands.version_cmd,
            platform_commands.mac_install_cmd,
            commands.version_cmd,
        ]

    #
    # Model Installation Tests
    #

    def test_install_model_already_installed(self, model_list_runner, commands):
        """Test installing a model that's already installed."""
        manager = OllamaInstallationManager(
            subprocess_runner=model_list_runner, ollama_commands=commands
        )
        assert manager.install_model("llama2") is True
        assert model_list_runner.called_commands == [commands.list_cmd]

    def test_install_model_success(self, commands):
        """Test successful model installation."""
        # First check doesn't find model, then it does after installation
        runner = MockSubprocessRunner(
            success=True,
            stdout=[
                "",
                "NAME            ID        SIZE     MODIFIED\nllama2          abcdef    1.2 GB   1 day ago\n",
            ],
        )

        manager = OllamaInstallationManager(
            subprocess_runner=runner, ollama_commands=commands
        )
        manager.is_model_installed = MagicMock(side_effect=[False, True])
        assert manager.install_model("llama2") is True

        expected_install_cmd = commands.install_model_cmd + ["llama2"]
        assert runner.called_commands == [
            expected_install_cmd,
        ]

    def test_install_model_installation_failure(
        self, selective_failing_runner, commands
    ):
        """Test model installation failure."""
        # Make the install command fail
        runner = MockSubprocessRunner(
            success=True,
            stdout="",
            raise_on_commands={
                "ollama pull": subprocess.CalledProcessError(
                    returncode=1,
                    cmd=["ollama", "pull"],
                    output="Failed to install model",
                    stderr="Failed to install model",
                )
            },
        )

        manager = OllamaInstallationManager(
            subprocess_runner=runner, ollama_commands=commands
        )

        with pytest.raises(ModelInstallationError) as exc_info:
            manager.install_model("llama2")

        assert "failed" in str(exc_info.value)

        expected_install_cmd = commands.install_model_cmd + ["llama2"]
        assert runner.called_commands == [commands.list_cmd, expected_install_cmd]

    def test_install_model_verification_failure(self, commands):
        """Test model installation that appears to succeed but verification fails."""
        # Installation succeeds but model still not found in list
        runner = MockSubprocessRunner(success=True, stdout="")  # Empty model list

        manager = OllamaInstallationManager(
            subprocess_runner=runner, ollama_commands=commands
        )

        with pytest.raises(ModelInstallationError) as exc_info:
            manager.install_model("llama2")

        assert "did not complete successfully" in str(exc_info.value)

        expected_install_cmd = commands.install_model_cmd + ["llama2"]
        assert runner.called_commands == [
            commands.list_cmd,
            expected_install_cmd,
            commands.list_cmd,
        ]

    #
    # Default SubprocessRunner Tests
    #

    @patch("subprocess.Popen")
    def test_default_subprocess_runner_streaming(self, mock_popen):
        """Test the DefaultSubprocessRunner run_with_streaming implementation."""
        # Set up the mock process
        mock_process = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.readline.side_effect = ["line1\n", "line2\n", ""]
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        runner = DefaultSubprocessRunner()
        result = runner.run_with_streaming(["echo", "hello"])

        # Check that Popen was called with correct arguments
        mock_popen.assert_called_once_with(
            ["echo", "hello"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Verify the result
        assert result.args == ["echo", "hello"]
        assert result.returncode == 0
        assert result.stdout == "line1\nline2"
        assert result.stderr == ""

    #
    # Command Configuration Tests
    #

    def test_ollama_commands_defaults(self):
        """Test OllamaCommands default values."""
        commands = OllamaCommands()
        assert commands.version_cmd == ["ollama", "--version"]
        assert commands.list_cmd == ["ollama", "list"]
        assert commands.install_model_cmd == ["ollama", "pull"]

    def test_platform_commands_defaults(self):
        """Test PlatformCommands default values."""
        commands = PlatformCommands()
        assert commands.mac_install_cmd == ["brew", "install", "ollama"]
        assert commands.linux_update_cmd == ["sudo", "apt-get", "update"]
        assert commands.linux_install_cmd == [
            "sudo",
            "apt-get",
            "install",
            "-y",
            "ollama",
        ]


class TestInstallationInterface:
    """Test suite for the installation facade."""

    @pytest.fixture
    def mock_manager(self):
        """Fixture for a mock OllamaInstallationManager."""
        return MagicMock(spec=OllamaInstallationManager)

    @patch(f"{OLLAMA_INSTALL_PATH}.OllamaInstallationManager")
    def test_auto_install_ollama_success(self, mock_manager_class, mock_manager):
        """Test auto_install_ollama when installation succeeds."""
        # Setup mock
        mock_manager.install_ollama.return_value = True
        mock_manager_class.return_value = mock_manager

        # Call function
        result = auto_install_ollama()

        # Verify
        assert result is True
        mock_manager_class.assert_called_once()
        mock_manager.install_ollama.assert_called_once()

    @patch(f"{OLLAMA_INSTALL_PATH}.OllamaInstallationManager")
    def test_auto_install_ollama_failure(self, mock_manager_class, mock_manager):
        """Test auto_install_ollama when installation fails."""
        # Setup mock
        mock_manager.install_ollama.side_effect = OllamaInstallationError(
            "Failed to install"
        )
        mock_manager_class.return_value = mock_manager

        # Call function and verify exception
        with pytest.raises(Exception) as exc_info:
            auto_install_ollama()

        assert "Failed to install" in str(exc_info.value)
        mock_manager_class.assert_called_once()
        mock_manager.install_ollama.assert_called_once()

    @patch(f"{OLLAMA_INSTALL_PATH}.OllamaInstallationManager")
    def test_auto_install_model_success(self, mock_manager_class, mock_manager):
        """Test auto_install_model when installation succeeds."""
        # Setup mock
        mock_manager.install_model.return_value = True
        mock_manager_class.return_value = mock_manager

        # Call function
        result = auto_install_model("llama2")

        # Verify
        assert result is True
        mock_manager_class.assert_called_once()
        mock_manager.install_model.assert_called_once_with("llama2")

    @patch(f"{OLLAMA_INSTALL_PATH}.OllamaInstallationManager")
    def test_auto_install_model_failure(self, mock_manager_class, mock_manager):
        """Test auto_install_model when installation fails."""
        # Setup mock
        mock_manager.install_model.side_effect = ModelInstallationError(
            "Failed to install model"
        )
        mock_manager_class.return_value = mock_manager

        # Call function and verify exception
        with pytest.raises(Exception) as exc_info:
            auto_install_model("llama2")

        assert "Failed to install model" in str(exc_info.value)
        mock_manager_class.assert_called_once()
        mock_manager.install_model.assert_called_once_with("llama2")


class TestDefaultSubprocessRunner:
    """Test suite for DefaultSubprocessRunner."""

    def test_run_success(self):
        """Test running a command successfully."""
        runner = DefaultSubprocessRunner()
        result = runner.run(["echo", "hello"])

        assert result.returncode == 0
        assert result.stdout.strip() == "hello"
        assert result.stderr == ""

    def test_run_failure(self):
        """Test running a command that fails."""
        runner = DefaultSubprocessRunner()

        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            runner.run(["false"])

        assert exc_info.value.returncode != 0
