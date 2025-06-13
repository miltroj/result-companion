import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer import echo
from typer.testing import CliRunner

from result_companion.core.analizers.local.ollama_server_manager import (
    OllamaServerManager,
)
from result_companion.entrypoints.cli.cli_app import (
    app,
    get_installed_models,
    install_ollama_model,
    setup_app,
)

existing_xml_path = Path(__file__).parent / "empty.xml"
IMPORT_PATH = "result_companion.entrypoints.cli.cli_app"


class TestAnalizeEntrypoint:
    ENTRYPOINT = "analyze"

    def setup_method(self):
        """Setup common test fixtures"""
        self.runner = CliRunner()

    def test_cli_fail_when_outpu_not_exists(self):
        result = self.runner.invoke(
            app, [self.ENTRYPOINT, "-o", "not_exists.xml"], obj={}
        )
        assert result.exit_code == 2
        assert "File 'not_exists.xml' does not exist" in result.output

    def test_cli_fail_when_config_not_exists(self):
        result = self.runner.invoke(
            app,
            [self.ENTRYPOINT, "-o", existing_xml_path, "-c", "config_not_exists"],
            obj={},
        )
        assert result.exit_code == 2
        assert "File 'config_not_exists' does not" in result.output

    def test_cli_by_default_uses_include_passing_false(self):
        result = self.runner.invoke(
            app, [self.ENTRYPOINT, "-o", existing_xml_path], obj={}
        )
        assert result.exit_code == 0
        assert "Include Passing: False" in result.output

    def test_cli_sets_include_passing(self):
        result = self.runner.invoke(
            app, [self.ENTRYPOINT, "-o", existing_xml_path, "-i"], obj={}
        )
        assert result.exit_code == 0
        assert "Include Passing: True" in result.output

    def test_cli_stets_generating_report(self):
        result = self.runner.invoke(
            app, [self.ENTRYPOINT, "-o", existing_xml_path, "-r", "report.html"], obj={}
        )
        assert result.exit_code == 0
        assert "Report: report.html" in result.output

    def test_cli_sets_info_as_default_log_level(self):
        result = self.runner.invoke(
            app, [self.ENTRYPOINT, "-o", existing_xml_path], obj={}
        )
        assert result.exit_code == 0
        assert "Log Level: INFO" in result.output

    def test_cli_calls_main_function(self):
        result = self.runner.invoke(
            app,
            [self.ENTRYPOINT, "-o", existing_xml_path],
            obj={"main": lambda *args: echo("RUNNING MAIN")},
        )
        assert result.exit_code == 0
        assert "Output: " in result.output
        assert "Log Level: " in result.output
        assert "Config: " in result.output
        assert "Report: " in result.output
        assert "Include Passing: " in result.output


class TestInstallOllamaModel:

    def setup_method(self):
        self.mock_server_manager = MagicMock(spec=OllamaServerManager)
        self.mock_server_manager.__enter__ = MagicMock(
            return_value=self.mock_server_manager
        )
        self.mock_server_manager.__exit__ = MagicMock(return_value=None)

    def test_success(self):
        with patch(f"{IMPORT_PATH}.check_ollama_installed") as mock_check, patch(
            f"{IMPORT_PATH}.auto_install_model", return_value=True
        ) as mock_auto_install, patch(
            f"{IMPORT_PATH}.resolve_server_manager",
            return_value=self.mock_server_manager,
        ) as mock_resolve:

            # Call the function
            result = install_ollama_model("llama2")

            # Verify all mocks were called correctly
            mock_check.assert_called_once()
            mock_resolve.assert_called_once_with(OllamaServerManager)
            self.mock_server_manager.__enter__.assert_called_once()
            mock_auto_install.assert_called_once_with(
                model_name="llama2",
                installation_cmd=["ollama", "pull"],
                ollama_list_cmd=["ollama", "list"],
            )
            assert result is True

    def test_installation_of_not_existing_model_failure(self):
        """Test handling of installation failure"""
        with patch(f"{IMPORT_PATH}.check_ollama_installed"), patch(
            f"{IMPORT_PATH}.auto_install_model", return_value=False
        ) as mock_auto_install, patch(
            f"{IMPORT_PATH}.resolve_server_manager",
            return_value=self.mock_server_manager,
        ):

            with pytest.raises(Exception) as exc_info:
                install_ollama_model("nonexistent-model")

            assert "Failed to install model 'nonexistent-model'" in str(exc_info.value)
            mock_auto_install.assert_called_once()

    def test_ollama_not_installed(self):
        """Test handling when Ollama is not installed"""
        with patch(
            f"{IMPORT_PATH}.check_ollama_installed",
            side_effect=Exception("Ollama not installed"),
        ), patch(f"{IMPORT_PATH}.auto_install_model") as mock_auto_install:

            with pytest.raises(Exception) as exc_info:
                install_ollama_model("llama2")

            assert "Ollama not installed" in str(exc_info.value)
            mock_auto_install.assert_not_called()

    def test_server_start_failure(self):
        """Test handling when server fails to start"""
        # Override the default mock to simulate failure
        failing_server_manager = MagicMock(spec=OllamaServerManager)
        failing_server_manager.__enter__ = MagicMock(
            side_effect=Exception("Server failed to start")
        )

        with patch(f"{IMPORT_PATH}.check_ollama_installed"), patch(
            f"{IMPORT_PATH}.resolve_server_manager", return_value=failing_server_manager
        ), patch(f"{IMPORT_PATH}.auto_install_model") as mock_auto_install:

            with pytest.raises(Exception) as exc_info:
                install_ollama_model("llama2")

            assert "Server failed to start" in str(exc_info.value)
            mock_auto_install.assert_not_called()


class TestSetupModelCommand:
    """Unit tests for the setup_model CLI command"""

    def setup_method(self):
        """Setup common test fixtures"""
        self.runner = CliRunner()

    def test_success(self):
        """Test successful CLI model installation"""
        with patch(
            f"{IMPORT_PATH}.install_ollama_model", return_value=True
        ) as mock_install:
            # Call the CLI command
            result = self.runner.invoke(setup_app, ["model", "llama2"])

            # Verify output and function call
            assert result.exit_code == 0
            assert "Installing model 'llama2'" in result.stdout
            assert "Model 'llama2' installed successfully" in result.stdout
            mock_install.assert_called_once_with("llama2")

    def test_failure(self):
        """Test CLI handling of installation failure"""
        with patch(
            f"{IMPORT_PATH}.install_ollama_model",
            side_effect=Exception("Installation failed"),
        ) as mock_install:
            # Call the CLI command
            result = self.runner.invoke(setup_app, ["model", "llama2"])

            # Verify output and exit code
            assert result.exit_code == 1
            assert "Installing model 'llama2'" in result.stdout
            assert (
                "Error installing model 'llama2': Installation failed" in result.stdout
            )
            mock_install.assert_called_once_with("llama2")

    def test_missing_argument(self):
        """Test CLI handling when model name is missing"""
        # Call without required argument
        result = self.runner.invoke(setup_app, ["model"])

        # Should fail with usage information
        assert result.exit_code != 0
        assert "Missing argument" in result.stdout or "Error" in result.stdout

    def test_cli_version(self):
        result = self.runner.invoke(app, ["--version"], obj={})
        assert result.exit_code == 0
        assert "result-companion version:" in result.output


class TestGetInstalledModels:
    """Unit tests for get_installed_models function"""

    def setup_method(self):
        """Setup common test fixtures"""
        self.mock_server_manager = MagicMock(spec=OllamaServerManager)
        self.mock_server_manager.__enter__ = MagicMock(
            return_value=self.mock_server_manager
        )
        self.mock_server_manager.__exit__ = MagicMock(return_value=None)

        self.mock_completed_process = MagicMock(spec=subprocess.CompletedProcess)
        self.mock_completed_process.stdout = "NAME            ID        SIZE     MODIFIED\nllama2          abcdef    1.2 GB   1 day ago\n"

        self.mock_command_runner = MagicMock(return_value=self.mock_completed_process)

    def test_successful_model_listing(self):
        """Test successfully getting the list of models"""
        with patch(
            f"{IMPORT_PATH}.resolve_server_manager",
            return_value=self.mock_server_manager,
        ):
            result = get_installed_models(command_runner=self.mock_command_runner)

            # Verify the result contains the expected output
            assert "llama2" in result
            assert "NAME" in result

            # Verify the command runner was called correctly
            self.mock_command_runner.assert_called_once_with(
                ["ollama", "list"], capture_output=True, text=True, check=True
            )

            # Verify server manager was used correctly
            self.mock_server_manager.__enter__.assert_called_once()
            self.mock_server_manager.__exit__.assert_called_once()

    def test_empty_model_list(self):
        """Test when no models are installed"""
        # Change the mock to return empty model list
        self.mock_completed_process.stdout = (
            "NAME            ID        SIZE     MODIFIED\n"
        )

        with patch(
            f"{IMPORT_PATH}.resolve_server_manager",
            return_value=self.mock_server_manager,
        ):
            result = get_installed_models(command_runner=self.mock_command_runner)

            # Verify the result contains only the header
            assert "NAME" in result
            assert "llama2" not in result

            # Verify the command was still called correctly
            self.mock_command_runner.assert_called_once()

    def test_server_start_failure(self):
        """Test when server fails to start"""
        # Override to simulate server startup failure
        failing_server_manager = MagicMock(spec=OllamaServerManager)
        failing_server_manager.__enter__ = MagicMock(
            side_effect=Exception("Server failed to start")
        )

        with patch(
            f"{IMPORT_PATH}.resolve_server_manager", return_value=failing_server_manager
        ):
            with pytest.raises(Exception) as exc_info:
                get_installed_models(command_runner=self.mock_command_runner)

            assert "Server failed to start" in str(exc_info.value)

            # Command should not have been called if server failed to start
            self.mock_command_runner.assert_not_called()

    def test_command_failure(self):
        """Test when the ollama list command fails"""
        # Make the command runner raise an exception
        failing_command_runner = MagicMock(
            side_effect=subprocess.CalledProcessError(
                returncode=1,
                cmd=["ollama", "list"],
                output="Command failed",
                stderr="Error listing models",
            )
        )

        with patch(
            f"{IMPORT_PATH}.resolve_server_manager",
            return_value=self.mock_server_manager,
        ):
            with pytest.raises(subprocess.CalledProcessError) as exc_info:
                get_installed_models(command_runner=failing_command_runner)

            assert exc_info.value.returncode == 1
            assert exc_info.value.cmd == ["ollama", "list"]

            # Server manager should still have been used correctly
            self.mock_server_manager.__enter__.assert_called_once()
            self.mock_server_manager.__exit__.assert_called_once()

    def test_custom_server_manager_instance(self):
        """Test using a custom server manager instance"""
        custom_server_manager = MagicMock(spec=OllamaServerManager)
        custom_server_manager.__enter__ = MagicMock(return_value=custom_server_manager)
        custom_server_manager.__exit__ = MagicMock(return_value=None)

        # No need to patch resolve_server_manager since we're passing the instance directly
        result = get_installed_models(
            server_manager=custom_server_manager,
            command_runner=self.mock_command_runner,
        )

        assert "llama2" in result
        custom_server_manager.__enter__.assert_called_once()
        custom_server_manager.__exit__.assert_called_once()

    def test_custom_server_manager_class(self):
        """Test using a custom server manager class"""
        # Create a mock for resolve_server_manager to verify it receives the class
        with patch(
            f"{IMPORT_PATH}.resolve_server_manager",
            return_value=self.mock_server_manager,
        ) as mock_resolve:
            custom_manager_class = MagicMock()

            result = get_installed_models(
                server_manager=custom_manager_class,
                command_runner=self.mock_command_runner,
            )

            assert "llama2" in result
            mock_resolve.assert_called_once_with(custom_manager_class)


class TestListModelsCommand:
    """Unit tests for the list_models CLI command"""

    def setup_method(self):
        """Setup common test fixtures"""
        self.runner = CliRunner()
        self.sample_model_output = "NAME            ID        SIZE     MODIFIED\nllama2          abcdef    1.2 GB   1 day ago\nmistral         ghijkl    2.3 GB   2 days ago\n"

    def test_successful_listing(self):
        """Test successful model listing"""
        with patch(
            f"{IMPORT_PATH}.get_installed_models", return_value=self.sample_model_output
        ):
            # Call the CLI command
            result = self.runner.invoke(setup_app, ["list-models"])

            # Verify output and exit code
            assert result.exit_code == 0
            assert "Installed models:" in result.stdout
            assert "llama2" in result.stdout
            assert "mistral" in result.stdout
            assert "1.2 GB" in result.stdout

    def test_empty_model_list(self):
        """Test when no models are installed"""
        empty_output = "NAME            ID        SIZE     MODIFIED\n"
        with patch(f"{IMPORT_PATH}.get_installed_models", return_value=empty_output):
            # Call the CLI command
            result = self.runner.invoke(setup_app, ["list-models"])

            # Verify output shows empty list but succeeds
            assert result.exit_code == 0
            assert "Installed models:" in result.stdout
            assert "NAME" in result.stdout
            assert "llama2" not in result.stdout

    def test_subprocess_error(self):
        """Test handling of subprocess errors"""
        with patch(
            f"{IMPORT_PATH}.get_installed_models",
            side_effect=subprocess.SubprocessError("Command failed"),
        ):
            # Call the CLI command
            result = self.runner.invoke(setup_app, ["list-models"])

            # Verify error handling
            assert result.exit_code == 1
            assert "Error: Failed to list models" in result.stdout
            assert "Is Ollama installed?" in result.stdout

    def test_generic_error(self):
        """Test handling of other errors"""
        with patch(
            f"{IMPORT_PATH}.get_installed_models",
            side_effect=Exception("Unexpected error"),
        ):
            # Call the CLI command
            result = self.runner.invoke(setup_app, ["list-models"])

            # Verify error handling
            assert result.exit_code == 1
            assert "Error: Unexpected error" in result.stdout

    @patch(f"{IMPORT_PATH}.logger")
    def test_debug_logging(self, mock_logger):
        """Test that model list is logged at debug level"""
        with patch(
            f"{IMPORT_PATH}.get_installed_models", return_value=self.sample_model_output
        ):
            # Call the CLI command
            result = self.runner.invoke(setup_app, ["list-models"])

            # Verify debug logging
            assert result.exit_code == 0
            mock_logger.debug.assert_called_once()
            # Check the model output is in the log message
            assert "llama2" in mock_logger.debug.call_args[0][0]
            assert "mistral" in mock_logger.debug.call_args[0][0]

    def test_calledprocesserror_handling(self):
        """Test specific handling of CalledProcessError"""
        # This is a more specific subprocess error that might occur
        error = subprocess.CalledProcessError(
            returncode=1,
            cmd=["ollama", "list"],
            output="",
            stderr="Error: ollama command not found",
        )

        with patch(f"{IMPORT_PATH}.get_installed_models", side_effect=error):
            # Call the CLI command
            result = self.runner.invoke(setup_app, ["list-models"])

            # Verify error handling
            assert result.exit_code == 1
            assert "Error: Failed to list models" in result.stdout
