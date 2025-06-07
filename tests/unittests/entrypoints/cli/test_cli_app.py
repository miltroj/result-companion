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
