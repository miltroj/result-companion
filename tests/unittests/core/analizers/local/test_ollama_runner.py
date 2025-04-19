import unittest
from unittest.mock import patch, MagicMock
import pytest
import sys

from result_companion.core.analizers.local.ollama_runner import (  # Assuming these are in your ollama_runner module
    ollama_on_init_strategy,
    check_ollama_installed,
    is_ollama_server_running,
    start_ollama_server,
    check_model_installed,
    OllamaNotInstalled,
    OllamaServerNotRunning,
    OllamaModelNotAvailable,
)


class TestOllamaOnInitStrategy(unittest.TestCase):
    def setUp(self):
        self.model_name = "llama3"
        self.server_url = "http://localhost:11434"

    @patch("result_companion.core.analizers.local.ollama_runner.check_ollama_installed")
    def test_ollama_not_installed(self, mock_check_ollama_installed):
        mock_check_ollama_installed.side_effect = OllamaNotInstalled
        with self.assertRaises(OllamaNotInstalled):
            ollama_on_init_strategy(self.model_name, self.server_url)

    @patch("result_companion.core.analizers.local.ollama_runner.start_ollama_server")
    @patch(
        "result_companion.core.analizers.local.ollama_runner.is_ollama_server_running"
    )
    @patch("result_companion.core.analizers.local.ollama_runner.check_ollama_installed")
    def test_ollama_server_fails_to_start(
        self,
        mock_check_ollama_installed,
        mock_is_ollama_server_running,
        mock_start_ollama_server,
    ):
        mock_is_ollama_server_running.return_value = False
        mock_start_ollama_server.side_effect = OllamaServerNotRunning
        with self.assertRaises(OllamaServerNotRunning):
            ollama_on_init_strategy(self.model_name, self.server_url)
        mock_start_ollama_server.assert_called_once_with(self.server_url, 30)
        mock_check_ollama_installed.assert_called_once()

    @patch("result_companion.core.analizers.local.ollama_runner.start_ollama_server")
    @patch(
        "result_companion.core.analizers.local.ollama_runner.is_ollama_server_running"
    )
    @patch("result_companion.core.analizers.local.ollama_runner.check_ollama_installed")
    @patch("result_companion.core.analizers.local.ollama_runner.check_model_installed")
    def test_ollama_server_starts_successfully(
        self,
        mock_check_model_installed,
        mock_check_ollama_installed,
        mock_is_ollama_server_running,
        mock_start_ollama_server,
    ):
        mock_is_ollama_server_running.side_effect = [False, True]
        mock_start_ollama_server.return_value = None
        ollama_on_init_strategy(self.model_name, self.server_url)
        mock_start_ollama_server.assert_called_once_with(self.server_url, 30)
        mock_check_ollama_installed.assert_called_once()
        mock_check_model_installed.assert_called_once_with(self.model_name)

    @patch(
        "result_companion.core.analizers.local.ollama_runner.is_ollama_server_running"
    )
    @patch("result_companion.core.analizers.local.ollama_runner.check_ollama_installed")
    @patch("result_companion.core.analizers.local.ollama_runner.check_model_installed")
    def test_ollama_server_already_running(
        self,
        mock_check_model_installed,
        mock_check_ollama_installed,
        mock_is_ollama_server_running,
    ):
        mock_is_ollama_server_running.return_value = True
        ollama_on_init_strategy(self.model_name, self.server_url)
        mock_is_ollama_server_running.assert_called_once_with(self.server_url)
        mock_check_ollama_installed.assert_called_once()
        mock_check_model_installed.assert_called_once_with(self.model_name)

    @patch("result_companion.core.analizers.local.ollama_runner.check_model_installed")
    @patch(
        "result_companion.core.analizers.local.ollama_runner.is_ollama_server_running"
    )
    @patch("result_companion.core.analizers.local.ollama_runner.check_ollama_installed")
    def test_ollama_model_not_available(
        self,
        mock_check_ollama_installed,
        mock_is_ollama_server_running,
        mock_check_model_installed,
    ):
        mock_is_ollama_server_running.return_value = True
        mock_check_model_installed.side_effect = OllamaModelNotAvailable
        with self.assertRaises(OllamaModelNotAvailable):
            ollama_on_init_strategy(self.model_name, self.server_url)
        mock_check_model_installed.assert_called_once_with(self.model_name)

    @patch("result_companion.core.analizers.local.ollama_runner.check_model_installed")
    @patch(
        "result_companion.core.analizers.local.ollama_runner.is_ollama_server_running"
    )
    @patch("result_companion.core.analizers.local.ollama_runner.check_ollama_installed")
    def test_ollama_model_available(
        self,
        mock_check_ollama_installed,
        mock_is_ollama_server_running,
        mock_check_model_installed,
    ):
        mock_is_ollama_server_running.return_value = True
        mock_check_model_installed.return_value = None
        ollama_on_init_strategy(self.model_name, self.server_url)
        mock_check_model_installed.assert_called_once_with(self.model_name)

    @patch("result_companion.core.analizers.local.ollama_runner.check_model_installed")
    @patch(
        "result_companion.core.analizers.local.ollama_runner.is_ollama_server_running"
    )
    @patch("result_companion.core.analizers.local.ollama_runner.check_ollama_installed")
    def test_successful_initialization(
        self,
        mock_check_ollama_installed,
        mock_is_ollama_server_running,
        mock_check_model_installed,
    ):
        mock_is_ollama_server_running.return_value = True
        mock_check_model_installed.return_value = None
        ollama_on_init_strategy(self.model_name, self.server_url)
        mock_check_ollama_installed.assert_called_once()
        mock_is_ollama_server_running.assert_called_once_with(self.server_url)
        mock_check_model_installed.assert_called_once_with(self.model_name)


@pytest.mark.skipif(sys.platform == "win32", reason="Test not applicable on Windows")
def test_ollama_is_installed() -> None:
    assert check_ollama_installed(ollama_version=["echo", "'Installed!'"]) is None


@pytest.mark.skipif(sys.platform == "win32", reason="Test not applicable on Windows")
def test_ollama_is_not_installed() -> None:
    with pytest.raises(Exception, match="Ollama is not installed."):
        check_ollama_installed(ollama_version=["exit", "1"])


@pytest.mark.skipif(sys.platform == "win32", reason="Test not applicable on Windows")
def test_model_is_installed() -> None:
    assert (
        check_model_installed(
            model_name="llama31", ollama_list_cmd=["echo", "'llama31'"]
        )
        is None
    )


@pytest.mark.skipif(sys.platform == "win32", reason="Test not applicable on Windows")
def test_model_is_not_installed() -> None:
    with pytest.raises(
        Exception,
        match="Failed to check if model is installed: Model not_exisitng_model is not installed.",
    ):
        check_model_installed(
            model_name="not_exisitng_model", ollama_list_cmd=["echo", "'llama31'"]
        )


@patch("result_companion.core.analizers.local.ollama_runner.check_model_installed")
@patch("result_companion.core.analizers.local.ollama_runner.is_ollama_server_running")
@patch("result_companion.core.analizers.local.ollama_runner.check_ollama_installed")
def test_ollama_on_init_strategy(
    mock_ollama_installed, mock_is_ollama_server_running, mock_model_installed
) -> None:
    assert (
        ollama_on_init_strategy(
            model_name="llama311", server_url="url", start_timeout=100
        )
        is None
    )
    mock_ollama_installed.assert_called_once()
    mock_model_installed.assert_called_once_with("llama311")
    mock_is_ollama_server_running.assert_called_once_with("url")
