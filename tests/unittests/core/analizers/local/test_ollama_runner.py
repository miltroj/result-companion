import sys
from unittest.mock import patch

import pytest

from result_companion.core.analizers.local.ollama_runner import (
    check_model_installed,
    check_ollama_installed,
    ollama_on_init_strategy,
)


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
@patch("result_companion.core.analizers.local.ollama_runner.check_ollama_installed")
def test_ollama_on_init_strategy(mock_ollama_installed, mock_model_installed) -> None:
    assert ollama_on_init_strategy(model_name="llama31") is None
    mock_ollama_installed.assert_called_once()
    mock_model_installed.assert_called_once_with(model_name="llama31")
