import sys
from unittest.mock import patch

import pytest

from result_companion.core.analizers.local.ollama_runner import (
    auto_install_model,
    auto_install_ollama,
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


def test_auto_install_ollama_on_mac() -> None:
    with patch(
        "result_companion.core.analizers.local.ollama_runner.platform",
        return_value="darwin",
    ):
        assert auto_install_ollama(
            brew_installation_cmd=["echo", "'Ollama installed!'"],
            ollama_version=["echo", "'Installed!'"],
        )


def test_auto_install_ollama_on_linux() -> None:
    with patch(
        "result_companion.core.analizers.local.ollama_runner.platform",
        return_value="linux",
    ):
        assert auto_install_ollama(
            linux_update_cmd=["echo", "'Ollama updated!'"],
            linux_install_cmd=["echo", "'Ollama installed!'"],
            ollama_version=["echo", "'Installed!'"],
        )


def test_auto_install_ollama_on_other_os() -> None:
    with patch(
        "result_companion.core.analizers.local.ollama_runner.platform",
    ) as mock_platform:
        mock_platform.return_value = "other_os"
        with pytest.raises(
            Exception,
            match="Automatic installation is not supported on your OS. Please install Ollama manually.",
        ):
            auto_install_ollama(
                ollama_version=["exit", "1"],
            )


def test_auto_install_ollama_installation_fails() -> None:

    with pytest.raises(
        Exception,
        match="Ollama installation did not complete successfully.",
    ):
        auto_install_ollama(
            brew_installation_cmd=["echo", "'Ollama installing!'"],
            linux_update_cmd=["echo", "'linux updated!'"],
            linux_install_cmd=["echo", "'Ollama installing!'"],
            ollama_version=["exit", "1"],
        )


def test_auto_model_installation_success() -> None:
    assert auto_install_model(
        model_name="my_test_model",
        installation_cmd=["echo", "'Model my_test_model installed!'"],
        ollama_list_cmd=["echo", "'Model my_test_model list!'"],
    )


def test_auto_model_installation_fails_on_final_check() -> None:
    with pytest.raises(
        Exception,
        match="Model 'not_exisitng_model' installation did not complete successfully.",
    ):
        auto_install_model(
            model_name="not_exisitng_model",
            installation_cmd=["echo", "'Model my_test_model installed!'"],
            ollama_list_cmd=["echo", "'my_test_model'"],
        )


def test_auto_model_installation_fails_on_installation_command() -> None:
    with pytest.raises(
        Exception,
        match="Automatic installation of model 'not_exisitng_model' failed. Please install it manually.",
    ):
        auto_install_model(
            model_name="not_exisitng_model",
            installation_cmd=["exit", "1"],
            ollama_list_cmd=["echo", "'my_test_model'"],
        )
