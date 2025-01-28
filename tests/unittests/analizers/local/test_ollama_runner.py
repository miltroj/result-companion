from result_companion.analizers.local.ollama_runner import check_ollama_installed, start_ollama_server
import sys
import pytest


@pytest.mark.skipif(sys.platform == "win32", reason="Test not applicable on Windows")
def test_ollama_is_installed() -> None:
    assert check_ollama_installed(ollama_version=["echo", "'Installed!'"]) is None


@pytest.mark.skipif(sys.platform == "win32", reason="Test not applicable on Windows")
def test_starting_ollama_server() -> None:
    assert start_ollama_server(start_cmd=["echo", "'Starting server!'"]) is None
