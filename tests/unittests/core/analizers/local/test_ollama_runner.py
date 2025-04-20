import subprocess

import pytest

from result_companion.core.analizers.local.ollama_exceptions import (
    OllamaModelNotAvailable,
    OllamaNotInstalled,
    OllamaServerNotRunning,
)
from result_companion.core.analizers.local.ollama_runner import (
    check_model_installed,
    check_ollama_installed,
    ollama_on_init_strategy,
)


# Dummy implementations for dependency functions.
def dummy_check_ollama_installed(*args, **kwargs):
    # Simulate a successful installation check.
    pass


def dummy_check_model_installed(*args, **kwargs):
    # Simulate that the model is installed.
    pass


def dummy_fail_ollama_installed(*args, **kwargs):
    raise OllamaNotInstalled("Dummy: Ollama not installed.")


def dummy_fail_model_installed(*args, **kwargs):
    raise OllamaModelNotAvailable("Dummy: Model not available.")


# Fake ServerManager classes to simulate different server behaviors.
class FakeServerManagerRunning:
    def __init__(self, *args, **kwargs):
        self.started = False

    def is_running(self, skip_logs=False):
        # Simulate that the server is already running.
        return True

    def start(self):
        self.started = True


class FakeServerManagerNotRunning:
    def __init__(self, *args, **kwargs):
        self.started = False

    def is_running(self, skip_logs=False):
        # Simulate that the server is not running.
        return False

    def start(self):
        # Simulate a successful server start.
        self.started = True


class FakeServerManagerStartFail:
    def __init__(self, *args, **kwargs):
        pass

    def is_running(self, skip_logs=False):
        return False

    def start(self):
        # Simulate failure during the start procedure.
        raise Exception("Server start failed.")


def test_ollama_on_init_success_already_running(monkeypatch):
    monkeypatch.setattr(
        "result_companion.core.analizers.local.ollama_runner.check_ollama_installed",
        dummy_check_ollama_installed,
    )
    monkeypatch.setattr(
        "result_companion.core.analizers.local.ollama_runner.check_model_installed",
        dummy_check_model_installed,
    )

    # Should complete without raising an exception.
    ollama_on_init_strategy(
        "dummy-model", server_manager_class=FakeServerManagerRunning
    )


def test_ollama_on_init_success_not_running(monkeypatch):
    fake_manager = FakeServerManagerNotRunning()
    monkeypatch.setattr(
        "result_companion.core.analizers.local.ollama_runner.check_ollama_installed",
        dummy_check_ollama_installed,
    )
    monkeypatch.setattr(
        "result_companion.core.analizers.local.ollama_runner.check_model_installed",
        dummy_check_model_installed,
    )
    # Override the server manager constructor to return our fake manager.
    monkeypatch.setattr(
        "result_companion.core.analizers.local.ollama_runner.OllamaServerManager",
        lambda *args, **kwargs: fake_manager,
    )

    ollama_on_init_strategy(
        "dummy-model", server_manager_class=FakeServerManagerNotRunning
    )
    # TODO: fix this test to use the actual server manager
    # assert fake_manager.started is True


def test_ollama_on_init_fail_not_installed(monkeypatch):
    monkeypatch.setattr(
        "result_companion.core.analizers.local.ollama_runner.check_ollama_installed",
        dummy_fail_ollama_installed,
    )
    with pytest.raises(OllamaNotInstalled):
        ollama_on_init_strategy("dummy-model")


def test_ollama_on_init_fail_model_not_available(monkeypatch):
    monkeypatch.setattr(
        "result_companion.core.analizers.local.ollama_runner.check_ollama_installed",
        dummy_check_ollama_installed,
    )
    monkeypatch.setattr(
        "result_companion.core.analizers.local.ollama_runner.check_model_installed",
        dummy_fail_model_installed,
    )
    with pytest.raises(OllamaModelNotAvailable):
        ollama_on_init_strategy("dummy-model")


def test_ollama_on_init_fail_server_start(monkeypatch):
    monkeypatch.setattr(
        "result_companion.core.analizers.local.ollama_runner.check_ollama_installed",
        dummy_check_ollama_installed,
    )
    monkeypatch.setattr(
        "result_companion.core.analizers.local.ollama_runner.check_model_installed",
        dummy_check_model_installed,
    )

    with pytest.raises(Exception, match="Server start failed"):
        ollama_on_init_strategy(
            "dummy-model", server_manager_class=FakeServerManagerStartFail
        )


# Dummy output to simulate a successful subprocess.run
class DummyCompletedProcess:
    def __init__(self, stdout):
        self.stdout = stdout
        self.returncode = 0


def test_check_ollama_installed_success(monkeypatch):
    def fake_run(cmd, capture_output, text, check):
        return DummyCompletedProcess(stdout="ollama version 0.6.3\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Should not raise any exception
    check_ollama_installed()


def test_check_ollama_installed_file_not_found():
    with pytest.raises(OllamaNotInstalled, match="Ollama command not found"):
        check_ollama_installed(ollama_version_cmd=["binary_not_existing"])


def test_check_ollama_installed_calledprocesserror(monkeypatch):
    error = subprocess.CalledProcessError(returncode=1, cmd=["ollama", "--version"])

    def fake_run(cmd, capture_output, text, check):
        raise error

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(OllamaNotInstalled, match="Ollama command failed"):
        check_ollama_installed()


def test_check_ollama_installed_generic_exception(monkeypatch):
    def fake_run(cmd, capture_output, text, check):
        raise Exception("Generic error")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(OllamaNotInstalled, match="Failed to check Ollama installation"):
        check_ollama_installed()


class DummyCompletedProcess:
    def __init__(self, stdout):
        self.stdout = stdout


def test_check_model_installed_success(monkeypatch):
    """
    Test that check_model_installed passes when the model is found in the output.
    """
    model_name = "deepseek-r11"
    stdout_output = f"{model_name}: some details\nanother line"

    def fake_run(cmd, capture_output, text, check):
        return DummyCompletedProcess(stdout=stdout_output)

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Should complete without raising an exception.
    check_model_installed(model_name)


def test_check_model_installed_not_found(monkeypatch):
    """
    Test that check_model_installed raises an exception when the model is not present.
    """
    model_name = "deepseek-r11"
    stdout_output = "different-model: details\nanother line"

    def fake_run(cmd, capture_output, text, check):
        return DummyCompletedProcess(stdout=stdout_output)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(
        OllamaModelNotAvailable,
        match=f"Model '{model_name}' is not installed in Ollama",
    ):
        check_model_installed(model_name)


def test_check_model_installed_subprocess_exception(monkeypatch):
    """
    Test that check_model_installed raises an Exception when subprocess.run fails.
    """
    model_name = "deepseek-r11"

    def fake_run(cmd, capture_output, text, check):
        raise subprocess.CalledProcessError(
            returncode=1, cmd=cmd, output="error", stderr="error occurred"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(
        OllamaServerNotRunning,
        match="'ollama list' command failed with error code 1: error occurred",
    ):
        check_model_installed(model_name)
