import subprocess
import time

import pytest
import requests

from result_companion.core.analizers.local.ollama_exceptions import (
    OllamaNotInstalled,
    OllamaServerNotRunning,
)
from result_companion.core.analizers.local.ollama_server_manager import (
    OllamaServerManager,
)


# A dummy process to simulate subprocess.Popen behavior in tests.
class DummyProcess:
    def __init__(self, pid=1234):
        self.pid = pid
        self.terminated = False
        self.killed = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout):
        time.sleep(0.1)  # Simulate slight delay
        if self.terminated:
            return
        raise subprocess.TimeoutExpired(cmd="dummy", timeout=timeout)

    def kill(self):
        self.killed = True


# Helper dummy response for requests
class DummyResponse:
    def __init__(self, status_code=200, text="Ollama is running"):
        self.status_code = status_code
        self.text = text


def test_is_running_true(monkeypatch):
    # Monkey-patch requests.get to simulate a running server.
    def fake_get(url, timeout):
        return DummyResponse(status_code=200, text="Ollama is running")

    monkeypatch.setattr(requests, "get", fake_get)

    manager = OllamaServerManager(server_url="http://localhost:11434")
    assert manager.is_running() is True


def test_is_running_false(monkeypatch):
    # Monkey-patch requests.get to simulate a non-running server.
    def fake_get(url, timeout):
        raise requests.exceptions.RequestException("Server not reachable")

    monkeypatch.setattr(requests, "get", fake_get)

    manager = OllamaServerManager(server_url="http://localhost:11434")
    assert manager.is_running() is False


def test_start_already_running(monkeypatch):
    # Simulate that the server is running so start() should do nothing.
    def fake_get(url, timeout):
        return DummyResponse(status_code=200, text="Ollama is running")

    monkeypatch.setattr(requests, "get", fake_get)

    # Also, simulate a Popen so that if it were called, would create a DummyProcess.
    def fake_popen(cmd, stdout, stderr):
        return DummyProcess(pid=9999)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    manager = OllamaServerManager(server_url="http://localhost:11434")
    # Call start; since is_running is True, it should not start a new process.
    manager.start()
    # _process should still be None
    assert manager._process is None


def test_start_not_installed(monkeypatch):
    # Simulate FileNotFoundError when Popen is called.
    def fake_popen(cmd, stdout, stderr):
        raise FileNotFoundError("Command not found")

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    manager = OllamaServerManager(server_url="http://localhost:11434")
    with pytest.raises(OllamaNotInstalled):
        manager.start()


def test_start_timeout(monkeypatch):
    # Simulate a scenario where the server doesn't come up (always returns not running)
    def fake_get(url, timeout):
        raise requests.exceptions.RequestException("Server not reachable")

    monkeypatch.setattr(requests, "get", fake_get)

    # Simulate a normal Popen returning a dummy process.
    dummy_proc = DummyProcess(pid=1111)

    def fake_popen(cmd, stdout, stderr):
        return dummy_proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    manager = OllamaServerManager(
        server_url="http://localhost:11434", start_timeout=0.1, wait_for_start=0.01
    )
    with pytest.raises(OllamaServerNotRunning):
        manager.start()
    # After timeout, cleanup should have been called so _process is None.
    assert manager._process is None


def test_cleanup(monkeypatch):
    # Check that cleanup terminates a running process.
    dummy_proc = DummyProcess(pid=2222)
    manager = OllamaServerManager(server_url="http://localhost:11434")
    manager._process = dummy_proc
    manager.cleanup()
    # Depending on simulated wait, either terminated or killed should be True.
    assert dummy_proc.terminated or dummy_proc.killed
    assert manager._process is None
