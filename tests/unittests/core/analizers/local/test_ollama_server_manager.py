import subprocess
import time
from unittest.mock import patch

import pytest
import requests

from result_companion.core.analizers.local.ollama_exceptions import (
    OllamaNotInstalled,
    OllamaServerNotRunning,
)
from result_companion.core.analizers.local.ollama_server_manager import (
    OllamaServerManager,
    resolve_server_manager,
)


# Test helpers
class DummyProcess:
    """Simulates subprocess.Popen behavior for testing."""

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


class DummyResponse:
    """Simulates requests.Response for testing."""

    def __init__(self, status_code=200, text="Ollama is running"):
        self.status_code = status_code
        self.text = text


# Fixtures
@pytest.fixture
def server_url():
    """Standard server URL used in tests."""
    return "http://localhost:11434"


@pytest.fixture
def running_server(monkeypatch):
    """Mocks a running Ollama server."""
    monkeypatch.setattr(
        requests, "get", lambda url, timeout: DummyResponse(status_code=200)
    )


@pytest.fixture
def stopped_server(monkeypatch):
    """Mocks a stopped Ollama server."""
    monkeypatch.setattr(
        requests,
        "get",
        lambda url, timeout: pytest.raises(
            requests.exceptions.RequestException("Not running")
        ),
    )


@pytest.fixture
def dummy_process():
    """Returns a dummy process for testing."""
    return DummyProcess(pid=12345)


@pytest.fixture
def mock_popen(monkeypatch, dummy_process):
    """Mocks subprocess.Popen to return a dummy process."""
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, stdout, stderr: dummy_process)
    return dummy_process


class TestServerStatus:
    """Tests for server status checking functionality."""

    def test_is_running_true(self, monkeypatch, server_url):
        """Test that is_running returns True when server responds successfully."""
        monkeypatch.setattr(
            requests, "get", lambda url, timeout: DummyResponse(status_code=200)
        )

        manager = OllamaServerManager(server_url=server_url)
        assert manager.is_running() is True

    def test_is_running_false(self, monkeypatch, server_url):
        """Test that is_running returns False when server request fails."""
        monkeypatch.setattr(
            requests,
            "get",
            lambda url, timeout: exec(
                "raise requests.exceptions.RequestException('Server not reachable')"
            ),
        )

        manager = OllamaServerManager(server_url=server_url)
        assert manager.is_running() is False


class TestServerStartup:
    """Tests for server startup functionality."""

    def test_start_already_running(self, monkeypatch, server_url, mock_popen):
        """Test that start() does nothing when server is already running."""
        monkeypatch.setattr(
            requests, "get", lambda url, timeout: DummyResponse(status_code=200)
        )

        manager = OllamaServerManager(server_url=server_url)
        manager.start()

        # Should not start a new process if already running
        assert manager._process is None

    def test_start_not_installed(self, monkeypatch, server_url):
        """Test that start() raises OllamaNotInstalled when executable is not found."""
        # Server is not running
        monkeypatch.setattr(
            requests,
            "get",
            lambda url, timeout: exec(
                "raise requests.exceptions.RequestException('Not running')"
            ),
        )

        # Command not found
        monkeypatch.setattr(
            subprocess,
            "Popen",
            lambda cmd, stdout, stderr: exec(
                "raise FileNotFoundError('Command not found')"
            ),
        )

        manager = OllamaServerManager(server_url=server_url)
        with pytest.raises(OllamaNotInstalled):
            manager.start()

    def test_start_timeout(self, monkeypatch, server_url, mock_popen):
        """Test that start() raises OllamaServerNotRunning when server doesn't start in time."""
        # Server never starts
        monkeypatch.setattr(
            requests,
            "get",
            lambda url, timeout: exec(
                "raise requests.exceptions.RequestException('Not running')"
            ),
        )

        manager = OllamaServerManager(
            server_url=server_url, start_timeout=0.1, wait_for_start=0.01
        )

        with pytest.raises(OllamaServerNotRunning):
            manager.start()

        # Should clean up after timeout
        assert manager._process is None


class TestCleanup:
    """Tests for cleanup functionality."""

    def test_cleanup_terminates_process(self, server_url, dummy_process):
        """Test that cleanup terminates a running process."""
        manager = OllamaServerManager(server_url=server_url)
        manager._process = dummy_process

        manager.cleanup()

        assert dummy_process.terminated or dummy_process.killed
        assert manager._process is None


class TestContextManager:
    """Tests for context manager functionality."""

    @patch.object(OllamaServerManager, "start")
    def test_enter_calls_start(self, mock_start, server_url):
        """Test that __enter__ calls start() and returns self."""
        manager = OllamaServerManager(server_url=server_url)

        with manager as ctx:
            mock_start.assert_called_once()
            assert ctx is manager

    @patch.object(OllamaServerManager, "cleanup")
    @patch.object(OllamaServerManager, "start")
    def test_exit_calls_cleanup(
        self, mock_cleanup, mock_start, server_url, monkeypatch, dummy_process
    ):
        """Test that __exit__ calls cleanup()."""
        manager = OllamaServerManager(server_url=server_url)

        def mock_popen(cmd, stdout, stderr):
            print(f"mocking Popen with cmd: {cmd}")
            return dummy_process

        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        with manager:
            pass

        mock_cleanup.assert_called_once()
        mock_start.assert_called_once()

    def test_complete_lifecycle(self, monkeypatch, server_url, dummy_process):
        """Test the complete context manager lifecycle."""
        # First not running, then running after start
        call_count = [0]

        def mock_get(url, timeout):
            call_count[0] += 1
            # Simulate server starting after 3 is_running calls
            if call_count[0] == 3:
                return DummyResponse(status_code=200)
            raise requests.exceptions.RequestException("Not running")

        def mock_popen(cmd, stdout, stderr):
            print(f"mocking Popen with cmd: {cmd}")
            return dummy_process

        monkeypatch.setattr(requests, "get", mock_get)
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        with OllamaServerManager(server_url=server_url, wait_for_start=0.01) as manager:
            print(f"call_count: {call_count}")
            assert manager._process is dummy_process
            assert not dummy_process.terminated

        # After context exit
        assert dummy_process.terminated or dummy_process.killed
        assert manager._process is None


class TestResolveServerManager:
    """Tests for resolve_server_manager function."""

    def test_resolve_with_class(self):
        """Test resolving with a class."""
        result = resolve_server_manager(OllamaServerManager)
        assert isinstance(result, OllamaServerManager)
        assert result.server_url == "http://localhost:11434"

    def test_resolve_with_none(self):
        """Test resolving with None."""
        result = resolve_server_manager(None)
        assert isinstance(result, OllamaServerManager)
        assert result.server_url == "http://localhost:11434"

    def test_resolve_with_instance(self):
        """Test resolving with an existing instance."""
        instance = OllamaServerManager(server_url="http://custom:8080")
        result = resolve_server_manager(instance)
        assert result is instance  # Should return the same instance
        assert result.server_url == "http://custom:8080"

    def test_resolve_with_kwargs(self):
        """Test resolving with custom kwargs."""
        result = resolve_server_manager(None, server_url="http://custom:9000")
        assert isinstance(result, OllamaServerManager)
        assert result.server_url == "http://custom:9000"

        # Test kwargs with a class
        result = resolve_server_manager(OllamaServerManager, start_timeout=60)
        assert isinstance(result, OllamaServerManager)
        assert result.start_timeout == 60
