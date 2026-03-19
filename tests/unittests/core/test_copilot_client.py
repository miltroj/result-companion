"""Tests for shared Copilot client helpers."""

import asyncio
import os
import stat

import pytest

from result_companion.core.copilot_client import (
    build_copilot_client_options,
    ensure_executable,
    resolve_copilot_cli_path,
    start_copilot_client,
    stop_copilot_client,
)


class FakeClient:
    """Simple fake Copilot client."""

    def __init__(self, cli_path: str = ""):
        self.options = {"cli_path": cli_path}
        self.stopped = False

    async def start(self) -> None:
        """Starts the fake client."""

    async def stop(self) -> None:
        """Stops the fake client."""
        self.stopped = True

    async def get_auth_status(self):
        """Returns an authenticated fake user."""
        return type("Auth", (), {"isAuthenticated": True, "login": "octocat"})()

    async def get_status(self):
        """Returns fake status data."""
        return type("Status", (), {"version": "1.0", "protocolVersion": 1})()

    async def list_models(self):
        """Returns fake model data."""
        return [type("Model", (), {"id": "gpt-5-mini"})()]


class TestResolveCopilotCliPath:
    """Tests for CLI path resolution."""

    def test_uses_environment_variable(self, monkeypatch):
        monkeypatch.setenv("COPILOT_CLI_PATH", "copilot")
        monkeypatch.setattr(
            "result_companion.core.copilot_client.shutil.which",
            lambda _: "/resolved/copilot",
        )

        result = resolve_copilot_cli_path()

        assert result == "/resolved/copilot"

    def test_builds_client_options(self, monkeypatch):
        monkeypatch.setattr(
            "result_companion.core.copilot_client.shutil.which",
            lambda _: "/resolved/copilot",
        )

        result = build_copilot_client_options(
            cli_path="copilot",
            cli_url="http://localhost:4141",
        )

        assert result == {
            "cli_path": "copilot",
            "cli_url": "http://localhost:4141",
        }


class TestEnsureExecutable:
    """Tests for executable permission fix-up."""

    def test_sets_execute_bits_on_regular_file(self, tmp_path):
        file_path = tmp_path / "copilot"
        file_path.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
        os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

        assert not os.access(file_path, os.X_OK)

        ensure_executable(str(file_path))

        assert os.access(file_path, os.X_OK)

    def test_skips_relative_path(self, tmp_path):
        file_path = tmp_path / "copilot"
        file_path.write_text("echo ok\n", encoding="utf-8")
        os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)
        original_mode = os.stat(file_path).st_mode

        ensure_executable("copilot")

        assert os.stat(file_path).st_mode == original_mode

    def test_skips_when_already_executable(self, tmp_path):
        file_path = tmp_path / "copilot"
        file_path.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
        os.chmod(
            file_path,
            stat.S_IRUSR
            | stat.S_IWUSR
            | stat.S_IXUSR
            | stat.S_IRGRP
            | stat.S_IXGRP
            | stat.S_IROTH
            | stat.S_IXOTH,
        )
        original_mode = os.stat(file_path).st_mode

        ensure_executable(str(file_path))

        assert os.stat(file_path).st_mode == original_mode


class TestStartCopilotClient:
    """Tests for startup helpers."""

    @pytest.mark.asyncio
    async def test_raises_on_startup_timeout(self, monkeypatch):
        class HangingClient(FakeClient):
            async def start(self) -> None:
                await asyncio.Event().wait()

        monkeypatch.setattr(
            "result_companion.core.copilot_client.ensure_executable",
            lambda _: None,
        )

        with pytest.raises(RuntimeError, match="failed to start"):
            await start_copilot_client(HangingClient(), startup_timeout=0.1)

    @pytest.mark.asyncio
    async def test_stops_client_when_authentication_fails(self, monkeypatch):
        class UnauthClient(FakeClient):
            async def get_auth_status(self):
                return type("Auth", (), {"isAuthenticated": False, "login": None})()

        monkeypatch.setattr(
            "result_companion.core.copilot_client.ensure_executable",
            lambda _: None,
        )

        client = UnauthClient()

        with pytest.raises(RuntimeError, match="not authenticated"):
            await start_copilot_client(client, startup_timeout=0.1)

        assert client.stopped is True

    @pytest.mark.asyncio
    async def test_stop_suppresses_shutdown_errors(self):
        class ExplodingClient(FakeClient):
            async def stop(self) -> None:
                raise RuntimeError("boom")

        await stop_copilot_client(ExplodingClient())
