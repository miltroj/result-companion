"""Tests for shared Copilot client helpers."""

import asyncio

import pytest

from result_companion.core.copilot_client import (
    build_copilot_client_options,
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
