"""Tests for Copilot LiteLLM adapter."""

import asyncio
import os
import stat
from contextlib import asynccontextmanager

import litellm
import pytest

from result_companion.core.analizers.remote.copilot import (
    CopilotLLM,
    SessionPool,
    ensure_executable,
    messages_to_prompt,
    register_copilot_provider,
)


class TestMessagesToPrompt:
    """Tests for messages_to_prompt function."""

    def test_converts_single_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]

        result = messages_to_prompt(messages)

        assert result == "[User]: Hello"

    def test_converts_multiple_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        result = messages_to_prompt(messages)

        assert "[System]: You are helpful" in result
        assert "[User]: Hello" in result
        assert "[Assistant]: Hi there" in result

    def test_handles_unknown_role(self):
        messages = [{"role": "unknown", "content": "test"}]

        result = messages_to_prompt(messages)

        assert ": test" in result


class TestCopilotLLM:
    """Tests for CopilotLLM class."""

    def test_extract_model_strips_provider_prefix(self):
        handler = CopilotLLM()

        result = handler._extract_model("copilot_sdk/gpt-4.1")

        assert result == "gpt-4.1"

    def test_extract_model_returns_default_for_empty(self):
        handler = CopilotLLM(model="claude-sonnet-4.5")

        result = handler._extract_model("")

        assert result == "claude-sonnet-4.5"

    def test_extract_model_passes_through_without_prefix(self):
        handler = CopilotLLM()

        result = handler._extract_model("gpt-5")

        assert result == "gpt-5"

    @pytest.mark.asyncio
    async def test_ensure_started_initializes_client_and_pool(self, monkeypatch):
        calls = {"ensure_executable": [], "client_opts": None, "start": 0, "pool": 0}

        class FakeClient:
            def __init__(self, opts=None):
                calls["client_opts"] = opts
                self.options = opts or {"cli_path": ""}

            async def start(self):
                calls["start"] += 1

        def fake_ensure_executable(path: str) -> None:
            calls["ensure_executable"].append(path)

        def fake_pool(client, model, pool_size):
            calls["pool"] += 1
            return object()

        monkeypatch.setattr(
            "result_companion.core.analizers.remote.copilot.CopilotClient",
            FakeClient,
        )
        monkeypatch.setattr(
            "result_companion.core.analizers.remote.copilot.ensure_executable",
            fake_ensure_executable,
        )
        monkeypatch.setattr(
            "result_companion.core.analizers.remote.copilot.SessionPool",
            fake_pool,
        )

        handler = CopilotLLM(
            model="gpt-4.1", cli_path="/tmp/copilot", cli_url="http://x"
        )

        await handler._ensure_started("gpt-4.1")

        assert calls["client_opts"] == {
            "cli_path": "/tmp/copilot",
            "cli_url": "http://x",
        }
        assert calls["ensure_executable"] == ["/tmp/copilot"]
        assert calls["start"] == 1
        assert calls["pool"] == 1
        assert handler._started is True

    @pytest.mark.asyncio
    async def test_acompletion_returns_model_response(self, monkeypatch):
        class FakeResponse:
            def __init__(self, content: str):
                self.data = type("Data", (), {"content": content})()

        class FakeSession:
            async def send_and_wait(
                self, options: dict, timeout: int = 300
            ) -> FakeResponse:
                assert options == {"prompt": "[User]: Hello"}
                assert timeout == 10
                return FakeResponse("hi")

        class FakePool:
            @asynccontextmanager
            async def acquire(self):
                yield FakeSession()

        handler = CopilotLLM(timeout=10)

        async def fake_ensure_started(model: str) -> None:
            handler._pool = FakePool()

        monkeypatch.setattr(handler, "_ensure_started", fake_ensure_started)

        result = await handler.acompletion(
            "copilot_sdk/gpt-4.1", [{"role": "user", "content": "Hello"}]
        )

        assert result.model == "gpt-4.1"
        assert result.choices[0]["message"]["content"] == "hi"

    def test_register_copilot_provider_sets_custom_provider_map(self, monkeypatch):
        monkeypatch.setattr(litellm, "custom_provider_map", [])

        handler = register_copilot_provider(model="gpt-4.1", pool_size=1, timeout=10)

        assert isinstance(handler, CopilotLLM)
        assert litellm.custom_provider_map == [
            {"provider": "copilot_sdk", "custom_handler": handler}
        ]

    def test_completion_wraps_async(self, monkeypatch):
        handler = CopilotLLM()
        expected = object()

        async def fake_acompletion(
            model: str, messages: list[dict[str, str]], **kwargs
        ) -> object:
            return expected

        monkeypatch.setattr(handler, "acompletion", fake_acompletion)

        result = handler.completion("gpt-4.1", [{"role": "user", "content": "hi"}])

        assert result is expected


class TestEnsureExecutable:
    """Tests for ensure_executable helper."""

    def test_sets_execute_bits_on_regular_file(self, tmp_path):
        # Create a non-executable file and confirm we add execute bits.
        file_path = tmp_path / "copilot"
        file_path.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
        os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        assert not os.access(file_path, os.X_OK)

        ensure_executable(str(file_path))

        assert os.access(file_path, os.X_OK)

    def test_skips_relative_path(self, tmp_path):
        # Relative paths should be ignored to avoid mutating unknown locations.
        file_path = tmp_path / "copilot"
        file_path.write_text("echo ok\n", encoding="utf-8")
        os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)
        original_mode = os.stat(file_path).st_mode

        ensure_executable("copilot")

        assert os.stat(file_path).st_mode == original_mode

    def test_skips_when_already_executable(self, tmp_path):
        # If the file is already executable, the mode should remain unchanged.
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


class FakeSession:
    """Fake Copilot session for testing."""

    def __init__(self, response_content: str = "test response"):
        self.response_content = response_content
        self.destroyed = False

    async def send_and_wait(self, options: dict, timeout: int = 300) -> object:
        """Returns fake response."""

        class FakeData:
            content = self.response_content

        class FakeResponse:
            data = FakeData()

        return FakeResponse()

    async def destroy(self) -> None:
        self.destroyed = True


class FakeCopilotClient:
    """Fake Copilot client for testing."""

    def __init__(self, response_content: str = "test response"):
        self.response_content = response_content
        self.sessions_created = 0

    async def create_session(self, config: dict) -> FakeSession:
        self.sessions_created += 1
        return FakeSession(self.response_content)


class TestSessionPool:
    """Tests for SessionPool class."""

    @pytest.mark.asyncio
    async def test_acquire_creates_session_when_pool_empty(self):
        client = FakeCopilotClient()
        pool = SessionPool(client, "gpt-4.1", pool_size=2)

        async with pool.acquire() as session:
            assert session is not None

        assert client.sessions_created == 1

    @pytest.mark.asyncio
    async def test_acquire_reuses_session(self):
        client = FakeCopilotClient()
        pool = SessionPool(client, "gpt-4.1", pool_size=2)

        async with pool.acquire():
            pass
        async with pool.acquire():
            pass

        assert client.sessions_created == 1

    @pytest.mark.asyncio
    async def test_acquire_waits_when_demand_exceeds_pool_size(self):
        client = FakeCopilotClient()
        pool = SessionPool(client, "gpt-4.1", pool_size=2)
        first_released = asyncio.Event()
        second_released = asyncio.Event()

        async def acquire_and_hold(release_event: asyncio.Event) -> None:
            async with pool.acquire():
                await release_event.wait()

        task_one = asyncio.create_task(acquire_and_hold(first_released))
        task_two = asyncio.create_task(acquire_and_hold(second_released))

        await asyncio.sleep(0)

        assert client.sessions_created == 2

        third_done = asyncio.Event()

        async def acquire_third() -> None:
            async with pool.acquire():
                third_done.set()

        task_three = asyncio.create_task(acquire_third())

        await asyncio.sleep(0)

        assert third_done.is_set() is False
        assert client.sessions_created == 2

        first_released.set()
        await third_done.wait()

        second_released.set()

        await task_one
        await task_two
        await task_three

    @pytest.mark.asyncio
    async def test_close_destroys_sessions(self):
        client = FakeCopilotClient()
        pool = SessionPool(client, "gpt-4.1", pool_size=2)

        async with pool.acquire():
            pass

        await pool.close()

        assert pool._created == 0
