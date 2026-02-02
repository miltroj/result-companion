"""Tests for Copilot LiteLLM adapter."""

import pytest

from result_companion.core.analizers.remote.copilot import (
    CopilotLLM,
    SessionPool,
    messages_to_prompt,
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
    async def test_close_destroys_sessions(self):
        client = FakeCopilotClient()
        pool = SessionPool(client, "gpt-4.1", pool_size=2)

        async with pool.acquire():
            pass

        await pool.close()

        assert pool._created == 0
