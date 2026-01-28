"""Tests for Copilot SDK LangChain adapter."""

from dataclasses import dataclass
from typing import Optional

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from result_companion.core.analizers.remote.copilot import (
    ChatCopilot,
    messages_to_prompt,
)


class TestMessagesToPrompt:
    """Tests for messages_to_prompt pure function."""

    def test_human_message_only(self):
        messages = [HumanMessage(content="Hello")]

        result = messages_to_prompt(messages)

        assert result == "[User]: Hello"

    def test_system_and_human_messages(self):
        messages = [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="What is 2+2?"),
        ]

        result = messages_to_prompt(messages)

        assert "[System]: You are helpful" in result
        assert "[User]: What is 2+2?" in result

    def test_multiple_messages_joined_with_double_newline(self):
        messages = [
            SystemMessage(content="System"),
            HumanMessage(content="User"),
            AIMessage(content="Assistant"),
        ]

        result = messages_to_prompt(messages)

        assert result == "[System]: System\n\n[User]: User\n\n[Assistant]: Assistant"

    def test_empty_messages_list(self):
        result = messages_to_prompt([])

        assert result == ""


@dataclass
class FakeResponseData:
    """Fake response data from Copilot SDK."""

    content: str


@dataclass
class FakeResponse:
    """Fake response from Copilot SDK."""

    data: FakeResponseData


class FakeCopilotSession:
    """Fake Copilot session for testing."""

    def __init__(self, response_content: str = "Test response"):
        self.response_content = response_content
        self.prompts_received: list[str] = []

    async def send_and_wait(self, options: dict) -> FakeResponse:
        """Records prompt and returns fake response."""
        self.prompts_received.append(options.get("prompt", ""))
        return FakeResponse(data=FakeResponseData(content=self.response_content))

    async def destroy(self) -> None:
        """Cleanup - no-op for fake."""
        pass


class FakeCopilotClient:
    """Fake Copilot client for testing."""

    def __init__(self, session: Optional[FakeCopilotSession] = None):
        self.session = session or FakeCopilotSession()
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        """Marks client as started."""
        self.started = True

    async def create_session(self, options: dict) -> FakeCopilotSession:
        """Returns the fake session."""
        return self.session

    async def stop(self) -> None:
        """Marks client as stopped."""
        self.stopped = True


class TestChatCopilot:
    """Tests for ChatCopilot adapter."""

    @pytest.mark.asyncio
    async def test_agenerate_returns_ai_message_with_response(self):
        fake_session = FakeCopilotSession(response_content="42")
        chat = ChatCopilot(model="gpt-4.1")
        chat._session = fake_session
        chat._started = True
        messages = [HumanMessage(content="What is 2+2?")]

        result = await chat._agenerate(messages)

        assert result.generations[0].message.content == "42"
        assert len(fake_session.prompts_received) == 1
        assert "[User]: What is 2+2?" in fake_session.prompts_received[0]

    @pytest.mark.asyncio
    async def test_agenerate_converts_messages_to_prompt(self):
        fake_session = FakeCopilotSession()
        chat = ChatCopilot(model="gpt-4.1")
        chat._session = fake_session
        chat._started = True
        messages = [
            SystemMessage(content="Be concise"),
            HumanMessage(content="Hello"),
        ]

        await chat._agenerate(messages)

        prompt = fake_session.prompts_received[0]
        assert "[System]: Be concise" in prompt
        assert "[User]: Hello" in prompt

    @pytest.mark.asyncio
    async def test_aclose_cleans_up_resources(self):
        fake_session = FakeCopilotSession()
        fake_client = FakeCopilotClient(session=fake_session)
        chat = ChatCopilot(model="gpt-4.1")
        chat._client = fake_client
        chat._session = fake_session
        chat._started = True

        await chat.aclose()

        assert fake_client.stopped
        assert chat._session is None
        assert chat._client is None
        assert not chat._started
