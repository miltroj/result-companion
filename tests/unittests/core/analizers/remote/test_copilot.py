"""Tests for Copilot SDK LangChain adapter."""

import asyncio
from dataclasses import dataclass

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from result_companion.core.analizers.remote.copilot import (
    ChatCopilot,
    SessionPool,
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
        assert messages_to_prompt([]) == ""


@dataclass
class FakeResponseData:
    content: str


@dataclass
class FakeResponse:
    data: FakeResponseData


class FakeCopilotSession:
    """Fake Copilot session for testing."""

    def __init__(self, response_content: str = "Test response", session_id: int = 0):
        self.response_content = response_content
        self.session_id = session_id
        self.prompts_received: list[str] = []

    async def send_and_wait(self, options: dict, timeout: int = 300) -> FakeResponse:
        self.prompts_received.append(options.get("prompt", ""))
        return FakeResponse(data=FakeResponseData(content=self.response_content))

    async def destroy(self) -> None:
        pass


class FakeCopilotClient:
    """Fake Copilot client for testing."""

    def __init__(self, response_content: str = "Test response"):
        self.started = False
        self.stopped = False
        self.sessions_created = 0
        self.response_content = response_content

    async def start(self) -> None:
        self.started = True

    async def create_session(self, options: dict) -> FakeCopilotSession:
        self.sessions_created += 1
        return FakeCopilotSession(
            session_id=self.sessions_created, response_content=self.response_content
        )

    async def stop(self) -> None:
        self.stopped = True


class TestSessionPool:
    """Tests for SessionPool."""

    @pytest.mark.asyncio
    async def test_acquire_creates_session_on_demand(self):
        client = FakeCopilotClient()
        pool = SessionPool(client, "gpt-4.1", pool_size=3)

        async with pool.acquire() as session:
            assert session is not None
            assert client.sessions_created == 1

    @pytest.mark.asyncio
    async def test_acquire_reuses_released_session(self):
        client = FakeCopilotClient()
        pool = SessionPool(client, "gpt-4.1", pool_size=3)

        async with pool.acquire() as session1:
            first_id = session1.session_id

        async with pool.acquire() as session2:
            assert session2.session_id == first_id
            assert client.sessions_created == 1  # No new session created

    @pytest.mark.asyncio
    async def test_blocked_acquire_gets_released_session(self):
        """Verifies waiting acquire gets session when another releases."""
        client = FakeCopilotClient()
        pool = SessionPool(client, "gpt-4.1", pool_size=1)  # Only 1 session

        acquired_ids: list[int] = []

        async def acquire_and_record():
            async with pool.acquire() as session:
                acquired_ids.append(session.session_id)
                await asyncio.sleep(0.05)  # Hold briefly

        # Start two tasks competing for 1 session
        task1 = asyncio.create_task(acquire_and_record())
        await asyncio.sleep(0.01)  # Let task1 acquire first
        task2 = asyncio.create_task(acquire_and_record())

        await asyncio.gather(task1, task2)

        # Both should have used the SAME session (reused after release)
        assert len(acquired_ids) == 2
        assert acquired_ids[0] == acquired_ids[1]
        assert client.sessions_created == 1

    @pytest.mark.asyncio
    async def test_session_released_on_error(self):
        """Verifies session returns to pool even if request fails."""
        client = FakeCopilotClient()
        pool = SessionPool(client, "gpt-4.1", pool_size=1)

        # First acquire fails with exception
        with pytest.raises(RuntimeError):
            async with pool.acquire() as session:
                raise RuntimeError("API failed")

        # Session should still be available for next acquire
        async with pool.acquire() as session:
            assert session is not None
            assert client.sessions_created == 1  # Same session reused

    @pytest.mark.asyncio
    async def test_pool_limits_concurrent_sessions(self):
        client = FakeCopilotClient()
        pool = SessionPool(client, "gpt-4.1", pool_size=2)

        # Acquire all sessions in pool
        async with pool.acquire():
            async with pool.acquire():
                assert client.sessions_created == 2

                # Third acquire should block (pool exhausted)
                acquire_started = asyncio.Event()
                acquired = asyncio.Event()

                async def blocked_acquire():
                    acquire_started.set()
                    async with pool.acquire():
                        acquired.set()

                task = asyncio.create_task(blocked_acquire())
                await acquire_started.wait()
                await asyncio.sleep(0.05)
                assert not acquired.is_set()  # Still waiting
                task.cancel()

    @pytest.mark.asyncio
    async def test_close_destroys_available_sessions(self):
        client = FakeCopilotClient()
        pool = SessionPool(client, "gpt-4.1", pool_size=2)

        async with pool.acquire():
            pass  # Session now in available queue

        assert pool._created == 1
        await pool.close()
        assert pool._created == 0
        assert pool._available.empty()


class TestChatCopilot:
    """Tests for ChatCopilot adapter."""

    @pytest.mark.asyncio
    async def test_agenerate_returns_ai_message_with_response(self):
        fake_client = FakeCopilotClient(response_content="4")
        chat = ChatCopilot(model="gpt-4.1", pool_size=1)
        chat._client = fake_client
        chat._pool = SessionPool(fake_client, "gpt-4.1", pool_size=1)
        chat._started = True

        result = await chat._agenerate([HumanMessage(content="What is 2+2?")])

        assert result.generations[0].message.content == "4"

    @pytest.mark.asyncio
    async def test_aclose_cleans_up_resources(self):
        fake_client = FakeCopilotClient()
        chat = ChatCopilot(model="gpt-4.1")
        chat._client = fake_client
        chat._pool = SessionPool(fake_client, "gpt-4.1", pool_size=1)
        chat._started = True

        await chat.aclose()

        assert fake_client.stopped
        assert chat._pool is None
        assert chat._client is None
        assert not chat._started

    @pytest.mark.asyncio
    async def test_concurrent_requests_use_separate_sessions(self):
        """Verifies pool enables true concurrency with separate sessions."""
        sessions_used: list[int] = []

        class TrackingClient(FakeCopilotClient):
            async def create_session(self, options: dict) -> FakeCopilotSession:
                self.sessions_created += 1
                session = FakeCopilotSession(session_id=self.sessions_created)
                original_send = session.send_and_wait

                async def tracking_send(opts: dict, timeout: int = 300) -> FakeResponse:
                    sessions_used.append(session.session_id)
                    await asyncio.sleep(0.05)
                    return await original_send(opts, timeout)

                session.send_and_wait = tracking_send
                return session

        fake_client = TrackingClient()
        chat = ChatCopilot(model="gpt-4.1", pool_size=3)
        chat._client = fake_client
        chat._pool = SessionPool(fake_client, "gpt-4.1", pool_size=3)
        chat._started = True

        # Launch 3 concurrent requests
        tasks = [
            chat._agenerate([HumanMessage(content="Request 1")]),
            chat._agenerate([HumanMessage(content="Request 2")]),
            chat._agenerate([HumanMessage(content="Request 3")]),
        ]
        await asyncio.gather(*tasks)

        # With pool, each request should use a different session
        assert len(set(sessions_used)) == 3  # 3 unique sessions
