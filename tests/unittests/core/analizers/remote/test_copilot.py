"""Tests for Copilot LiteLLM adapter."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any

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


class FakeStartableClient:
    """Fake client that passes startup checks (auth, status, models)."""

    def __init__(self, opts=None):
        self.options = opts or {"cli_path": ""}

    async def start(self):
        pass

    async def get_auth_status(self):
        return type("R", (), {"isAuthenticated": True, "login": "user"})()

    async def get_status(self):
        return type("R", (), {"version": "1.0", "protocolVersion": 1})()

    async def list_models(self):
        return [type("M", (), {"id": "gpt-4.1"})()]


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

        class TrackingClient(FakeStartableClient):
            def __init__(self, opts=None):
                super().__init__(opts)
                calls["client_opts"] = opts

            async def start(self):
                calls["start"] += 1

        def fake_ensure_executable(path: str) -> None:
            calls["ensure_executable"].append(path)

        def fake_pool(client, model, pool_size):
            calls["pool"] += 1
            return object()

        monkeypatch.setattr(
            "result_companion.core.analizers.remote.copilot.CopilotClient",
            TrackingClient,
        )
        monkeypatch.setattr(
            "result_companion.core.copilot_client.ensure_executable",
            fake_ensure_executable,
        )
        monkeypatch.setattr(
            "result_companion.core.analizers.remote.copilot.SessionPool",
            fake_pool,
        )
        monkeypatch.setattr(
            "result_companion.core.copilot_client.shutil.which",
            lambda _: "/tmp/copilot",
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
    async def test_ensure_started_raises_on_startup_timeout(self, monkeypatch):
        class HangingClient:
            def __init__(self, opts=None):
                self.options = opts or {"cli_path": ""}

            async def start(self):
                await asyncio.Event().wait()

        monkeypatch.setattr(
            "result_companion.core.analizers.remote.copilot.CopilotClient",
            HangingClient,
        )
        monkeypatch.setattr(
            "result_companion.core.copilot_client.ensure_executable",
            lambda _: None,
        )

        handler = CopilotLLM(startup_timeout=0.1)

        with pytest.raises(RuntimeError, match="failed to start"):
            await handler._ensure_started("gpt-4.1")

        assert handler._started is False

    @pytest.mark.asyncio
    async def test_ensure_started_raises_when_not_authenticated(self, monkeypatch):
        class UnauthClient:
            def __init__(self, opts=None):
                self.options = opts or {"cli_path": ""}
                self.stopped = False

            async def start(self):
                pass

            async def get_auth_status(self):
                return type("R", (), {"isAuthenticated": False, "login": None})()

            async def stop(self):
                self.stopped = True

        monkeypatch.setattr(
            "result_companion.core.analizers.remote.copilot.CopilotClient",
            UnauthClient,
        )
        monkeypatch.setattr(
            "result_companion.core.copilot_client.ensure_executable",
            lambda _: None,
        )

        handler = CopilotLLM()

        with pytest.raises(RuntimeError, match="not authenticated"):
            await handler._ensure_started("gpt-4.1")

        assert handler._started is False
        assert handler._client is None

    @pytest.mark.asyncio
    async def test_ensure_started_uses_resolved_cli_when_no_explicit_path(
        self, monkeypatch
    ):
        captured_opts = {}

        class CapturingClient(FakeStartableClient):
            def __init__(self, opts=None):
                super().__init__(opts)
                captured_opts.update(opts or {})

        monkeypatch.setattr(
            "result_companion.core.analizers.remote.copilot.CopilotClient",
            CapturingClient,
        )
        monkeypatch.setattr(
            "result_companion.core.copilot_client.ensure_executable",
            lambda _: None,
        )
        monkeypatch.setattr(
            "result_companion.core.analizers.remote.copilot.SessionPool",
            lambda *a: object(),
        )
        monkeypatch.setenv("COPILOT_CLI_PATH", "copilot")
        monkeypatch.setattr(
            "result_companion.core.copilot_client.shutil.which",
            lambda _: "/resolved/copilot",
        )

        handler = CopilotLLM()
        await handler._ensure_started("gpt-4.1")

        assert captured_opts["cli_path"] == "/resolved/copilot"

    @pytest.mark.asyncio
    async def test_ensure_started_raises_when_cli_missing(self, monkeypatch):
        handler = CopilotLLM()
        monkeypatch.setattr(
            "result_companion.core.copilot_client.shutil.which",
            lambda _: None,
        )
        monkeypatch.setenv("COPILOT_CLI_PATH", "/missing/copilot")

        with pytest.raises(FileNotFoundError):
            await handler._ensure_started("gpt-4.1")

    @pytest.mark.asyncio
    async def test_stop_client_suppresses_errors(self):
        class ExplodingClient:
            async def stop(self):
                raise RuntimeError("stop failed")

        handler = CopilotLLM()
        await handler._stop_client(ExplodingClient())

    @pytest.mark.asyncio
    async def test_log_diagnostics_suppresses_errors(self):
        class BrokenClient:
            async def get_status(self):
                raise ValueError("Missing required field 'vision'")

        handler = CopilotLLM()
        await handler._log_diagnostics(BrokenClient())

    @pytest.mark.asyncio
    async def test_aclose_stops_client_and_resets_state(self):
        class FakeClient:
            def __init__(self):
                self.stopped = False

            async def stop(self):
                self.stopped = True

        class FakePool:
            def __init__(self):
                self.closed = False

            async def close(self):
                self.closed = True

        handler = CopilotLLM()
        handler._client = FakeClient()
        handler._pool = FakePool()
        handler._started = True

        await handler.aclose()

        assert handler._client is None
        assert handler._pool is None
        assert handler._started is False

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


class TestRateLimitDetection:
    """Tests for _is_rate_limit_error."""

    def test_detects_copilot_rate_limit_message(self):
        handler = CopilotLLM()
        error = Exception(
            "Sorry, you've hit a rate limit that restricts the number of "
            "Copilot model requests"
        )

        assert handler._is_rate_limit_error(error) is True

    def test_ignores_token_limit_error(self):
        handler = CopilotLLM()
        error = Exception("prompt token count of 143855 exceeds the limit of 128000")

        assert handler._is_rate_limit_error(error) is False

    def test_ignores_unrelated_error(self):
        handler = CopilotLLM()

        assert handler._is_rate_limit_error(ConnectionError("timeout")) is False

    def test_is_case_insensitive(self):
        handler = CopilotLLM()

        assert handler._is_rate_limit_error(Exception("RATE LIMIT exceeded")) is True


class TestSendPrompt:
    """Tests for _send_prompt."""

    @pytest.mark.asyncio
    async def test_delegates_prompt_and_timeout_to_session(self):
        captured = {}

        class StubSession:
            async def send_and_wait(self, options: dict, timeout: int = 300) -> str:
                captured["options"] = options
                captured["timeout"] = timeout
                return "response"

        class StubPool:
            @asynccontextmanager
            async def acquire(self):
                yield StubSession()

        handler = CopilotLLM(timeout=42)
        handler._pool = StubPool()

        result = await handler._send_prompt("hello")

        assert result == "response"
        assert captured == {"options": {"prompt": "hello"}, "timeout": 42}


class TestHandleRateLimit:
    """Tests for _handle_rate_limit code paths."""

    @pytest.mark.asyncio
    async def test_fails_fast_when_no_prior_success(self):
        handler = CopilotLLM()
        original = Exception("rate limit hit")

        with pytest.raises(RuntimeError, match="first request") as exc_info:
            await handler._handle_rate_limit(original, attempt=1)

        assert exc_info.value.__cause__ is original

    @pytest.mark.asyncio
    async def test_skips_sleep_on_last_attempt(self):
        handler = CopilotLLM(max_retries=3)
        handler._has_succeeded = True
        slept = []

        original_sleep = asyncio.sleep

        async def tracking_sleep(delay: float) -> None:
            slept.append(delay)

        asyncio.sleep = tracking_sleep
        try:
            await handler._handle_rate_limit(Exception("rate limit"), attempt=3)  # last
        finally:
            asyncio.sleep = original_sleep

        assert slept == []

    @pytest.mark.asyncio
    async def test_waits_with_exponential_backoff(self):
        handler = CopilotLLM(max_retries=3, retry_base_delay=10.0)
        handler._has_succeeded = True
        slept = []

        original_sleep = asyncio.sleep

        async def tracking_sleep(delay: float) -> None:
            slept.append(delay)

        asyncio.sleep = tracking_sleep
        try:
            await handler._handle_rate_limit(Exception("rate limit"), attempt=0)
            await handler._handle_rate_limit(Exception("rate limit"), attempt=1)
            await handler._handle_rate_limit(Exception("rate limit"), attempt=2)
        finally:
            asyncio.sleep = original_sleep

        assert slept == [10.0, 20.0, 40.0]


class TestSendWithRetry:
    """Tests for _send_with_retry orchestration."""

    def _make_handler(self, responses: list, **kwargs) -> CopilotLLM:
        """Creates a CopilotLLM with a fake _send_prompt that returns from a list.

        Args:
            responses: Items to return in order. Exceptions are raised.
            **kwargs: Passed to CopilotLLM constructor.

        Returns:
            Configured CopilotLLM instance.
        """
        it = iter(responses)

        async def fake_send(prompt: str) -> Any:
            value = next(it)
            if isinstance(value, Exception):
                raise value
            return value

        handler = CopilotLLM(retry_base_delay=0.0, **kwargs)
        handler._has_succeeded = True
        handler._send_prompt = fake_send
        return handler

    @pytest.mark.asyncio
    async def test_returns_on_first_success(self):
        handler = self._make_handler(["ok"])

        result = await handler._send_with_retry("hello")

        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit_then_succeeds(self):
        handler = self._make_handler(
            [
                Exception("rate limit"),
                "ok",
            ],
            max_retries=3,
        )

        result = await handler._send_with_retry("hello")

        assert result == "ok"

    @pytest.mark.asyncio
    async def test_raises_immediately_on_non_rate_limit_error(self):
        handler = self._make_handler(
            [
                ConnectionError("network down"),
                "ok",
            ]
        )

        with pytest.raises(ConnectionError, match="network down"):
            await handler._send_with_retry("hello")

    @pytest.mark.asyncio
    async def test_fails_fast_on_rate_limit_without_prior_success(self):
        handler = self._make_handler([Exception("rate limit")])
        handler._has_succeeded = False

        with pytest.raises(RuntimeError, match="first request"):
            await handler._send_with_retry("hello")

    @pytest.mark.asyncio
    async def test_raises_last_error_after_all_retries_exhausted(self):
        handler = self._make_handler(
            [
                Exception("rate limit 1"),
                Exception("rate limit 2"),
                Exception("rate limit 3"),
                Exception("rate limit 4"),
            ],
            max_retries=3,
        )

        with pytest.raises(Exception, match="rate limit 4"):
            await handler._send_with_retry("hello")


class FakeSession:
    """Fake Copilot session for testing."""

    def __init__(
        self, response_content: str = "test response", fail_on_use: bool = False
    ):
        self.response_content = response_content
        self.fail_on_use = fail_on_use
        self.destroyed = False
        self.destroy_error = None

    async def send_and_wait(self, options: dict, timeout: int = 300) -> object:
        """Returns fake response."""
        if self.fail_on_use:
            raise ConnectionError("session broken")

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
        self.created_sessions: list[FakeSession] = []

    async def create_session(self, config: dict) -> FakeSession:
        self.sessions_created += 1
        session = FakeSession(self.response_content)
        self.created_sessions.append(session)
        return session


class TestSessionPool:
    """Tests for SessionPool class."""

    @pytest.fixture(autouse=True)
    def patch_permission_handler(self, monkeypatch):
        class FakePermissionHandler:
            approve_all = staticmethod(lambda *_: None)

        monkeypatch.setattr(
            "result_companion.core.analizers.remote.copilot.PermissionHandler",
            FakePermissionHandler,
        )

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
    async def test_failed_session_is_destroyed_and_not_reused(self):
        client = FakeCopilotClient()
        pool = SessionPool(client, "gpt-4.1", pool_size=2)

        async with pool.acquire() as session:
            pass

        bad_session = client.created_sessions[0]
        bad_session.fail_on_use = True

        with pytest.raises(ConnectionError):
            async with pool.acquire() as session:
                await session.send_and_wait({})

        assert bad_session.destroyed is True
        assert pool._created == 0

        async with pool.acquire() as session:
            assert session is not bad_session

        assert client.sessions_created == 2

    @pytest.mark.asyncio
    async def test_failed_session_frees_slot_for_new_creation(self):
        client = FakeCopilotClient()
        pool = SessionPool(client, "gpt-4.1", pool_size=1)

        async with pool.acquire():
            pass

        bad_session = client.created_sessions[0]
        bad_session.fail_on_use = True

        with pytest.raises(ConnectionError):
            async with pool.acquire() as session:
                await session.send_and_wait({})

        assert pool._created == 0

        async with pool.acquire() as new_session:
            assert new_session is not bad_session
            assert new_session.fail_on_use is False

        assert pool._created == 1

    @pytest.mark.asyncio
    async def test_discard_handles_destroy_failure(self):
        client = FakeCopilotClient()
        pool = SessionPool(client, "gpt-4.1", pool_size=1)

        async with pool.acquire():
            pass

        bad_session = client.created_sessions[0]

        async def exploding_destroy(self):
            self.destroy_error = True
            raise RuntimeError("destroy failed")

        bad_session.destroy = exploding_destroy
        bad_session.fail_on_use = True
        bad_session.destroy_error = True

        with pytest.raises(ConnectionError):
            async with pool.acquire() as session:
                await session.send_and_wait({})

        assert pool._created == 0
        assert client.sessions_created == 1

        async with pool.acquire():
            pass

        assert client.sessions_created == 2

    @pytest.mark.asyncio
    async def test_close_destroys_sessions(self):
        client = FakeCopilotClient()
        pool = SessionPool(client, "gpt-4.1", pool_size=2)

        async with pool.acquire():
            pass

        await pool.close()

        assert pool._created == 0


class TestEnsureStartedEarlyReturn:
    """Tests for _ensure_started early-return guard."""

    @pytest.mark.asyncio
    async def test_skips_initialization_when_already_started(self):
        handler = CopilotLLM()
        handler._started = True
        handler._client = object()
        handler._pool = object()

        await handler._ensure_started("gpt-4.1")

        assert handler._started is True


class TestCheckAuth:
    """Tests for _check_auth delegation."""

    @pytest.mark.asyncio
    async def test_passes_when_authenticated(self):
        client = FakeStartableClient()
        handler = CopilotLLM()

        await handler._check_auth(client)


class TestResolveCliPath:
    """Tests for _resolve_cli_path delegation."""

    def test_returns_none_when_cli_url_set(self):
        handler = CopilotLLM(cli_url="http://localhost:8080")
        assert handler._resolve_cli_path() is None
