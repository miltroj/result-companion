"""Tests for LLM router helpers."""

import litellm
import pytest

from result_companion.core.analizers.llm_router import (
    _NON_RETRYABLE_ERRORS,
    _is_retryable,
    _smart_acompletion,
)


class FakeCallTracker:
    """Tracks calls, raising errors for the first fail_count calls."""

    def __init__(self, fail_count: int, error: Exception, response: object):
        self.calls = 0
        self.fail_count = fail_count
        self.error = error
        self.response = response

    async def __call__(self, messages: list[dict], **kwargs):
        self.calls += 1
        if self.calls <= self.fail_count:
            raise self.error
        return self.response


@pytest.fixture
def no_retry_delay(monkeypatch):
    """Removes sleep delay between retries."""

    async def instant(delay):
        pass

    monkeypatch.setattr(
        "result_companion.core.analizers.llm_router.asyncio.sleep", instant
    )


class TestSmartAcompletion:
    """Tests for _smart_acompletion function."""

    @pytest.mark.asyncio
    async def test_routes_to_copilot_when_model_starts_with_copilot_sdk(
        self, monkeypatch
    ):
        expected_response = object()
        called_with = {}

        class FakeHandler:
            async def acompletion(self, model: str, messages: list[dict]):
                called_with["model"] = model
                called_with["messages"] = messages
                return expected_response

        def fake_get_handler():
            return FakeHandler()

        monkeypatch.setattr(
            "result_companion.core.analizers.llm_router._get_copilot_handler",
            fake_get_handler,
        )

        messages = [{"role": "user", "content": "test"}]
        result = await _smart_acompletion(
            messages=messages, model="copilot_sdk/gpt-4.1"
        )

        assert result is expected_response
        assert called_with["model"] == "copilot_sdk/gpt-4.1"
        assert called_with["messages"] == messages


class TestGetCopilotHandler:
    """Tests for _get_copilot_handler function."""

    def test_returns_same_instance(self, monkeypatch):
        class FakeCopilotLLM:
            pass

        monkeypatch.setattr(
            "result_companion.core.analizers.remote.copilot.CopilotLLM",
            FakeCopilotLLM,
        )
        monkeypatch.setattr(
            "result_companion.core.analizers.llm_router._copilot_handler",
            None,
        )

        from result_companion.core.analizers.llm_router import _get_copilot_handler

        first = _get_copilot_handler()
        second = _get_copilot_handler()

        assert isinstance(first, FakeCopilotLLM)
        assert first is second


class TestIsRetryable:
    """Tests for _is_retryable error classification."""

    def test_generic_exception_is_retryable(self):
        assert _is_retryable(Exception("server error")) is True

    def test_connection_error_is_retryable(self):
        assert _is_retryable(ConnectionError("refused")) is True

    def test_timeout_error_is_retryable(self):
        assert _is_retryable(TimeoutError("timed out")) is True

    def test_non_retryable_types_include_auth_and_bad_request(self):
        expected = {
            litellm.AuthenticationError,
            litellm.BadRequestError,
            litellm.NotFoundError,
        }
        assert expected.issubset(set(_NON_RETRYABLE_ERRORS))


class TestSmartAcompletionRetry:
    """Tests for retry behavior in _smart_acompletion."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_attempt(self, monkeypatch):
        expected = object()
        tracker = FakeCallTracker(
            fail_count=0, error=Exception("unused"), response=expected
        )
        monkeypatch.setattr(
            "result_companion.core.analizers.llm_router._route_completion",
            tracker,
        )

        result = await _smart_acompletion(
            messages=[{"role": "user", "content": "test"}], model="test"
        )

        assert result is expected
        assert tracker.calls == 1

    @pytest.mark.asyncio
    async def test_succeeds_after_transient_failure(self, monkeypatch, no_retry_delay):
        expected = object()
        tracker = FakeCallTracker(
            fail_count=1, error=ConnectionError("transient"), response=expected
        )
        monkeypatch.setattr(
            "result_companion.core.analizers.llm_router._route_completion",
            tracker,
        )

        result = await _smart_acompletion(
            messages=[{"role": "user", "content": "test"}], model="test"
        )

        assert result is expected
        assert tracker.calls == 2

    @pytest.mark.asyncio
    async def test_raises_after_max_retries_exhausted(
        self, monkeypatch, no_retry_delay
    ):
        tracker = FakeCallTracker(
            fail_count=10,
            error=ConnectionError("persistent"),
            response=object(),
        )
        monkeypatch.setattr(
            "result_companion.core.analizers.llm_router._route_completion",
            tracker,
        )

        with pytest.raises(ConnectionError, match="persistent"):
            await _smart_acompletion(
                messages=[{"role": "user", "content": "test"}], model="test"
            )

        assert tracker.calls == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable_error(self, monkeypatch, no_retry_delay):
        monkeypatch.setattr(
            "result_companion.core.analizers.llm_router._is_retryable",
            lambda e: False,
        )
        tracker = FakeCallTracker(
            fail_count=3,
            error=ValueError("non-retryable"),
            response=object(),
        )
        monkeypatch.setattr(
            "result_companion.core.analizers.llm_router._route_completion",
            tracker,
        )

        with pytest.raises(ValueError, match="non-retryable"):
            await _smart_acompletion(
                messages=[{"role": "user", "content": "test"}], model="test"
            )

        assert tracker.calls == 1
