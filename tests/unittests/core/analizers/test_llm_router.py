"""Tests for LLM router helpers."""

import pytest

from result_companion.core.analizers.llm_router import _smart_acompletion


def make_fake_handler(response=None, called_with=None):
    """Creates a fake Copilot handler."""

    class FakeHandler:
        async def acompletion(self, messages, num_retries, **kwargs):
            if called_with is not None:
                called_with["messages"] = messages
                called_with["num_retries"] = num_retries
                called_with.update(kwargs)
            return response

    return FakeHandler()


class TestSmartAcompletion:
    """Tests for _smart_acompletion function."""

    @pytest.mark.asyncio
    async def test_routes_to_litellm_with_default_retries(self, monkeypatch):
        expected_response = object()
        called_with = {}

        async def fake_acompletion(messages, num_retries, **kwargs):
            called_with["messages"] = messages
            called_with["num_retries"] = num_retries
            called_with.update(kwargs)
            return expected_response

        monkeypatch.setattr(
            "result_companion.core.analizers.llm_router.acompletion", fake_acompletion
        )

        messages = [{"role": "user", "content": "test"}]
        result = await _smart_acompletion(messages=messages, model="gpt-4")

        assert result is expected_response
        assert called_with["model"] == "gpt-4"
        assert called_with["messages"] == messages
        assert called_with["num_retries"] == 3

    @pytest.mark.asyncio
    async def test_respects_provided_num_retries(self, monkeypatch):
        called_with = {}

        async def fake_acompletion(messages, num_retries, **kwargs):
            called_with["num_retries"] = num_retries
            return object()

        monkeypatch.setattr(
            "result_companion.core.analizers.llm_router.acompletion", fake_acompletion
        )

        await _smart_acompletion(messages=[], model="gpt-4", num_retries=5)

        assert called_with["num_retries"] == 5

    @pytest.mark.asyncio
    async def test_raises_persistent_errors(self, monkeypatch):
        async def fake_acompletion(messages, num_retries, **kwargs):
            raise ValueError("Persistent error")

        monkeypatch.setattr(
            "result_companion.core.analizers.llm_router.acompletion", fake_acompletion
        )

        with pytest.raises(ValueError, match="Persistent error"):
            await _smart_acompletion(messages=[], model="gpt-4")

    @pytest.mark.asyncio
    async def test_routes_to_copilot_handler_when_model_starts_with_copilot_sdk(
        self, monkeypatch
    ):
        expected_response = object()
        called_with = {}
        fake_handler = make_fake_handler(
            response=expected_response, called_with=called_with
        )

        monkeypatch.setattr(
            "result_companion.core.analizers.llm_router._copilot_handler", fake_handler
        )

        messages = [{"role": "user", "content": "test"}]
        result = await _smart_acompletion(
            messages=messages, model="copilot_sdk/gpt-4.1"
        )

        assert result is expected_response
        assert called_with["model"] == "copilot_sdk/gpt-4.1"
        assert called_with["messages"] == messages
        assert called_with["num_retries"] == 3

    @pytest.mark.asyncio
    async def test_copilot_handler_receives_num_retries(self, monkeypatch):
        called_with = {}
        fake_handler = make_fake_handler(response=object(), called_with=called_with)

        monkeypatch.setattr(
            "result_companion.core.analizers.llm_router._copilot_handler", fake_handler
        )

        await _smart_acompletion(
            messages=[], model="copilot_sdk/gpt-4.1", num_retries=7
        )

        assert called_with["num_retries"] == 7

    def test_get_copilot_handler_returns_same_instance(self, monkeypatch):
        monkeypatch.setattr(
            "result_companion.core.analizers.llm_router._copilot_handler", None
        )

        from result_companion.core.analizers.llm_router import _get_copilot_handler

        first = _get_copilot_handler()
        second = _get_copilot_handler()

        assert first is second
