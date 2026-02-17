"""Tests for LLM router helpers."""

import pytest

from result_companion.core.analizers.llm_router import _smart_acompletion


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
