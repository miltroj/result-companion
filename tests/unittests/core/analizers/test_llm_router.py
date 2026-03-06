"""Tests for LLM router helpers."""

import pytest

from result_companion.core.analizers.llm_router import _smart_acompletion


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
            "result_companion.core.analizers.llm_router.acompletion",
            fake_acompletion,
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
            called_with.update(kwargs)
            return object()

        monkeypatch.setattr(
            "result_companion.core.analizers.llm_router.acompletion",
            fake_acompletion,
        )

        await _smart_acompletion(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4",
            num_retries=5,
        )

        assert called_with["num_retries"] == 5

    @pytest.mark.asyncio
    async def test_raises_persistent_errors(self, monkeypatch):
        """Tests that persistent errors (or exhausted retries) are raised."""

        async def fake_acompletion(messages, num_retries, **kwargs):
            raise ValueError("Persistent error")

        monkeypatch.setattr(
            "result_companion.core.analizers.llm_router.acompletion",
            fake_acompletion,
        )

        with pytest.raises(ValueError, match="Persistent error"):
            await _smart_acompletion(
                messages=[{"role": "user", "content": "test"}],
                model="gpt-4",
            )

    @pytest.mark.asyncio
    async def test_registers_copilot_when_model_starts_with_copilot_sdk(
        self, monkeypatch
    ):
        expected_response = object()
        called_with = {}
        registration_called = False

        def fake_register():
            nonlocal registration_called
            registration_called = True

        async def fake_acompletion(messages, num_retries, **kwargs):
            called_with["model"] = kwargs.get("model")
            return expected_response

        monkeypatch.setattr(
            "result_companion.core.analizers.llm_router.acompletion",
            fake_acompletion,
        )
        monkeypatch.setattr(
            "result_companion.core.analizers.llm_router._copilot_registered",
            False,
        )
        monkeypatch.setattr(
            "result_companion.core.analizers.remote.copilot.register_copilot_provider",
            fake_register,
        )

        messages = [{"role": "user", "content": "test"}]
        result = await _smart_acompletion(
            messages=messages, model="copilot_sdk/gpt-4.1"
        )

        assert result is expected_response
        assert called_with["model"] == "copilot_sdk/gpt-4.1"
        assert registration_called is True

    @pytest.mark.asyncio
    async def test_only_registers_copilot_once(self, monkeypatch):
        registration_calls = 0

        def fake_register():
            nonlocal registration_calls
            registration_calls += 1

        async def fake_acompletion(messages, num_retries, **kwargs):
            return object()

        monkeypatch.setattr(
            "result_companion.core.analizers.llm_router.acompletion",
            fake_acompletion,
        )
        monkeypatch.setattr(
            "result_companion.core.analizers.llm_router._copilot_registered",
            False,
        )
        monkeypatch.setattr(
            "result_companion.core.analizers.remote.copilot.register_copilot_provider",
            fake_register,
        )

        # Call twice
        await _smart_acompletion(messages=[], model="copilot_sdk/test")
        await _smart_acompletion(messages=[], model="copilot_sdk/test")

        assert registration_calls == 1
