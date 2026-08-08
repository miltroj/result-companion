"""Tests for the stream_chat async generator."""

import pytest

import result_companion.core.assistant.stream_chat as stream_chat_module
from result_companion.core.assistant.stream_chat import stream_chat


class FakeModelResponse:
    """Fake non-streaming response (copilot_sdk path)."""

    class _Choice:
        class _Message:
            content = "full copilot reply"

        message = _Message()

    choices = [_Choice()]


class FakeChunk:
    """Fake streaming chunk (litellm path)."""

    class _Choice:
        class _Delta:
            def __init__(self, content):
                self.content = content

        def __init__(self, content):
            self.delta = self._Delta(content)

    def __init__(self, content):
        self.choices = [self._Choice(content)]


def make_fake_acompletion(token_contents: list):
    """Returns a fake acompletion that streams the given token contents."""

    async def _fake(**kwargs):
        async def _gen():
            for content in token_contents:
                yield FakeChunk(content)

        return _gen()

    return _fake


async def collect(gen) -> list[str]:
    return [t async for t in gen]


@pytest.mark.asyncio
async def test_stream_chat_copilot_sdk_yields_full_content(monkeypatch):
    async def fake_smart_acompletion(messages, **kwargs):
        return FakeModelResponse()

    monkeypatch.setattr(stream_chat_module, "_smart_acompletion", fake_smart_acompletion)

    tokens = await collect(stream_chat([], {"model": "copilot_sdk/gpt-5-mini"}))

    assert tokens == ["full copilot reply"]


@pytest.mark.asyncio
async def test_stream_chat_litellm_yields_tokens(monkeypatch):
    monkeypatch.setattr(stream_chat_module, "acompletion", make_fake_acompletion(["hello", " world"]))

    tokens = await collect(stream_chat([], {"model": "openai/gpt-4o"}))

    assert tokens == ["hello", " world"]


@pytest.mark.asyncio
async def test_stream_chat_skips_empty_tokens(monkeypatch):
    monkeypatch.setattr(stream_chat_module, "acompletion", make_fake_acompletion(["hi", "", None]))

    tokens = await collect(stream_chat([], {"model": "openai/gpt-4o"}))

    assert tokens == ["hi"]


@pytest.mark.asyncio
async def test_stream_chat_passes_messages_and_params(monkeypatch):
    captured = {}

    async def fake_smart_acompletion(messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return FakeModelResponse()

    monkeypatch.setattr(stream_chat_module, "_smart_acompletion", fake_smart_acompletion)

    messages = [{"role": "user", "content": "hello"}]
    await collect(stream_chat(messages, {"model": "copilot_sdk/gpt-5-mini", "api_base": "http://x"}))

    assert captured["messages"] == messages
    assert captured["kwargs"]["api_base"] == "http://x"
