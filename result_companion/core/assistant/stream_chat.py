"""LLM streaming for the assistant chat command."""

from typing import Any, AsyncGenerator

from litellm import acompletion

from result_companion.core.analizers.llm_router import _smart_acompletion


async def stream_chat(
    messages: list[dict], llm_params: dict[str, Any]
) -> AsyncGenerator[str, None]:
    """Yields reply tokens. Falls back to single yield for copilot_sdk models.

    Args:
        messages: Conversation history including system prompt.
        llm_params: LiteLLM parameters (model, api_base, api_key, …).

    Yields:
        Reply tokens as they arrive.
    """
    if llm_params.get("model", "").startswith("copilot_sdk/"):
        response = await _smart_acompletion(messages=messages, **llm_params)
        yield response.choices[0].message.content
        return

    response = await acompletion(stream=True, messages=messages, **llm_params)
    async for chunk in response:
        token = chunk.choices[0].delta.content or ""
        if token:
            yield token
