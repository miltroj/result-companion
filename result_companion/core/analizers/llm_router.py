"""Shared LLM routing helpers for Copilot SDK and LiteLLM."""

from typing import Any

from litellm import acompletion

# TODO: Replace global singleton with dependency injection (pass handler to callers).
_copilot_handler = None


def _get_copilot_handler():
    """Returns Copilot handler, initializing lazily."""
    global _copilot_handler
    if _copilot_handler is None:
        from result_companion.core.analizers.remote.copilot import CopilotLLM

        _copilot_handler = CopilotLLM()
    return _copilot_handler


async def _smart_acompletion(
    messages: list[dict], num_retries: int = 3, **llm_params: Any
):
    """Routes to Copilot SDK or LiteLLM using native retry capabilities.

    Args:
        messages: List of message dicts.
        num_retries: Number of times to retry transient errors (handled by LiteLLM).
        **llm_params: LLM parameters including model.

    Returns:
        Model response.
    """
    model = llm_params.get("model", "")

    if model.startswith("copilot_sdk/"):
        handler = _get_copilot_handler()
        return await handler.acompletion(
            messages=messages, num_retries=num_retries, **llm_params
        )

    return await acompletion(messages=messages, num_retries=num_retries, **llm_params)
