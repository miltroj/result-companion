"""Shared LLM routing helpers for Copilot SDK and LiteLLM."""

from typing import Any

from litellm import acompletion

# Lazy-loaded Copilot handler to avoid import if not needed
_copilot_handler = None


def _get_copilot_handler():
    """Returns Copilot handler, initializing lazily."""
    global _copilot_handler
    if _copilot_handler is None:
        from result_companion.core.analizers.remote.copilot import CopilotLLM

        _copilot_handler = CopilotLLM()
    return _copilot_handler


async def _smart_acompletion(messages: list[dict], **llm_params: Any):
    """Routes to Copilot SDK or LiteLLM based on model prefix.

    Args:
        messages: List of message dicts.
        **llm_params: LLM parameters including model.

    Returns:
        Model response.
    """
    model = llm_params.get("model", "")

    if model.startswith("copilot_sdk/"):
        handler = _get_copilot_handler()
        return await handler.acompletion(model=model, messages=messages)

    return await acompletion(messages=messages, **llm_params)
