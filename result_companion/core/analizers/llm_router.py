"""Shared LLM routing helpers for Copilot SDK and LiteLLM."""

from typing import Any

from litellm import acompletion

_copilot_registered = False


def _ensure_copilot_registered():
    """Registers Copilot handler lazily."""
    global _copilot_registered
    if not _copilot_registered:
        from result_companion.core.analizers.remote.copilot import (
            register_copilot_provider,
        )

        register_copilot_provider()
        _copilot_registered = True


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
        _ensure_copilot_registered()

    return await acompletion(messages=messages, num_retries=num_retries, **llm_params)
