"""Shared LLM routing helpers for Copilot SDK and LiteLLM."""

import asyncio
import logging
from typing import Any

import litellm
from litellm import acompletion

logger = logging.getLogger(__name__)

_copilot_handler = None

_MAX_RETRIES = 3
_BACKOFF_BASE_DELAY = 1.0

_NON_RETRYABLE_ERRORS = (
    litellm.AuthenticationError,
    litellm.BadRequestError,
    litellm.NotFoundError,
)


def _is_retryable(error: Exception) -> bool:
    """Returns True for transient errors worth retrying.

    Args:
        error: The exception to evaluate.

    Returns:
        True if the error is transient and should be retried.
    """
    return not isinstance(error, _NON_RETRYABLE_ERRORS)


def _get_copilot_handler():
    """Returns Copilot handler, initializing lazily."""
    global _copilot_handler
    if _copilot_handler is None:
        from result_companion.core.analizers.remote.copilot import CopilotLLM

        _copilot_handler = CopilotLLM()
    return _copilot_handler


async def _route_completion(messages: list[dict], **llm_params: Any):
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


async def _smart_acompletion(messages: list[dict], **llm_params: Any):
    """Routes to Copilot SDK or LiteLLM with retry and exponential backoff.

    Retries up to _MAX_RETRIES times on transient errors (rate limits,
    timeouts, server errors). Permanent errors (auth, bad request) fail
    immediately.

    Args:
        messages: List of message dicts.
        **llm_params: LLM parameters including model.

    Returns:
        Model response.
    """
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return await _route_completion(messages, **llm_params)
        except Exception as error:
            if not _is_retryable(error) or attempt == _MAX_RETRIES:
                raise
            delay = _BACKOFF_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                attempt,
                _MAX_RETRIES,
                error,
                delay,
            )
            await asyncio.sleep(delay)
