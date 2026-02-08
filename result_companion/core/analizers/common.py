"""Common utilities for LLM analysis.

This module contains shared utilities for LLM-based analysis.
The main analysis logic is in factory_common.py.
"""

import asyncio
from typing import Any

from litellm import acompletion


async def run_with_semaphore(semaphore: asyncio.Semaphore, coroutine: Any) -> Any:
    """Runs a coroutine with semaphore-based concurrency control.

    Args:
        semaphore: Semaphore for limiting concurrency.
        coroutine: Coroutine to run.

    Returns:
        Result of the coroutine.
    """
    async with semaphore:
        return await coroutine


async def streaming_llm_call(
    prompt: str,
    llm_params: dict[str, Any],
) -> str:
    """Makes a streaming LLM call and returns the full response.

    Args:
        prompt: The prompt to send to the LLM.
        llm_params: Parameters for LiteLLM acompletion.

    Returns:
        The complete LLM response content.
    """
    messages = [{"role": "user", "content": prompt}]
    params = {**llm_params, "stream": True}

    result_chunks = []
    response = await acompletion(messages=messages, **params)

    async for chunk in response:
        if chunk.choices[0].delta.content:
            result_chunks.append(chunk.choices[0].delta.content)

    return "".join(result_chunks)
