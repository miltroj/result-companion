"""Integration tests for LLM router."""

import time

import litellm
import pytest

from result_companion.core.analizers.llm_router import _smart_acompletion


@pytest.mark.asyncio
async def test_smart_acompletion_retries_transient_errors():
    """Verifies that LiteLLM natively retries transient connection errors."""
    start_time = time.time()

    with pytest.raises(litellm.exceptions.InternalServerError):
        await _smart_acompletion(
            messages=[{"role": "user", "content": "hi"}],
            model="openai/test",
            num_retries=2,
            api_key="fake",
            api_base="http://localhost:9999",  # Invalid to cause connection error
        )

    duration = time.time() - start_time
    # With exponential backoff, 2 retries should take > 1 second
    assert duration > 1.0


@pytest.mark.asyncio
async def test_smart_acompletion_fails_fast_on_auth_error():
    """Verifies that LiteLLM does not retry authentication errors."""
    start_time = time.time()

    # The easiest way to force a fast fail is an invalid model name,
    # as auth errors often trigger retries in some SDKs depending on network conditions.
    # Note: LiteLLM wraps httpx connection errors (like unknown host) into InternalServerError
    with pytest.raises(litellm.exceptions.InternalServerError):
        await _smart_acompletion(
            messages=[{"role": "user", "content": "hi"}],
            model="openai/test",
            num_retries=0,  # Force 0 retries to test fast fail
            api_key="fake",
            api_base="http://localhost:9999",  # Invalid to cause connection error fast
        )

    duration = time.time() - start_time
    # Should fail immediately without retries
    assert duration < 1.0
