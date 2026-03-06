import os

import pytest
from litellm import acompletion

from result_companion.core.analizers.local.ollama_runner import ollama_on_init_strategy

MODEL_NAME = os.getenv("OLLAMA_TEST_MODEL", "deepseek-r1:1.5b")
MODEL_FULL = os.getenv("OLLAMA_TEST_MODEL_LITELLM", f"ollama_chat/{MODEL_NAME}")
API_BASE = os.getenv("OLLAMA_TEST_API_BASE", "http://localhost:11434")

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_ollama_runner_smoke():
    server_manager = ollama_on_init_strategy(MODEL_NAME, server_url=API_BASE)
    try:
        response = await acompletion(
            model=MODEL_FULL,
            messages=[{"role": "user", "content": "Give one short fact."}],
            api_base=API_BASE,
        )
        content = response.choices[0].message.content
        assert isinstance(content, str)
        assert content.strip()
    finally:
        server_manager.cleanup()
