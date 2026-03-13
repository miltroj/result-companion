"""Manual test: verifies fail-fast when Copilot CLI is missing."""

import asyncio

from result_companion.core.analizers.llm_router import _smart_acompletion

asyncio.run(
    _smart_acompletion(
        messages=[{"role": "user", "content": "Hello"}],
        model="copilot_sdk/gpt-5-mini",
    )
)
