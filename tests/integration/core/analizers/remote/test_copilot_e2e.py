"""End-to-end tests for Copilot SDK integration (LiteLLM adapter)."""

import pytest

from result_companion.core.analizers.remote.copilot import CopilotLLM

# Mark all tests in this module as e2e
pytestmark = pytest.mark.e2e


def copilot_cli_available() -> bool:
    """Checks if Copilot CLI is available in PATH."""
    import shutil

    return shutil.which("copilot") is not None


class TestCopilotE2E:
    """End-to-end tests for CopilotLLM with real Copilot SDK."""

    @pytest.mark.skipif(
        not copilot_cli_available(),
        reason="Copilot CLI not found in PATH",
    )
    @pytest.mark.asyncio
    async def test_simple_prompt_returns_response(self):
        """Tests basic prompt-response flow with real Copilot."""
        handler = CopilotLLM(model="gpt-4.1")
        messages = [{"role": "user", "content": "What is 2 + 2?"}]

        try:
            result = await handler.acompletion("copilot_sdk/gpt-4.1", messages)
            content = result.choices[0]["message"]["content"]
            assert content
            assert "4" in content
        finally:
            await handler.aclose()
