"""End-to-end tests for Copilot SDK integration (LiteLLM adapter)."""

import pytest

from result_companion.core.analizers.remote.copilot import CopilotLLM

# Mark all tests in this module as e2e
pytestmark = pytest.mark.e2e


class TestErrorHandlingE2E:
    """E2E tests for error handling against real Copilot API."""

    @pytest.mark.asyncio
    async def test_fails_fast_when_prompt_exceeds_token_limit(self):
        """Massive prompt should trigger error on first request → Exception."""
        huge_prompt = "Analyze this: " + ("x " * 200_000)
        handler = CopilotLLM(model="gpt-4.1", pool_size=1, max_retries=3)
        messages = [{"role": "user", "content": huge_prompt}]

        try:
            # CAPIError: 400 prompt token count of 209961 exceeds the limit of 64000
            with pytest.raises(Exception, match="CAPIError: 400 prompt token count of"):
                await handler.acompletion("copilot_sdk/gpt-4.1", messages)
        finally:
            await handler.aclose()


class TestCopilotE2E:
    """End-to-end tests for CopilotLLM with real Copilot SDK."""

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

    @pytest.mark.asyncio
    async def test_smart_acompletion_routes_to_copilot(self):
        """Tests that _smart_acompletion successfully routes to the Copilot SDK."""
        from result_companion.core.analizers.llm_router import _smart_acompletion

        messages = [{"role": "user", "content": "Return exactly the word 'SUCCESS'"}]

        try:
            result = await _smart_acompletion(
                messages=messages, model="copilot_sdk/gpt-4.1", num_retries=1
            )
            content = result.choices[0]["message"]["content"]
            assert content
            assert "SUCCESS" in content.upper()
        finally:
            # Clean up the lazily initialized module-level handler
            from result_companion.core.analizers.llm_router import _copilot_handler

            if _copilot_handler:
                await _copilot_handler.aclose()
