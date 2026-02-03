"""End-to-end tests for Copilot SDK integration.

These tests connect to real GitHub Copilot service.
Requires:
    - GitHub Copilot CLI installed (`copilot` in PATH)
    - Valid GitHub Copilot license (business or individual)

Run with: pytest tests/integration/core/analizers/remote/test_copilot_e2e.py -v -s
Skip in CI by using: pytest -m "not e2e"
"""

import asyncio

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from result_companion.core.analizers.remote.copilot import ChatCopilot

# Mark all tests in this module as e2e
pytestmark = pytest.mark.e2e


def copilot_cli_available() -> bool:
    """Checks if Copilot CLI is available in PATH."""
    import shutil

    return shutil.which("copilot") is not None


@pytest.fixture
def chat_copilot() -> ChatCopilot:
    """Creates ChatCopilot instance for testing."""
    return ChatCopilot(model="gpt-4.1")


class TestCopilotE2E:
    """End-to-end tests for ChatCopilot with real Copilot SDK."""

    @pytest.mark.skipif(
        not copilot_cli_available(),
        reason="Copilot CLI not found in PATH",
    )
    @pytest.mark.asyncio
    async def test_simple_prompt_returns_response(self, chat_copilot: ChatCopilot):
        """Tests basic prompt-response flow with real Copilot."""
        messages = [HumanMessage(content="What is 2 + 2? Answer with just the number.")]

        try:
            result = await chat_copilot._agenerate(messages)
            content = result.generations[0].message.content

            assert content is not None
            assert len(content) > 0
            print(f"\nResponse: {content}")
        finally:
            await chat_copilot.aclose()

    @pytest.mark.skipif(
        not copilot_cli_available(),
        reason="Copilot CLI not found in PATH",
    )
    @pytest.mark.asyncio
    async def test_system_message_with_human_message(self, chat_copilot: ChatCopilot):
        """Tests system + human message combination."""
        messages = [
            SystemMessage(
                content="You are a helpful assistant. Always respond in uppercase."
            ),
            HumanMessage(content="Say hello"),
        ]

        try:
            result = await chat_copilot._agenerate(messages)
            content = result.generations[0].message.content

            assert content is not None
            assert len(content) > 0
            print(f"\nResponse: {content}")
        finally:
            await chat_copilot.aclose()

    @pytest.mark.skipif(
        not copilot_cli_available(),
        reason="Copilot CLI not found in PATH",
    )
    @pytest.mark.asyncio
    async def test_robot_framework_failure_analysis(self, chat_copilot: ChatCopilot):
        """Tests real-world use case: analyzing Robot Framework test failure."""
        sample_failure = """
        Test: Login With Valid Credentials
        Status: FAIL
        Message: ElementNotVisibleException: Element 'id:login-button' not visible

        Keywords executed:
        - Open Browser  https://example.com  chrome
        - Wait Until Element Is Visible  id:login-form  timeout=10s
        - Input Text  id:username  testuser
        - Input Text  id:password  secret123
        - Click Button  id:login-button  # FAILED HERE
        """

        messages = [
            SystemMessage(
                content="You are a Robot Framework test failure analyst. "
                "Provide brief, actionable insights."
            ),
            HumanMessage(content=f"Analyze this test failure:\n{sample_failure}"),
        ]

        try:
            result = await chat_copilot._agenerate(messages)
            content = result.generations[0].message.content

            assert content is not None
            assert len(content) > 50  # Expect meaningful analysis
            print(f"\nAnalysis:\n{content}")
        finally:
            await chat_copilot.aclose()


class TestCopilotSDKDirect:
    """Direct Copilot SDK tests without LangChain wrapper.

    Useful for debugging SDK connectivity issues.
    """

    @pytest.mark.skipif(
        not copilot_cli_available(),
        reason="Copilot CLI not found in PATH",
    )
    @pytest.mark.asyncio
    async def test_sdk_direct_connection(self):
        """Tests direct SDK connection without LangChain wrapper."""
        from copilot import CopilotClient

        client = CopilotClient()
        await client.start()

        try:
            session = await client.create_session({"model": "gpt-4.1"})

            done = asyncio.Event()
            response_content = []

            def on_event(event):
                if event.type.value == "assistant.message":
                    response_content.append(event.data.content)
                elif event.type.value == "session.idle":
                    done.set()

            session.on(on_event)
            await session.send(
                {"prompt": "What is the capital of France? One word answer."}
            )
            await asyncio.wait_for(done.wait(), timeout=30.0)

            assert len(response_content) > 0
            print(f"\nDirect SDK response: {response_content[0]}")

            await session.destroy()
        finally:
            await client.stop()

    @pytest.mark.skipif(
        not copilot_cli_available(),
        reason="Copilot CLI not found in PATH",
    )
    @pytest.mark.asyncio
    async def test_sdk_list_models(self):
        """Tests retrieving available models from SDK."""
        from copilot import CopilotClient

        client = CopilotClient()
        await client.start()

        try:
            # Check if models method exists and works
            if hasattr(client, "get_models"):
                models = await client.get_models()
                print(f"\nAvailable models: {models}")
                assert models is not None
            else:
                print("\nNote: get_models() not available in this SDK version")
        finally:
            await client.stop()


if __name__ == "__main__":
    """Run tests directly for quick manual testing."""
    import sys

    async def run_quick_test():
        """Quick connectivity test."""
        print("Testing Copilot SDK connectivity...")

        if not copilot_cli_available():
            print("ERROR: Copilot CLI not found in PATH")
            print("Install it following: https://github.com/github/copilot-sdk")
            sys.exit(1)

        chat = ChatCopilot(model="gpt-4.1")
        messages = [HumanMessage(content="Say 'Hello from Copilot SDK!' exactly.")]

        try:
            print("Connecting to Copilot...")
            result = await chat._agenerate(messages)
            content = result.generations[0].message.content
            print(f"SUCCESS! Response: {content}")
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        finally:
            await chat.aclose()

    asyncio.run(run_quick_test())
