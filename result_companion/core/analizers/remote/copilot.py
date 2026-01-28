"""LangChain adapter for GitHub Copilot SDK."""

import asyncio
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict


def messages_to_prompt(messages: list[BaseMessage]) -> str:
    """Converts LangChain messages to a single prompt string.

    Args:
        messages: List of LangChain messages.

    Returns:
        Formatted prompt string for Copilot SDK.
    """
    parts = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            parts.append(f"[System]: {msg.content}")
        elif isinstance(msg, HumanMessage):
            parts.append(f"[User]: {msg.content}")
        elif isinstance(msg, AIMessage):
            parts.append(f"[Assistant]: {msg.content}")
        else:
            parts.append(str(msg.content))
    return "\n\n".join(parts)


class ChatCopilot(BaseChatModel):
    """LangChain adapter for GitHub Copilot SDK.

    Uses lazy initialization - client starts on first call.
    Session is reused across calls for efficiency.
    """

    model: str = "gpt-4.1"
    cli_path: Optional[str] = None
    cli_url: Optional[str] = None

    _client: Any = None
    _session: Any = None
    _started: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "copilot"

    async def _ensure_started(self) -> None:
        """Ensures client and session are initialized."""
        if self._started:
            return

        from copilot import CopilotClient

        client_opts = {}
        if self.cli_path:
            client_opts["cli_path"] = self.cli_path
        if self.cli_url:
            client_opts["cli_url"] = self.cli_url

        self._client = CopilotClient(client_opts) if client_opts else CopilotClient()
        await self._client.start()
        self._session = await self._client.create_session({"model": self.model})
        self._started = True

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation using Copilot SDK.

        Args:
            messages: LangChain message sequence.
            stop: Stop sequences (not supported by Copilot SDK).
            run_manager: Callback manager.

        Returns:
            ChatResult with AI response.
        """
        await self._ensure_started()
        prompt = messages_to_prompt(messages)
        response = await self._session.send_and_wait({"prompt": prompt})
        content = response.data.content if response and response.data else ""
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation - wraps async implementation."""
        return asyncio.run(self._agenerate(messages, stop, run_manager, **kwargs))

    async def aclose(self) -> None:
        """Cleanup client and session resources."""
        if self._session:
            await self._session.destroy()
            self._session = None
        if self._client:
            await self._client.stop()
            self._client = None
        self._started = False
