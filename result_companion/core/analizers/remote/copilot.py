"""LangChain adapter for GitHub Copilot SDK with session pooling."""

import asyncio
import logging
import os
import stat
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from copilot import CopilotClient
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict

logger = logging.getLogger(__name__)


def ensure_executable(path: str) -> None:
    """Adds execute permission to a binary if missing.

    Pip wheel extraction strips execute bits from bundled binaries.
    This fixes the permission so subprocess.Popen can run the file.

    Args:
        path: Absolute path to the binary file.
    """
    if os.access(path, os.X_OK):
        return
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    logger.info("Fixed execute permission on copilot binary: %s", path)


def messages_to_prompt(messages: list[BaseMessage]) -> str:
    """Converts LangChain messages to a single prompt string."""
    prefixes = {
        SystemMessage: "[System]",
        HumanMessage: "[User]",
        AIMessage: "[Assistant]",
    }
    parts = [f"{prefixes.get(type(m), '')}: {m.content}" for m in messages]
    return "\n\n".join(parts)


class SessionPool:
    """Pool of Copilot sessions for concurrent requests."""

    def __init__(self, client: Any, model: str, pool_size: int = 5):
        self._client = client
        self._model = model
        self._pool_size = pool_size
        self._available: asyncio.Queue = asyncio.Queue()
        self._created = 0
        self._lock = asyncio.Lock()

    async def _create_session(self) -> Any:
        """Creates a new session."""
        return await self._client.create_session({"model": self._model})

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Any]:
        """Acquires a session from pool, creates if needed."""
        session = None

        # Try to get existing session or create new one
        async with self._lock:
            if not self._available.empty():
                session = self._available.get_nowait()
            elif self._created < self._pool_size:
                session = await self._create_session()
                self._created += 1

        # If pool exhausted, wait for available session
        if session is None:
            session = await self._available.get()

        try:
            yield session
        finally:
            await self._available.put(session)

    async def close(self) -> None:
        """Destroys all sessions in the pool."""
        while not self._available.empty():
            session = self._available.get_nowait()
            try:
                await session.destroy()
            except Exception:
                pass
        self._created = 0


class ChatCopilot(BaseChatModel):
    """LangChain adapter for GitHub Copilot SDK.

    Uses session pool for concurrent requests - each request gets its own session.
    """

    model: str = "gpt-4.1"
    cli_path: Optional[str] = None
    cli_url: Optional[str] = None
    timeout: int = 300
    pool_size: int = 5

    _client: Any = None
    _pool: Optional[SessionPool] = None
    _started: bool = False
    _init_lock: Optional[asyncio.Lock] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "copilot"

    def _get_init_lock(self) -> asyncio.Lock:
        """Returns init lock, creating lazily in current event loop."""
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        return self._init_lock

    async def _ensure_started(self) -> None:
        """Ensures client and pool are initialized. Thread-safe."""
        if self._started:
            return

        async with self._get_init_lock():
            if self._started:
                return

            opts = {}
            if self.cli_path:
                opts["cli_path"] = self.cli_path
            if self.cli_url:
                opts["cli_url"] = self.cli_url

            self._client = CopilotClient(opts) if opts else CopilotClient()

            cli_path = self._client.options.get("cli_path", "")
            if cli_path:
                ensure_executable(cli_path)

            await self._client.start()
            self._pool = SessionPool(self._client, self.model, self.pool_size)
            self._started = True

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation using session from pool."""
        await self._ensure_started()

        prompt = messages_to_prompt(messages)
        async with self._pool.acquire() as session:
            response = await session.send_and_wait(
                {"prompt": prompt}, timeout=self.timeout
            )

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
        """Cleanup pool and client resources."""
        if self._pool:
            await self._pool.close()
            self._pool = None
        if self._client:
            await self._client.stop()
            self._client = None
        self._started = False
