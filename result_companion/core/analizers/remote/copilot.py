"""LiteLLM adapter for GitHub Copilot SDK."""

import asyncio
import logging
import os
import stat
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

import litellm
from copilot import CopilotClient
from litellm import CustomLLM
from litellm.types.utils import ModelResponse

logger = logging.getLogger(__name__)


def messages_to_prompt(messages: list[dict[str, str]]) -> str:
    """Converts LiteLLM messages to prompt string.

    Args:
        messages: List of message dicts with role and content.

    Returns:
        Single formatted prompt string.
    """
    prefixes = {"system": "[System]", "user": "[User]", "assistant": "[Assistant]"}
    parts = [f"{prefixes.get(m['role'], '')}: {m['content']}" for m in messages]
    return "\n\n".join(parts)


def ensure_executable(path: str) -> None:
    """Adds execute permission to a binary if missing.

    Args:
        path: Absolute path to the binary file.
    """
    if not path or not os.path.isabs(path) or not os.path.isfile(path):
        return
    if os.access(path, os.X_OK):
        return
    current_mode = os.stat(path).st_mode
    os.chmod(path, current_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    logger.info(f"Fixed execute permission on copilot binary: {path}")


class SessionPool:
    """Pool of Copilot sessions for concurrent requests."""

    def __init__(self, client: CopilotClient, model: str, pool_size: int = 5):
        self._client = client
        self._model = model
        self._pool_size = pool_size
        self._available: asyncio.Queue = asyncio.Queue()
        self._created = 0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Any]:
        """Acquires a session from pool, creates if needed."""
        session = None
        async with self._lock:
            if not self._available.empty():
                session = self._available.get_nowait()
            elif self._created < self._pool_size:
                session = await self._client.create_session({"model": self._model})
                self._created += 1

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


class CopilotLLM(CustomLLM):
    """LiteLLM custom provider for GitHub Copilot SDK."""

    def __init__(
        self,
        model: str = "gpt-4.1",
        pool_size: int = 5,
        timeout: int = 300,
        cli_path: Optional[str] = None,
        cli_url: Optional[str] = None,
    ):
        super().__init__()
        self._default_model = model
        self._pool_size = pool_size
        self._timeout = timeout
        self._cli_path = cli_path
        self._cli_url = cli_url
        self._client: Optional[CopilotClient] = None
        self._pool: Optional[SessionPool] = None
        self._started = False
        self._init_lock: Optional[asyncio.Lock] = None

    async def _ensure_started(self, model: str) -> None:
        """Ensures client and pool are initialized."""
        if self._started:
            return

        if self._init_lock is None:
            self._init_lock = asyncio.Lock()

        async with self._init_lock:
            if self._started:
                return

            opts = {}
            if self._cli_path:
                opts["cli_path"] = self._cli_path
            if self._cli_url:
                opts["cli_url"] = self._cli_url

            self._client = CopilotClient(opts) if opts else CopilotClient()
            cli_path = self._client.options.get("cli_path", "")
            ensure_executable(cli_path)
            await self._client.start()
            self._pool = SessionPool(self._client, model, self._pool_size)
            self._started = True

    def _extract_model(self, model: str) -> str:
        """Extracts model name from 'copilot/model-name' format."""
        if "/" in model:
            return model.split("/", 1)[1]
        return model or self._default_model

    async def acompletion(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Async completion using Copilot SDK.

        Args:
            model: Model identifier (e.g., copilot/gpt-4.1).
            messages: List of message dicts.
            **kwargs: Additional parameters (ignored).

        Returns:
            LiteLLM ModelResponse with completion.
        """
        actual_model = self._extract_model(model)
        await self._ensure_started(actual_model)

        prompt = messages_to_prompt(messages)
        async with self._pool.acquire() as session:
            response = await session.send_and_wait(
                {"prompt": prompt}, timeout=self._timeout
            )

        content = response.data.content if response and response.data else ""

        return litellm.ModelResponse(
            choices=[
                {
                    "message": {"role": "assistant", "content": content},
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            model=actual_model,
        )

    def completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Sync completion - wraps async implementation."""
        return asyncio.run(self.acompletion(model, messages, **kwargs))

    async def aclose(self) -> None:
        """Cleanup pool and client resources."""
        if self._pool:
            await self._pool.close()
            self._pool = None
        if self._client:
            await self._client.stop()
            self._client = None
        self._started = False


# Module-level instance for registration
_copilot_handler: Optional[CopilotLLM] = None


def register_copilot_provider(
    model: str = "gpt-4.1",
    pool_size: int = 5,
    timeout: int = 300,
    cli_path: Optional[str] = None,
    cli_url: Optional[str] = None,
) -> CopilotLLM:
    """Registers Copilot SDK as a LiteLLM provider.

    Uses 'copilot_sdk' prefix to avoid conflict with LiteLLM's built-in
    'copilot' provider (which uses the blocked API).

    Args:
        model: Default Copilot model.
        pool_size: Number of concurrent sessions.
        timeout: Request timeout in seconds.
        cli_path: Optional path to Copilot CLI.
        cli_url: Optional URL of existing CLI server.

    Returns:
        The registered handler instance.

    Example:
        register_copilot_provider()
        response = await litellm.acompletion(
            model="copilot_sdk/gpt-4.1",
            messages=[{"role": "user", "content": "Hello"}]
        )
    """
    global _copilot_handler
    _copilot_handler = CopilotLLM(
        model=model,
        pool_size=pool_size,
        timeout=timeout,
        cli_path=cli_path,
        cli_url=cli_url,
    )
    # Use 'copilot_sdk' to avoid conflict with built-in 'copilot' provider
    litellm.custom_provider_map = [
        {"provider": "copilot_sdk", "custom_handler": _copilot_handler}
    ]
    return _copilot_handler
