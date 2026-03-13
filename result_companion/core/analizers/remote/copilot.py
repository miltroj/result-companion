"""LiteLLM adapter for GitHub Copilot SDK."""

import asyncio
import os
import shutil
import stat
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

import litellm
from copilot import CopilotClient, PermissionHandler
from litellm import CustomLLM
from litellm.types.utils import ModelResponse

from result_companion.core.utils.logging_config import get_progress_logger

logger = get_progress_logger("COPILOT")


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

    async def _discard_session(self, session: Any) -> None:
        """Destroys a failed session and frees its pool slot."""
        async with self._lock:
            self._created -= 1
        try:
            await session.destroy()
        except Exception as e:
            logger.debug(f"Failed to destroy session: {session} - {e}")

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Any]:
        """Acquires a session from pool, creates if needed."""
        session = None
        async with self._lock:
            if not self._available.empty():
                session = self._available.get_nowait()
            elif self._created < self._pool_size:
                session = await self._client.create_session(
                    {
                        "model": self._model,
                        "on_permission_request": PermissionHandler.approve_all,
                    }
                )
                self._created += 1

        if session is None:
            session = await self._available.get()

        failed = False
        try:
            yield session
        except Exception as e:
            failed = True
            logger.debug(f"Session failed during use: {session} - {e}")
            raise
        finally:
            if failed:
                await self._discard_session(session)
            else:
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
        model: str = "gpt-5-mini",
        pool_size: int = 5,
        timeout: int = 300,
        cli_path: Optional[str] = None,
        cli_url: Optional[str] = None,
        max_retries: int = 3,
        retry_base_delay: float = 10.0,
        startup_timeout: float = 30.0,
    ):
        super().__init__()
        self._default_model = model
        self._pool_size = pool_size
        self._timeout = timeout
        self._cli_path = cli_path
        self._cli_url = cli_url
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._startup_timeout = startup_timeout
        self._client: Optional[CopilotClient] = None
        self._pool: Optional[SessionPool] = None
        self._started = False
        self._has_succeeded = False
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

            resolved_cli = self._resolve_cli_path()
            opts = {}
            if self._cli_path:
                opts["cli_path"] = self._cli_path
            elif resolved_cli:
                opts["cli_path"] = resolved_cli
            if self._cli_url:
                opts["cli_url"] = self._cli_url

            client = CopilotClient(opts) if opts else CopilotClient()
            cli_path = client.options.get("cli_path", "")
            ensure_executable(cli_path)
            try:
                await asyncio.wait_for(client.start(), timeout=self._startup_timeout)
            except asyncio.TimeoutError:
                raise RuntimeError(
                    "Copilot CLI failed to start within "
                    f"{self._startup_timeout:.0f}s. "
                    'Try: copilot -i "/login"'
                )
            try:
                await self._check_auth(client)
                status = await client.get_status()
                logger.debug(
                    "Copilot CLI v%s (protocol %s)",
                    status.version,
                    status.protocolVersion,
                )
                models = await client.list_models()
                logger.debug("Available models: %s", [m.id for m in models])
            except Exception:
                await self._stop_client(client)
                raise

            self._client = client
            self._pool = SessionPool(self._client, model, self._pool_size)
            self._started = True

    async def _check_auth(self, client: CopilotClient) -> None:
        """Raises if the Copilot CLI is not authenticated."""
        auth = await client.get_auth_status()
        if not auth.isAuthenticated:
            raise RuntimeError(
                "Copilot CLI is not authenticated. Run: copilot auth login"
            )
        logger.debug(f"Copilot authenticated as {auth.login}")

    async def _stop_client(self, client: CopilotClient) -> None:
        """Stops a client, suppressing errors."""
        try:
            await client.stop()
        except Exception as exc:
            logger.debug(f"Failed to stop Copilot client: {exc}")

    def _resolve_cli_path(self) -> str | None:
        """Resolves an explicit Copilot CLI path, or None to use the SDK bundled binary."""
        if self._cli_url:
            return None
        cli_path = self._cli_path or os.getenv("COPILOT_CLI_PATH")
        if not cli_path:
            return None
        resolved = shutil.which(cli_path) or (
            cli_path if os.path.isabs(cli_path) and os.path.isfile(cli_path) else None
        )
        if not resolved:
            raise FileNotFoundError(
                f"Copilot CLI not found at '{cli_path}'. Check COPILOT_CLI_PATH or cli_path."
            )
        return resolved

    def _extract_model(self, model: str) -> str:
        """Extracts model name from 'copilot/model-name' format."""
        if "/" in model:
            return model.split("/", 1)[1]
        return model or self._default_model

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Checks if an exception is a Copilot rate limit error."""
        return "rate limit" in str(error).lower()

    async def _send_prompt(self, prompt: str) -> Any:
        """Sends a single prompt through the session pool.

        Args:
            prompt: The formatted prompt string.

        Returns:
            Copilot session response.
        """
        async with self._pool.acquire() as session:
            return await session.send_and_wait(
                {"prompt": prompt}, timeout=self._timeout
            )

    async def _handle_rate_limit(self, error: Exception, attempt: int) -> None:
        """Handles a rate limit error: fails fast or waits with backoff.

        Args:
            error: The caught rate limit exception.
            attempt: Current attempt number (0-based).

        Raises:
            RuntimeError: If no prior request ever succeeded.
        """
        if not self._has_succeeded:
            raise RuntimeError(
                "Rate limited on first request — likely exceeding Copilot's "
                "token limit. Reduce max_content_tokens in result-companion config."
            ) from error
        if attempt >= self._max_retries:
            return
        delay = self._retry_base_delay * (2**attempt)
        logger.warning(
            "Rate limited (attempt %d/%d), retrying in %.0fs...",
            attempt,
            self._max_retries,
            delay,
        )
        await asyncio.sleep(delay)

    async def _send_with_retry(self, prompt: str) -> Any:
        """Sends prompt with exponential backoff on rate limits.

        Args:
            prompt: The formatted prompt string.

        Returns:
            Copilot session response.

        Raises:
            RuntimeError: If rate limited on first-ever request (likely token limit).
            Exception: Re-raises last rate limit error after all retries exhausted.
        """
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._send_prompt(prompt)
                self._has_succeeded = True
                return response
            except Exception as e:
                if not self._is_rate_limit_error(e):
                    raise
                last_error = e
                await self._handle_rate_limit(e, attempt)
        raise last_error

    async def acompletion(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Async completion using Copilot SDK.

        Args:
            model: Model identifier (e.g., copilot/gpt-5-mini).
            messages: List of message dicts.
            **kwargs: Additional parameters (ignored).

        Returns:
            LiteLLM ModelResponse with completion.
        """
        actual_model = self._extract_model(model)
        await self._ensure_started(actual_model)

        prompt = messages_to_prompt(messages)
        response = await self._send_with_retry(prompt)

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
            prompt=prompt,
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
