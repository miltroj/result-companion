"""Shared GitHub Copilot client lifecycle helpers."""

import asyncio
import os
import shutil
import stat

from copilot import CopilotClient

from result_companion.core.utils.logging_config import get_progress_logger

logger = get_progress_logger("COPILOT")


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


def resolve_copilot_cli_path(
    cli_path: str | None = None,
    cli_url: str | None = None,
) -> str | None:
    """Resolves an explicit Copilot CLI path.

    Args:
        cli_path: Optional explicit CLI path or executable name.
        cli_url: Optional CLI server URL. When set, local CLI resolution is skipped.

    Returns:
        Resolved CLI path, or None when the SDK should use its bundled binary.

    Raises:
        FileNotFoundError: If an explicit CLI path cannot be resolved.
    """
    if cli_url:
        return None

    resolved_input = cli_path or os.getenv("COPILOT_CLI_PATH")
    if not resolved_input:
        return None

    resolved_path = shutil.which(resolved_input)
    if resolved_path:
        return resolved_path

    if os.path.isabs(resolved_input) and os.path.isfile(resolved_input):
        return resolved_input

    raise FileNotFoundError(
        "Copilot CLI not found at "
        f"'{resolved_input}'. Check COPILOT_CLI_PATH or cli_path."
    )


def build_copilot_client_options(
    cli_path: str | None = None,
    cli_url: str | None = None,
) -> dict[str, str]:
    """Builds Copilot client options from explicit and environment settings.

    Args:
        cli_path: Optional CLI path or executable name.
        cli_url: Optional Copilot CLI server URL.

    Returns:
        Options dict for CopilotClient.
    """
    options: dict[str, str] = {}
    resolved_cli = resolve_copilot_cli_path(cli_path=cli_path, cli_url=cli_url)

    if cli_path:
        options["cli_path"] = cli_path
    elif resolved_cli:
        options["cli_path"] = resolved_cli

    if cli_url:
        options["cli_url"] = cli_url

    return options


async def check_copilot_auth(client: CopilotClient) -> None:
    """Raises if the Copilot CLI is not authenticated.

    Args:
        client: Started Copilot client.
    """
    auth = await client.get_auth_status()
    if not auth.isAuthenticated:
        raise RuntimeError('Copilot CLI is not authenticated. Run: copilot -i "/login"')
    logger.debug(f"Copilot authenticated as {auth.login}")


async def log_copilot_diagnostics(client: CopilotClient) -> None:
    """Logs Copilot version and available models.

    Args:
        client: Started Copilot client.
    """
    try:
        status = await client.get_status()
        logger.debug(
            f"Copilot CLI v{status.version} (protocol {status.protocolVersion})",
        )
        models = await client.list_models()
        logger.debug(f"Available models: {[model.id for model in models]}")
    except Exception as exc:
        logger.warning(f"Could not fetch Copilot diagnostics: {exc}")


async def stop_copilot_client(client: CopilotClient) -> None:
    """Stops a Copilot client, suppressing shutdown errors.

    Args:
        client: Copilot client to stop.
    """
    try:
        await client.stop()
    except Exception as exc:
        logger.debug(f"Failed to stop Copilot client: {exc}")


async def start_copilot_client(
    client: CopilotClient,
    startup_timeout: float = 30.0,
) -> None:
    """Starts a Copilot client and validates authentication.

    Args:
        client: Copilot client to start.
        startup_timeout: Max seconds to wait for startup.

    Raises:
        RuntimeError: If startup times out or the CLI is unauthenticated.
    """
    cli_path = client.options.get("cli_path", "")
    ensure_executable(cli_path)

    try:
        await asyncio.wait_for(client.start(), timeout=startup_timeout)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            "Copilot CLI failed to start within "
            f"{startup_timeout:.0f}s. "
            'Try: copilot -i "/login"'
        ) from exc

    try:
        await check_copilot_auth(client)
    except Exception:
        await stop_copilot_client(client)
        raise

    await log_copilot_diagnostics(client)
