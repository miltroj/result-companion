"""GitHub Copilot initialization strategy for LiteLLM."""

from pathlib import Path

from result_companion.core.utils.logging_config import logger

DEFAULT_TOKEN_DIR = Path.home() / ".config" / "litellm" / "github_copilot"


def check_copilot_tokens_exist(token_dir: Path = DEFAULT_TOKEN_DIR) -> bool:
    """Checks if Copilot OAuth tokens exist locally.

    Args:
        token_dir: Directory where LiteLLM stores Copilot tokens.

    Returns:
        True if tokens exist, False otherwise.
    """
    if not token_dir.exists():
        return False

    # LiteLLM stores access-token and api-key.json
    access_token = token_dir / "access-token"
    api_key = token_dir / "api-key.json"

    return access_token.exists() or api_key.exists()


def copilot_on_init_strategy(token_dir: Path = DEFAULT_TOKEN_DIR) -> None:
    """Initializes GitHub Copilot provider.

    LiteLLM handles OAuth device flow automatically on first use.
    This function logs whether authentication will be needed.

    Args:
        token_dir: Directory where LiteLLM stores Copilot tokens.
    """
    if check_copilot_tokens_exist(token_dir):
        logger.debug("GitHub Copilot tokens found - authentication ready")
        return

    logger.info(
        "GitHub Copilot authentication required. "
        "LiteLLM will prompt you to authenticate via GitHub on first request."
    )
