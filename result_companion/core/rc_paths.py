import os
import shutil
from pathlib import Path

from result_companion.core.utils.logging_config import logger

RC_USER_DIR = Path.home() / "result-companion"

BUNDLED_DEFAULT_CONFIG = (
    Path(__file__).resolve().parent / "configs" / "default_config.yaml"
)


def _is_running_as_root() -> bool:
    """Checks if the process is running with elevated privileges."""
    return os.getuid() == 0


def ensure_default_config() -> Path:
    """Ensures default_config.yaml exists in the RC user directory.

    On first run, creates ~/result-companion/ and copies the bundled
    default_config.yaml there. Subsequent runs use the existing copy.
    Falls back to bundled config when running as root to avoid
    creating root-owned files in a normal user's home directory.

    Returns:
        Path to the default config file in the user directory.
    """
    user_config = RC_USER_DIR / "default_config.yaml"
    if user_config.exists():
        return user_config

    if _is_running_as_root():
        logger.warning(
            "Running as root — using bundled config to avoid permission issues"
        )
        return BUNDLED_DEFAULT_CONFIG

    RC_USER_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUNDLED_DEFAULT_CONFIG, user_config)
    logger.info(f"Created default config at {user_config}")
    return user_config
