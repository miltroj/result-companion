from pathlib import Path

RC_USER_DIR = Path.home() / "result-companion"

BUNDLED_DEFAULT_CONFIG = (
    Path(__file__).resolve().parent / "configs" / "default_config.yaml"
)

USER_CONFIG = RC_USER_DIR / "config.yaml"


def resolve_user_config(cli_config: Path | None = None) -> Path | None:
    """Resolves which user config to use for merging on top of bundled defaults.

    Priority: explicit --config flag > ~/result-companion/config.yaml > None.

    Args:
        cli_config: Config path passed via CLI flag.

    Returns:
        Path to user config, or None if no overrides exist.
    """
    if cli_config:
        return cli_config

    if USER_CONFIG.exists():
        return USER_CONFIG

    return None
