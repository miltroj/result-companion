from pathlib import Path

_STATE_DIR_NAME = ".result-companion"


def get_or_create_current_user_rc_state_dir(home: Path | None = None) -> Path:
    """Returns the durable user state directory, creating it on first use.

    Use this path for configs and artifacts so they survive pip installs and
    updates. Extension packages depending on result-companion should import this
    function rather than hardcoding a location.

    Args:
        home: Optional home directory; defaults to ``Path.home()`` for normal use.

    Returns:
        Absolute path to the state directory (e.g. ``~/.result-companion``).
    """
    root = home if home is not None else Path.home()
    path = root / _STATE_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path
