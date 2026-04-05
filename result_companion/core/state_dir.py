from pathlib import Path

_STATE_DIR_NAME = ".result-companion"

CLI = "cli"
APP = "app"


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


def get_or_create_rc_state_subdirectory(*parts: str, home: Path | None = None) -> Path:
    """Returns a directory under ``~/.result-companion``, creating it if missing.

    Same root as ``get_or_create_current_user_rc_state_dir``; ``parts`` are extra
    folders (e.g. ``CLI``, or ``CLI`` + ``"data"``). Do not pass file names—use
    ``get_or_create_rc_state_subdirectory(CLI) / "config.yaml"`` for files.

    Args:
        *parts: Directory segments relative to the RC state root.
        home: Optional home directory for tests.

    Returns:
        Absolute path to that directory (the state root if ``parts`` is empty).
    """
    base = get_or_create_current_user_rc_state_dir(home=home)
    if not parts:
        return base
    path = base.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
