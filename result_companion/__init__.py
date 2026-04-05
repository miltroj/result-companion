"""Result Companion - AI-powered Robot Framework test analysis."""

try:
    from importlib.metadata import version

    __version__ = version("result-companion")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"

from result_companion.api import analyze, review, run_analysis  # noqa: E402, F401
from result_companion.core.state_dir import (  # noqa: E402, F401
    APP,
    CLI,
    get_or_create_current_user_rc_state_dir,
    get_or_create_rc_state_subdirectory,
)
