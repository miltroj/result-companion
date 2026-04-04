"""Result Companion - AI-powered Robot Framework test analysis."""

try:
    from importlib.metadata import version

    __version__ = version("result-companion")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"

from result_companion.api import analyze, review, run_analysis  # noqa: E402, F401
