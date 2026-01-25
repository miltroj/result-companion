"""Result Companion - AI-powered Robot Framework test analysis."""

try:
    from importlib.metadata import version

    __version__ = version("result-companion")
except Exception:
    __version__ = "0.0.0"
