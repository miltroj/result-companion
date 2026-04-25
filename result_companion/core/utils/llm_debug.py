from __future__ import annotations

from pathlib import Path
from typing import Callable

_writer: Callable[[str], None] | None = None


def enable_llm_debug(path: Path) -> None:
    """Activates LLM prompt/response logging to path.

    Args:
        path: File to append debug records to.
    """
    global _writer
    _writer = _make_writer(path)


def _make_writer(path: Path) -> Callable[[str], None]:
    def write(text: str) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)

    return write


def _format_llm_record(label: str, prompt: str, response: str) -> str:
    """Formats a single LLM prompt/response pair as a debug record."""
    return (
        f"\n{'='*60}\n{label}\n"
        f"--- PROMPT ---\n{prompt}\n"
        f"--- RESPONSE ---\n{response}\n"
    )


def is_llm_debug_enabled() -> bool:
    """Returns True if LLM debug logging is active."""
    return _writer is not None


def write_llm_record(label: str, prompt: str, response: str) -> None:
    """Appends a prompt/response record to the debug file.

    Args:
        label: Record header (e.g. '[Test Name] Chunk 1/3').
        prompt: The formatted prompt sent to the LLM.
        response: The LLM response content.
    """
    _writer(_format_llm_record(label, prompt, response))  # type: ignore[misc]
