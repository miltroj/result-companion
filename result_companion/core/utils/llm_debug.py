"""Injectable LLM debug logger — replaces global state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LLMDebugLogger:
    """Logs LLM prompts and responses to a file for debugging.

    Args:
        path: File to append debug records to. None = disabled (no-op).
    """

    path: Path | None = None

    @classmethod
    def from_path(cls, path: Path) -> LLMDebugLogger:
        """Creates an enabled logger writing to path."""
        return cls(path=path)

    @property
    def enabled(self) -> bool:
        """True if debug logging is active."""
        return self.path is not None

    def write_record(self, label: str, prompt: str, response: str) -> None:
        """Appends a prompt/response record to the debug file.

        No-op when logger is disabled (path is None).

        Args:
            label: Record header (e.g. '[Test Name] Chunk 1/3').
            prompt: The formatted prompt sent to the LLM.
            response: The LLM response content.
        """
        if self.path is None:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(_format_record(label, prompt, response))


def _format_record(label: str, prompt: str, response: str) -> str:
    """Formats a single LLM prompt/response pair as a debug record."""
    return (
        f"\n{'='*60}\n{label}\n"
        f"--- PROMPT ---\n{prompt}\n"
        f"--- RESPONSE ---\n{response}\n"
    )
