from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

TAG_FILTERS = "tags"


@dataclass(frozen=True)
class ParseOptions:
    """Options shared by result parser plugins."""

    include_tags: list[str] | None = None
    exclude_tags: list[str] | None = None
    exclude_fields: list[str] | None = None
    exclude_passing: bool = True


class AnalysisResults(Protocol):
    """Parsed result object consumed by the analysis pipeline."""

    test_names: list[str]
    total_test_count: int
    source_hash: str
    has_chunking: bool

    def set_chunking(self, strategy: Any) -> "AnalysisResults":
        """Attaches a chunking strategy."""

    def render_chunks(self) -> Iterator[tuple[str, list[str], Any, str]]:
        """Yields chunked test payloads for LLM analysis."""


class ResultParserPlugin(Protocol):
    """Protocol for built-in result parser plugins."""

    name: str
    capabilities: frozenset[str]

    def can_parse(self, path: Path) -> bool:
        """Returns True when plugin can parse the given artifact."""

    def parse(self, path: Path, options: ParseOptions) -> AnalysisResults:
        """Parses an artifact into Result Companion's current result model."""
