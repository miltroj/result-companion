from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Protocol

from result_companion.core.chunking.chunking import ChunkingStrategy
from result_companion.core.chunking.utils import Chunking


@dataclass(frozen=True)
class ParseOptions:
    """Options shared by result parser plugins."""

    include_tags: list[str] | None = None
    exclude_tags: list[str] | None = None
    exclude_fields: list[str] | None = None
    exclude_passing: bool = True


class ParsedResults(Protocol):
    """Parsed result object consumed by the analysis pipeline."""

    test_names: list[str]
    total_test_count: int
    source_hash: str
    has_chunking: bool

    def set_chunking(self, strategy: ChunkingStrategy) -> "ParsedResults":
        """Attaches a chunking strategy."""

    def render_chunks(self) -> Iterator["TestChunkPayload"]:
        """Yields chunked test payloads for LLM analysis."""


class TestChunkPayload(NamedTuple):
    """Chunked LLM payload for one test or analysis unit."""

    test_name: str
    chunks: list[str]
    chunk_stats: Chunking
    status: str


class ResultParserPlugin(Protocol):
    """Protocol for result parser plugins.

    Optional capabilities (duck-typed):
    - ``supports_tag_filters = True`` to honor include_tags/exclude_tags.
    - ``render_html_report(...)`` method to enable HTML report output.
    """

    name: str

    def can_parse(self, path: Path) -> bool:
        """Returns True when plugin can parse the given artifact."""

    def parse(self, path: Path, options: ParseOptions) -> ParsedResults:
        """Parses an artifact into Result Companion's current result model."""
