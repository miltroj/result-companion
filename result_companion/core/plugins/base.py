from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from result_companion.core.chunking.rf_results import ContextAwareRobotResults

TAG_FILTERS = "tags"


@dataclass(frozen=True)
class ParseOptions:
    """Options shared by result parser plugins."""

    include_tags: list[str] | None = None
    exclude_tags: list[str] | None = None
    exclude_fields: list[str] | None = None
    exclude_passing: bool = True


class ResultParserPlugin(Protocol):
    """Protocol for built-in result parser plugins."""

    name: str
    capabilities: frozenset[str]

    def can_parse(self, path: Path) -> bool:
        """Returns True when plugin can parse the given artifact."""

    def parse(self, path: Path, options: ParseOptions) -> ContextAwareRobotResults:
        """Parses an artifact into Result Companion's current result model."""
