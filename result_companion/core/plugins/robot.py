from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from result_companion.core.chunking.rf_results import (
    ContextAwareRobotResults,
    get_rc_robot_results,
)
from result_companion.core.plugins.base import TAG_FILTERS, ParseOptions


class RobotPlugin:
    """Parses Robot Framework output.xml artifacts."""

    name = "robot"
    capabilities = frozenset({TAG_FILTERS})

    def can_parse(self, path: Path) -> bool:
        """Returns True when the XML root looks like Robot Framework output."""
        try:
            with path.open("rb") as source:
                for _event, root in ET.iterparse(source, events=("start",)):
                    return root.tag == "robot"
        except ET.ParseError:
            return False
        return False

    def parse(self, path: Path, options: ParseOptions) -> ContextAwareRobotResults:
        """Parses Robot Framework output XML with Result Companion filters."""
        return get_rc_robot_results(
            file_path=path,
            include_tags=options.include_tags,
            exclude_tags=options.exclude_tags,
            exclude_fields=options.exclude_fields,
            exclude_passing=options.exclude_passing,
        )
