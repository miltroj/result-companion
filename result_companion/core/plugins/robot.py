from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from result_companion.core.chunking.rf_results import (
    ContextAwareRobotResults,
    get_rc_robot_results,
)
from result_companion.core.html.html_creator import create_llm_html_log
from result_companion.core.plugins.base import ParseOptions


class RobotPlugin:
    """Parses Robot Framework output.xml artifacts."""

    name = "robot"
    supports_tag_filters = True

    def can_parse(self, path: Path) -> bool:
        """Returns True when the XML root looks like Robot Framework output."""
        try:
            with path.open("rb") as source:
                _event, root = next(ET.iterparse(source, events=("start",)))
                return root.tag == "robot"
        except (ET.ParseError, StopIteration, OSError):
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

    def render_html_report(
        self,
        input_path: Path,
        output_path: Path,
        llm_results: dict[str, str],
        model_info: dict[str, str] | None = None,
        overall_summary: str | None = None,
    ) -> None:
        """Renders Robot Framework HTML log with embedded LLM results.

        Args:
            input_path: Robot Framework output.xml path.
            output_path: HTML log output path.
            llm_results: Per-test LLM analysis results.
            model_info: Optional LLM model metadata.
            overall_summary: Optional synthesized summary.
        """
        create_llm_html_log(
            input_result_path=input_path,
            llm_output_path=output_path,
            llm_results=llm_results,
            model_info=model_info,
            overall_summary=overall_summary,
        )
