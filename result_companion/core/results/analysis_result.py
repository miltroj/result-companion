from dataclasses import dataclass, field
from typing import Optional

from result_companion.core.results.text_report import render_text_report


@dataclass(frozen=True)
class AnalysisResult:
    """Container for programmatic analysis results."""

    llm_results: dict[str, str] = field(default_factory=dict)
    test_names: list[str] = field(default_factory=list)
    summary: Optional[str] = None

    @property
    def text_report(self) -> str:
        """Renders a plain-text report from the stored results."""

        return render_text_report(
            llm_results=self.llm_results,
            analyzed_test_names=self.test_names,
            overall_summary=self.summary,
        )
