from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from result_companion.core.analizers.factory_common import _build_llm_params
from result_companion.core.analizers.llm_router import _smart_acompletion
from result_companion.core.parsers.config import DefaultConfigModel
from result_companion.core.utils.logging_config import logger


def compute_source_hash(test_cases: list[dict]) -> str:
    """Computes a short SHA-256 hash of raw test data before LLM processing."""
    blob = json.dumps(test_cases, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


@dataclass
class AnalyzeReport:
    """Structured output from result-companion analyze."""

    failed_test_count: int
    analyzed_tests: list[str]
    per_test_results: dict[str, str] = field(default_factory=dict)
    overall_summary: str | None = None
    model: str | None = None
    source_file: str | None = None
    total_test_count: int | None = None
    source_hash: str | None = None
    timestamp: str | None = None

    def to_json(self) -> str:
        """Serializes report to JSON string."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "AnalyzeReport":
        """Deserializes report from JSON string."""
        return cls(**json.loads(text))

    def to_text(self) -> str:
        """Renders report as plain text (same format as render_text_report)."""
        return render_text_report(
            self.per_test_results, self.analyzed_tests, self.overall_summary
        )

    def has_failures(self) -> bool:
        """Returns True if the report contains analyzed test failures."""
        return self.failed_test_count > 0


def render_text_report(
    llm_results: dict[str, str],
    analyzed_test_names: list[str],
    overall_summary: str | None,
) -> str:
    """Builds concise plain-text report from LLM per-test results.

    Args:
        llm_results: Mapping of test names to LLM analysis.
        analyzed_test_names: Names of tests included in current analysis.
        overall_summary: Optional synthesized summary.

    Returns:
        Text report content for file or console output.
    """
    lines = [
        "Result Companion - Summary",
        f"Tests analyzed: {len(analyzed_test_names)}",
    ]

    if overall_summary:
        lines.extend(
            [
                "",
                "Overall Summary:",
                overall_summary.strip(),
            ]
        )

    if analyzed_test_names:
        lines.extend(["", "Analyzed Tests:"])
        for test_name in analyzed_test_names:
            lines.append(f"- {test_name}")

    if llm_results:
        lines.extend(["", "PER-TEST-ANALYSIS:"])
        for test_name in analyzed_test_names:
            result = llm_results.get(test_name)
            if not result:
                continue
            lines.extend(["", result.strip()])

    return "\n".join(lines) + "\n"


def render_json_report(
    llm_results: dict[str, str],
    analyzed_test_names: list[str],
    overall_summary: str | None,
    model: str | None = None,
    source_file: str | None = None,
    total_test_count: int | None = None,
    source_hash: str | None = None,
) -> str:
    """Builds JSON report from LLM per-test results.

    Args:
        llm_results: Mapping of test names to LLM analysis.
        analyzed_test_names: Names of tests included in current analysis.
        overall_summary: Optional synthesized summary.
        model: LLM model used for analysis.
        source_file: Path to the input output.xml.
        total_test_count: Total tests before pass/fail filtering.
        source_hash: Short content hash of the source data.

    Returns:
        JSON string of the AnalyzeReport.
    """
    report = AnalyzeReport(
        failed_test_count=len(analyzed_test_names),
        analyzed_tests=analyzed_test_names,
        per_test_results=llm_results,
        overall_summary=overall_summary,
        model=model,
        source_file=source_file,
        total_test_count=total_test_count,
        source_hash=source_hash,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    return report.to_json()


def _build_overall_summary_prompt(
    llm_results: dict[str, str], prompt_template: str
) -> str:
    """Builds prompt for concise synthesis of all failed test analyses."""
    per_test_sections = []
    for test_name, result in llm_results.items():
        per_test_sections.append(f"### {test_name}\n{result.strip()}\n")

    analyses = "\n".join(per_test_sections)
    return prompt_template.format(analyses=analyses)


async def summarize_failures_with_llm(
    llm_results: dict[str, str],
    config: DefaultConfigModel,
) -> str | None:
    """Generates concise overall summary from per-test LLM results.

    Args:
        llm_results: Mapping of failed test names to analyses.
        config: Parsed configuration with LLM params and prompt templates.

    Returns:
        Generated summary text, or None when unavailable.
    """
    if not llm_results:
        return None

    try:
        llm_params = _build_llm_params(config.llm_factory)
        prompt_template = config.llm_config.summary_prompt_template
        prompt = _build_overall_summary_prompt(llm_results, prompt_template)
        messages = [{"role": "user", "content": prompt}]
        response = await _smart_acompletion(messages=messages, **llm_params)
        return response.choices[0].message.content
    except Exception:
        logger.warning("Failed to generate overall summary", exc_info=True)
        return None
