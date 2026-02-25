from typing import Optional

from result_companion.core.analizers.factory_common import _build_llm_params
from result_companion.core.analizers.llm_router import _smart_acompletion
from result_companion.core.parsers.config import DefaultConfigModel
from result_companion.core.utils.logging_config import logger


def render_text_report(
    llm_results: dict[str, str],
    failed_test_names: list[str],
    overall_summary: Optional[str],
) -> str:
    """Builds concise plain-text report from LLM per-test results.

    Args:
        llm_results: Mapping of test names to LLM analysis.
        failed_test_names: Names of failed tests in current run.
        overall_summary: Optional synthesized summary.

    Returns:
        Text report content for file or console output.
    """
    lines = [
        "Result Companion - Failure Summary",
        f"Failed tests analyzed: {len(failed_test_names)}",
    ]

    if overall_summary:
        lines.extend(
            [
                "",
                "Overall Summary:",
                overall_summary.strip(),
            ]
        )

    if failed_test_names:
        lines.extend(["", "Failed Tests:"])
        for test_name in failed_test_names:
            lines.append(f"- {test_name}")

    if llm_results:
        lines.extend(["", "PER-TEST-ANALYSIS:"])
        for test_name in failed_test_names:
            result = llm_results.get(test_name)
            if not result:
                continue
            lines.extend(["", result.strip()])

    return "\n".join(lines) + "\n"


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
    model_name: str,
    config: DefaultConfigModel,
) -> Optional[str]:
    """Generates concise overall summary from per-test LLM results.

    Args:
        llm_results: Mapping of failed test names to analyses.
        model_name: Model identifier to use for synthesis.
        config: Parsed configuration with LLM params and prompt templates.

    Returns:
        Generated summary text, or None when unavailable.
    """
    if not llm_results:
        return None

    try:
        llm_params = _build_llm_params(config.llm_factory)
        llm_params["model"] = model_name
        prompt_template = config.llm_config.summary_prompt_template
        prompt = _build_overall_summary_prompt(llm_results, prompt_template)
        messages = [{"role": "user", "content": prompt}]
        response = await _smart_acompletion(messages=messages, **llm_params)
        return response.choices[0].message.content
    except Exception:
        logger.warning("Failed to generate overall summary", exc_info=True)
        return None
