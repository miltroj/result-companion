"""Public programmatic API for Result Companion."""

import asyncio
from pathlib import Path
from typing import Optional

from result_companion.core.analizers.factory_common import execute_llm_and_get_results
from result_companion.core.parsers.config import load_config
from result_companion.core.parsers.result_parser import (
    get_robot_results_from_file_as_dict,
)
from result_companion.core.results.analysis_result import AnalysisResult
from result_companion.core.results.text_report import summarize_failures_with_llm
from result_companion.core.utils.log_levels import LogLevels
from result_companion.core.utils.logging_config import set_global_log_level
from result_companion.entrypoints.run_rc import _run_provider_init_strategies


async def _analyze(
    output: Path,
    config: Optional[Path] = None,
    include_passing: bool = False,
    include_tags: Optional[list[str]] = None,
    exclude_tags: Optional[list[str]] = None,
    summarize_failures: bool = False,
    test_case_concurrency: Optional[int] = None,
    chunk_concurrency: Optional[int] = None,
    dryrun: bool = False,
    quiet: bool = True,
) -> AnalysisResult:
    """Runs LLM analysis on Robot Framework output.xml and returns structured results.

    Args:
        output: Path to Robot Framework output.xml.
        config: Optional path to YAML config file.
        include_passing: Whether to include passing tests.
        include_tags: RF tag patterns to include.
        exclude_tags: RF tag patterns to exclude.
        summarize_failures: Whether to generate an overall failure summary.
        test_case_concurrency: Override config test-case parallelism.
        chunk_concurrency: Override config chunk parallelism.
        dryrun: If True, skip LLM calls.
        quiet: If True, suppress logs and progress output.

    Returns:
        AnalysisResult with llm_results, test_names, and optional summary.
    """
    if quiet:
        set_global_log_level("ERROR")

    parsed_config = load_config(config)

    if test_case_concurrency is not None:
        parsed_config.concurrency.test_case = test_case_concurrency
    if chunk_concurrency is not None:
        parsed_config.concurrency.chunk = chunk_concurrency

    final_include = include_tags or parsed_config.test_filter.include_tags or None
    final_exclude = exclude_tags or parsed_config.test_filter.exclude_tags or None

    test_cases = get_robot_results_from_file_as_dict(
        file_path=output,
        log_level=LogLevels.DEBUG,
        include_tags=final_include,
        exclude_tags=final_exclude,
    )

    should_include_passing = (
        include_passing or parsed_config.test_filter.include_passing
    )
    if not should_include_passing:
        test_cases = [t for t in test_cases if t.get("status") != "PASS"]

    _run_provider_init_strategies(model_name=parsed_config.llm_factory.model)

    llm_results = await execute_llm_and_get_results(
        test_cases=test_cases,
        config=parsed_config,
        dryrun=dryrun,
        quiet=quiet,
    )

    test_names = [t["name"] for t in test_cases]

    summary = None
    if summarize_failures and llm_results and not dryrun:
        summary = await summarize_failures_with_llm(
            llm_results=llm_results,
            config=parsed_config,
        )

    await asyncio.sleep(0.25)

    return AnalysisResult(
        llm_results=llm_results,
        test_names=test_names,
        summary=summary,
    )


def analyze(
    output: str | Path,
    config: Optional[str | Path] = None,
    **kwargs,
) -> AnalysisResult:
    """Analyzes Robot Framework results with LLM assistance.

    Args:
        output: Path to Robot Framework output.xml.
        config: Optional path to YAML config file.
        **kwargs: Additional options forwarded to _analyze
            (include_passing, include_tags, exclude_tags,
            summarize_failures, test_case_concurrency,
            chunk_concurrency, dryrun, quiet).

    Returns:
        AnalysisResult with llm_results, test_names, and optional summary.
    """
    return asyncio.run(
        _analyze(
            output=Path(output),
            config=Path(config) if config else None,
            **kwargs,
        )
    )
