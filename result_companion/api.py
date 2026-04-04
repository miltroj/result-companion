"""Public programmatic API for Result Companion."""

import asyncio
from pathlib import Path
from typing import Optional

from result_companion._internal.analysis_helpers import (
    load_and_filter_test_cases,
    run_provider_init_strategies,
)
from result_companion.core.analizers.factory_common import execute_llm_and_get_results
from result_companion.core.parsers.config import DefaultConfigModel
from result_companion.core.results.analysis_result import AnalysisResult
from result_companion.core.results.text_report import summarize_failures_with_llm
from result_companion.core.review.pr_reviewer import review  # noqa: F401
from result_companion.core.utils.logging_config import set_global_log_level


async def run_analysis(
    config: DefaultConfigModel,
    test_cases: list[dict],
    summarize_failures: bool = False,
    dryrun: bool = False,
    quiet: bool = True,
) -> AnalysisResult:
    """Runs LLM analysis on pre-parsed test cases with a loaded config.

    This is the core programmatic entry point. It accepts already-loaded
    objects so callers are not forced to use file paths.

    Args:
        config: Loaded configuration object.
        test_cases: Parsed test case dicts (e.g. from get_robot_results_from_file_as_dict).
        summarize_failures: Whether to generate an overall failure summary.
        dryrun: If True, skip LLM calls.
        quiet: If True, suppress logs and progress output.

    Returns:
        AnalysisResult with llm_results, test_names, and optional summary.
    """
    run_provider_init_strategies(model_name=config.llm_factory.model)

    llm_results = await execute_llm_and_get_results(
        test_cases=test_cases,
        config=config,
        dryrun=dryrun,
        quiet=quiet,
    )

    test_names = [t["name"] for t in test_cases]

    summary = None
    if summarize_failures and llm_results and not dryrun:
        summary = await summarize_failures_with_llm(
            llm_results=llm_results,
            config=config,
        )

    # Allow aiohttp SSL connections to cleanup before event loop closes
    # This prevents "Event loop is closed" errors from liteLLM's internal aiohttp client
    await asyncio.sleep(0.25)

    return AnalysisResult(
        llm_results=llm_results,
        test_names=test_names,
        summary=summary,
    )


def analyze(
    output: str | Path | list[dict],
    config: DefaultConfigModel,
    include_passing: bool = False,
    include_tags: Optional[list[str]] = None,
    exclude_tags: Optional[list[str]] = None,
    summarize_failures: bool = False,
    dryrun: bool = False,
    quiet: bool = True,
) -> AnalysisResult:
    """Main programmatic entry point for Result Companion.

    Accepts either a path to output.xml (loads and filters test cases)
    or pre-parsed test cases as a list of dicts.

    Args:
        output: Path to RF output.xml, or pre-parsed test case dicts.
        config: Loaded configuration object.
        include_passing: Whether to include passing tests (path mode only).
        include_tags: RF tag patterns to include (path mode only).
        exclude_tags: RF tag patterns to exclude (path mode only).
        summarize_failures: Whether to generate an overall failure summary.
        dryrun: If True, skip LLM calls.
        quiet: If True, suppress logs and progress output.

    Returns:
        AnalysisResult with llm_results, test_names, and optional summary.
    """
    if quiet:
        set_global_log_level("ERROR")

    if isinstance(output, list):
        test_cases = output
    else:
        test_cases = load_and_filter_test_cases(
            output=Path(output),
            config=config,
            include_passing=include_passing,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
        )

    return asyncio.run(
        run_analysis(
            config=config,
            test_cases=test_cases,
            summarize_failures=summarize_failures,
            dryrun=dryrun,
            quiet=quiet,
        )
    )
