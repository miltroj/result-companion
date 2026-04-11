"""Public programmatic API for Result Companion."""

import asyncio
from pathlib import Path
from typing import Optional

from result_companion._internal.analysis_helpers import (
    build_chunkable,
    run_provider_init_strategies,
)
from result_companion.core.analizers.factory_common import execute_llm_and_get_results
from result_companion.core.chunking.rf_chunker import ChunkableResult
from result_companion.core.parsers.config import DefaultConfigModel
from result_companion.core.results.analysis_result import AnalysisResult
from result_companion.core.results.text_report import summarize_failures_with_llm
from result_companion.core.review.pr_reviewer import review  # noqa: F401
from result_companion.core.utils.logging_config import set_global_log_level


async def run_analysis(
    config: DefaultConfigModel,
    chunkable: ChunkableResult,
    include_passing: bool = False,
    summarize_failures: bool = False,
    dryrun: bool = False,
    quiet: bool = True,
) -> AnalysisResult:
    """Runs LLM analysis on tests from ChunkableResult with a loaded config.

    Args:
        config: Loaded configuration object.
        chunkable: ChunkableResult wrapping the RF output.
        include_passing: Whether to include PASS tests.
        summarize_failures: Whether to generate an overall failure summary.
        dryrun: If True, skip LLM calls.
        quiet: If True, suppress logs and progress output.

    Returns:
        AnalysisResult with llm_results, test_names, and optional summary.
    """
    run_provider_init_strategies(model_name=config.llm_factory.model)

    llm_results = await execute_llm_and_get_results(
        chunkable=chunkable,
        config=config,
        include_passing=include_passing,
        dryrun=dryrun,
        quiet=quiet,
    )

    test_names = list(llm_results.keys())

    summary = None
    if summarize_failures and llm_results and not dryrun:
        summary = await summarize_failures_with_llm(
            llm_results=llm_results,
            config=config,
        )

    # Allow aiohttp SSL connections to cleanup before event loop closes
    await asyncio.sleep(0.25)

    return AnalysisResult(
        llm_results=llm_results,
        test_names=test_names,
        summary=summary,
    )


def analyze(
    output: str | Path,
    config: DefaultConfigModel,
    include_passing: bool = False,
    include_tags: Optional[list[str]] = None,
    exclude_tags: Optional[list[str]] = None,
    summarize_failures: bool = False,
    dryrun: bool = False,
    quiet: bool = True,
) -> AnalysisResult:
    """Main programmatic entry point for Result Companion.

    Args:
        output: Path to RF output.xml.
        config: Loaded configuration object.
        include_passing: Whether to include passing tests.
        include_tags: RF tag patterns to include.
        exclude_tags: RF tag patterns to exclude.
        summarize_failures: Whether to generate an overall failure summary.
        dryrun: If True, skip LLM calls.
        quiet: If True, suppress logs and progress output.

    Returns:
        AnalysisResult with llm_results, test_names, and optional summary.
    """
    if quiet:
        set_global_log_level("ERROR")

    chunkable = build_chunkable(
        output=Path(output),
        config=config,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
    )

    return asyncio.run(
        run_analysis(
            config=config,
            chunkable=chunkable,
            include_passing=include_passing,
            summarize_failures=summarize_failures,
            dryrun=dryrun,
            quiet=quiet,
        )
    )
