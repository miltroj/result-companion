"""Public programmatic API for Result Companion."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional, Sequence

from result_companion._internal.analysis_helpers import run_provider_init_strategies
from result_companion.core.analizers.factory_common import execute_llm_and_get_results
from result_companion.core.chunking.chunking import ChunkingStrategy
from result_companion.core.parsers.config import DefaultConfigModel
from result_companion.core.plugins.base import (
    AnalysisResults,
    ParseOptions,
    ResultParserPlugin,
)
from result_companion.core.plugins.registry import load_results
from result_companion.core.results.analysis_result import AnalysisResult
from result_companion.core.results.text_report import summarize_failures_with_llm
from result_companion.core.review.pr_reviewer import review  # noqa: F401
from result_companion.core.utils.logging_config import set_global_log_level


async def run_analysis(
    config: DefaultConfigModel,
    results: AnalysisResults,
    summarize_failures: bool = False,
    dryrun: bool = False,
    quiet: bool = True,
) -> AnalysisResult:
    """Runs LLM analysis on configured parsed results.

    Args:
        config: Loaded configuration object.
        results: Configured results with chunking strategy set.
        summarize_failures: Whether to generate an overall failure summary.
        dryrun: If True, skip LLM calls.
        quiet: If True, suppress logs and progress output.

    Returns:
        AnalysisResult with llm_results, test_names, and optional summary.
    """
    run_provider_init_strategies(model_name=config.llm_factory.model)

    llm_results = await execute_llm_and_get_results(
        results=results,
        config=config,
        dryrun=dryrun,
        quiet=quiet,
    )

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
        test_names=list(llm_results.keys()),
        summary=summary,
    )


def analyze(
    output: str | Path | AnalysisResults,
    config: DefaultConfigModel,
    include_passing: bool = False,
    include_tags: Optional[list[str]] = None,
    exclude_tags: Optional[list[str]] = None,
    summarize_failures: bool = False,
    dryrun: bool = False,
    quiet: bool = True,
    result_format: str | None = None,
    plugins: Sequence[ResultParserPlugin] | None = None,
) -> AnalysisResult:
    """Main programmatic entry point for Result Companion.

    Accepts a result artifact path or pre-configured parsed results.

    Args:
        output: Path to a result artifact, or pre-configured parsed results.
        config: Loaded configuration object.
        include_passing: Whether to include passing tests (path mode only).
        include_tags: Tag patterns to include (path mode only).
        exclude_tags: Tag patterns to exclude (path mode only).
        summarize_failures: Whether to generate an overall failure summary.
        dryrun: If True, skip LLM calls.
        quiet: If True, suppress logs and progress output.
        result_format: Optional parser plugin name. Auto-detects when omitted.
        plugins: Optional parser plugins to use instead of built-ins.

    Returns:
        AnalysisResult with llm_results, test_names, and optional summary.
    """
    if quiet:
        set_global_log_level("ERROR")

    if isinstance(output, (str, Path)):
        results = load_results(
            path=Path(output),
            format_name=result_format,
            options=ParseOptions(
                include_tags=include_tags,
                exclude_tags=exclude_tags,
                exclude_fields=config.rendering.exclude_fields or None,
                exclude_passing=not include_passing,
            ),
            plugins=plugins,
        )
    else:
        results = output

    if not results.has_chunking:
        strategy = ChunkingStrategy(
            tokenizer_config=config.tokenizer,
            system_prompt=config.llm_config.question_prompt,
        )
        results.set_chunking(strategy)

    return asyncio.run(
        run_analysis(
            config=config,
            results=results,
            summarize_failures=summarize_failures,
            dryrun=dryrun,
            quiet=quiet,
        )
    )
