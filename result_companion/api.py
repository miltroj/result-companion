"""Public programmatic API for Result Companion."""

import asyncio
from pathlib import Path
from typing import Optional

from result_companion.core.analizers.factory_common import execute_llm_and_get_results
from result_companion.core.analizers.local.ollama_runner import ollama_on_init_strategy
from result_companion.core.parsers.config import DefaultConfigModel, load_config
from result_companion.core.parsers.result_parser import (
    get_robot_results_from_file_as_dict,
)
from result_companion.core.results.analysis_result import AnalysisResult
from result_companion.core.results.text_report import summarize_failures_with_llm
from result_companion.core.utils.logging_config import logger, set_global_log_level


def _run_provider_init_strategies(model_name: str) -> None:
    """Runs provider-specific initialization based on LiteLLM model prefix.

    Args:
        model_name: LiteLLM model identifier (e.g., ollama_chat/llama2).
    """
    provider = model_name.split("/", 1)[0]
    strategies = {
        "ollama": lambda: _run_ollama_init_strategy(model_name),
        "ollama_chat": lambda: _run_ollama_init_strategy(model_name),
    }
    strategy = strategies.get(provider)
    if strategy:
        strategy()


def _run_ollama_init_strategy(model_name: str) -> None:
    """Runs Ollama initialization strategy if model is Ollama.

    Args:
        model_name: LiteLLM model identifier (e.g., ollama_chat/llama2).
    """
    if not model_name.startswith("ollama"):
        return

    parts = model_name.split("/")
    model_short = parts[1].split(":")[0] if len(parts) > 1 else model_name.split(":")[0]
    logger.debug(f"Running Ollama init strategy for model: {model_short}")
    ollama_on_init_strategy(model_name=model_short)


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
    _run_provider_init_strategies(model_name=config.llm_factory.model)

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


def _apply_concurrency_overrides(
    config: DefaultConfigModel,
    test_case_concurrency: Optional[int],
    chunk_concurrency: Optional[int],
) -> None:
    """Mutates config concurrency settings with CLI/caller overrides."""
    if test_case_concurrency is not None:
        config.concurrency.test_case = test_case_concurrency
    if chunk_concurrency is not None:
        config.concurrency.chunk = chunk_concurrency


def _resolve_tags(
    caller_tags: Optional[list[str]],
    config_tags: list[str],
) -> Optional[list[str]]:
    """Returns caller tags if provided, else config tags, else None."""
    return caller_tags or config_tags or None


def filter_passing_tests(
    test_cases: list[dict],
    include_passing: bool,
    config: DefaultConfigModel,
) -> list[dict]:
    """Removes passing tests unless include_passing or config says otherwise.

    Args:
        test_cases: All parsed test case dicts.
        include_passing: Caller override to keep passing tests.
        config: Loaded configuration object.

    Returns:
        Filtered list of test case dicts.
    """
    if include_passing or config.test_filter.include_passing:
        return test_cases
    return [t for t in test_cases if t.get("status") != "PASS"]


def load_and_filter_test_cases(
    output: Path,
    config: DefaultConfigModel,
    include_passing: bool = False,
    include_tags: Optional[list[str]] = None,
    exclude_tags: Optional[list[str]] = None,
) -> list[dict]:
    """Loads test cases from output.xml and applies tag/pass filters.

    Args:
        output: Path to Robot Framework output.xml.
        config: Loaded configuration object.
        include_passing: Whether to keep passing tests.
        include_tags: RF tag patterns to include (overrides config).
        exclude_tags: RF tag patterns to exclude (overrides config).

    Returns:
        Filtered list of test case dicts.
    """
    final_include = _resolve_tags(include_tags, config.test_filter.include_tags)
    final_exclude = _resolve_tags(exclude_tags, config.test_filter.exclude_tags)

    test_cases = get_robot_results_from_file_as_dict(
        file_path=output,
        include_tags=final_include,
        exclude_tags=final_exclude,
    )

    return filter_passing_tests(test_cases, include_passing, config)


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
    """Convenience async wrapper that loads files, then delegates to run_analysis.

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
    _apply_concurrency_overrides(
        parsed_config, test_case_concurrency, chunk_concurrency
    )

    test_cases = load_and_filter_test_cases(
        output=output,
        config=parsed_config,
        include_passing=include_passing,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
    )

    return await run_analysis(
        config=parsed_config,
        test_cases=test_cases,
        summarize_failures=summarize_failures,
        dryrun=dryrun,
        quiet=quiet,
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
