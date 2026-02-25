import asyncio
import time
from pathlib import Path
from typing import Optional

from result_companion.core.analizers.factory_common import execute_llm_and_get_results
from result_companion.core.analizers.local.ollama_runner import ollama_on_init_strategy
from result_companion.core.analizers.remote.copilot import register_copilot_provider
from result_companion.core.html.html_creator import create_llm_html_log
from result_companion.core.parsers.config import load_config
from result_companion.core.parsers.result_parser import (
    get_robot_results_from_file_as_dict,
)
from result_companion.core.results.text_report import (
    render_text_report,
    summarize_failures_with_llm,
)
from result_companion.core.utils.log_levels import LogLevels
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
        "copilot_sdk": lambda: _register_copilot_if_needed(model_name),
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

    # Extract short model name for Ollama server check
    # e.g., "ollama_chat/deepseek-r1:1.5b" -> "deepseek-r1"
    parts = model_name.split("/")
    if len(parts) > 1:
        model_short = parts[1].split(":")[0]
    else:
        model_short = model_name.split(":")[0]

    logger.debug(f"Running Ollama init strategy for model: {model_short}")
    ollama_on_init_strategy(model_name=model_short)


def _register_copilot_if_needed(model_name: str) -> None:
    """Registers Copilot SDK provider if model uses copilot_sdk prefix.

    Args:
        model_name: LiteLLM model identifier (e.g., copilot_sdk/gpt-5-mini).
    """
    if not model_name.startswith("copilot_sdk/"):
        return

    logger.debug(f"Registering Copilot SDK provider for model: {model_name}")
    register_copilot_provider()


async def _main(
    output: Path,
    log_level: LogLevels,
    config: Optional[Path],
    report: Optional[str],
    include_passing: bool,
    test_case_concurrency: Optional[int] = None,
    chunk_concurrency: Optional[int] = None,
    include_tags: Optional[list[str]] = None,
    exclude_tags: Optional[list[str]] = None,
    dryrun: bool = False,
    html_report: bool = True,
    text_report: Optional[str] = None,
    print_text_report: bool = False,
    summarize_failures: bool = False,
    quiet: bool = False,
) -> bool:
    resolved_log_level = "ERROR" if quiet else str(log_level)
    set_global_log_level(resolved_log_level)

    logger.info("Starting Result Companion!")
    start = time.time()
    parsed_config = load_config(config)

    if test_case_concurrency is not None:
        parsed_config.concurrency.test_case = test_case_concurrency
    if chunk_concurrency is not None:
        parsed_config.concurrency.chunk = chunk_concurrency

    # Merge CLI tags with config (CLI takes precedence)
    final_include = include_tags or parsed_config.test_filter.include_tags or None
    final_exclude = exclude_tags or parsed_config.test_filter.exclude_tags or None

    test_cases = get_robot_results_from_file_as_dict(
        file_path=output,
        log_level=LogLevels.DEBUG,
        include_tags=final_include,
        exclude_tags=final_exclude,
    )

    # Filter passing tests (RF doesn't have this natively)
    should_include_passing = (
        include_passing or parsed_config.test_filter.include_passing
    )
    if not should_include_passing:
        test_cases = [t for t in test_cases if t.get("status") != "PASS"]

    logger.info(f"Filtered to {len(test_cases)} test cases")

    _run_provider_init_strategies(model_name=parsed_config.llm_factory.model)

    logger.debug(f"Using model: {parsed_config.llm_factory.model}")

    llm_results = await execute_llm_and_get_results(
        test_cases=test_cases,
        config=parsed_config,
        dryrun=dryrun,
        quiet=quiet,
    )

    failed_test_names = [t["name"] for t in test_cases if t.get("status") == "FAIL"]
    report_path = report if report else "rc_log.html"
    if llm_results and html_report:
        model_info = {"model": parsed_config.llm_factory.model}
        create_llm_html_log(
            input_result_path=output,
            llm_output_path=report_path,
            llm_results=llm_results,
            model_info=model_info,
        )
        logger.info(f"Report created: {Path(report_path).resolve()}")

    overall_summary = None
    if summarize_failures and llm_results and not dryrun:
        overall_summary = await summarize_failures_with_llm(
            llm_results=llm_results,
            model_name=parsed_config.llm_factory.model,
            config=parsed_config,
        )

    should_emit_text = bool(text_report) or print_text_report
    if should_emit_text:
        text_output = render_text_report(
            llm_results=llm_results,
            failed_test_names=failed_test_names,
            overall_summary=overall_summary,
        )
        if text_report:
            Path(text_report).write_text(text_output)
            logger.info(f"Text report created: {Path(text_report).resolve()}")
        if print_text_report:
            print(text_output)

    stop = time.time()
    logger.debug(f"Execution time: {stop - start}")

    # Allow aiohttp SSL connections to cleanup before event loop closes
    # This prevents "Event loop is closed" errors from liteLLM's internal aiohttp client
    await asyncio.sleep(0.25)

    return True


def run_rc(
    output: Path,
    log_level: LogLevels,
    config: Optional[Path],
    report: Optional[str],
    include_passing: bool,
    test_case_concurrency: Optional[int] = None,
    chunk_concurrency: Optional[int] = None,
    include_tags: Optional[list[str]] = None,
    exclude_tags: Optional[list[str]] = None,
    dryrun: bool = False,
    html_report: bool = True,
    text_report: Optional[str] = None,
    print_text_report: bool = False,
    summarize_failures: bool = False,
    quiet: bool = False,
) -> bool:
    """Runs the Result Companion analysis.

    Args:
        output: Path to Robot Framework output.xml file.
        log_level: Logging verbosity level.
        config: Optional path to user config file.
        report: Optional HTML report output path.
        include_passing: Whether to include passing tests.
        test_case_concurrency: Number of test cases to process in parallel.
        chunk_concurrency: Number of chunks to process in parallel.
        include_tags: RF tag patterns to include.
        exclude_tags: RF tag patterns to exclude.
        dryrun: If True, skip LLM calls.
        html_report: Whether to generate HTML report.
        text_report: Optional text summary output path.
        print_text_report: Whether to print text report to stdout.
        summarize_failures: Whether to ask LLM for overall failure summary.
        quiet: Whether to suppress logs and progress output.

    Returns:
        True if analysis completed successfully.
    """
    try:
        return asyncio.run(
            _main(
                output=output,
                log_level=log_level,
                config=config,
                report=report,
                html_report=html_report,
                text_report=text_report,
                print_text_report=print_text_report,
                summarize_failures=summarize_failures,
                quiet=quiet,
                include_passing=include_passing,
                test_case_concurrency=test_case_concurrency,
                chunk_concurrency=chunk_concurrency,
                include_tags=include_tags,
                exclude_tags=exclude_tags,
                dryrun=dryrun,
            )
        )
    except Exception:
        logger.critical("Unhandled exception", exc_info=True)
        raise
