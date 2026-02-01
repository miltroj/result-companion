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
from result_companion.core.utils.log_levels import LogLevels
from result_companion.core.utils.logging_config import logger, set_global_log_level


def _run_ollama_init_strategy(model_name: str, strategy_params: dict) -> None:
    """Runs Ollama initialization strategy if model is Ollama.

    Args:
        model_name: LiteLLM model identifier (e.g., ollama_chat/llama2).
        strategy_params: Parameters for the init strategy.
    """
    if not model_name.startswith("ollama"):
        return

    # Extract short model name for Ollama server check
    # e.g., "ollama_chat/deepseek-r1:1.5b" -> "deepseek-r1"
    model_short = strategy_params.get("model_name")
    if not model_short:
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
        model_name: LiteLLM model identifier (e.g., copilot_sdk/gpt-4.1).
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
) -> bool:
    set_global_log_level(str(log_level))

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

    # Run provider init strategies
    _run_ollama_init_strategy(
        model_name=parsed_config.llm_factory.model,
        strategy_params=parsed_config.llm_factory.strategy.parameters,
    )
    _register_copilot_if_needed(parsed_config.llm_factory.model)

    logger.debug(f"Using model: {parsed_config.llm_factory.model}")

    llm_results = await execute_llm_and_get_results(
        test_cases=test_cases,
        config=parsed_config,
        dryrun=dryrun,
    )

    report_path = report if report else "rc_log.html"
    if llm_results:
        model_info = {"model": parsed_config.llm_factory.model}
        create_llm_html_log(
            input_result_path=output,
            llm_output_path=report_path,
            llm_results=llm_results,
            model_info=model_info,
        )
        logger.info(f"Report created: {Path(report_path).resolve()}")

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
) -> bool:
    """Runs the Result Companion analysis.

    Args:
        output: Path to Robot Framework output.xml file.
        log_level: Logging verbosity level.
        config: Optional path to user config file.
        report: Optional output report path.
        include_passing: Whether to include passing tests.
        test_case_concurrency: Number of test cases to process in parallel.
        chunk_concurrency: Number of chunks to process in parallel.
        include_tags: RF tag patterns to include.
        exclude_tags: RF tag patterns to exclude.
        dryrun: If True, skip LLM calls.

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
