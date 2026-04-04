"""Shared analysis helpers for API and CLI entrypoints."""

from pathlib import Path
from typing import Optional

from result_companion.core.analizers.local.ollama_runner import ollama_on_init_strategy
from result_companion.core.parsers.config import DefaultConfigModel
from result_companion.core.parsers.result_parser import (
    get_robot_results_from_file_as_dict,
)
from result_companion.core.utils.logging_config import logger


def run_provider_init_strategies(model_name: str) -> None:
    """Runs provider-specific initialization based on LiteLLM model prefix.

    Args:
        model_name: LiteLLM model identifier (e.g., ollama_chat/llama2).
    """
    if not model_name.startswith("ollama"):
        return

    parts = model_name.split("/")
    model_short = parts[1].split(":")[0] if len(parts) > 1 else model_name.split(":")[0]
    logger.debug(f"Running Ollama init strategy for model: {model_short}")
    ollama_on_init_strategy(model_name=model_short)


def apply_concurrency_overrides(
    config: DefaultConfigModel,
    test_case_concurrency: Optional[int],
    chunk_concurrency: Optional[int],
) -> None:
    """Mutates config concurrency settings with CLI/caller overrides."""
    if test_case_concurrency is not None:
        config.concurrency.test_case = test_case_concurrency
    if chunk_concurrency is not None:
        config.concurrency.chunk = chunk_concurrency


def resolve_tags(
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
    final_include = resolve_tags(include_tags, config.test_filter.include_tags)
    final_exclude = resolve_tags(exclude_tags, config.test_filter.exclude_tags)

    test_cases = get_robot_results_from_file_as_dict(
        file_path=output,
        include_tags=final_include,
        exclude_tags=final_exclude,
    )

    return filter_passing_tests(test_cases, include_passing, config)
