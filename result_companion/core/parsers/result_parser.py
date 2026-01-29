from pathlib import Path

from robot.api import ExecutionResult

from result_companion.core.results.visitors import UniqueNameResultVisitor
from result_companion.core.utils.log_levels import LogLevels
from result_companion.core.utils.logging_config import logger


def search_for_test_cases(
    tests: dict | list, accumulated: list | None = None
) -> list[dict]:
    """Recursively extracts test cases from nested suite structure."""
    if accumulated is None:
        accumulated = []
    if isinstance(tests, list):
        for el in tests:
            search_for_test_cases(el, accumulated)
    elif isinstance(tests, dict):
        if tests.get("tests"):
            accumulated.extend(tests["tests"])
        elif tests.get("suites"):
            search_for_test_cases(tests["suites"], accumulated)
    return accumulated


def remove_redundant_fields(data: list[dict]) -> list[dict]:
    fields_to_remove = [
        "elapsed_time",
        "lineno",
        "owner",
        "start_time",
        "html",
        "type",
        "assign",
        "level",
        "timestamp",
    ]
    # Note: 'id' is NOT removed - it's essential for matching LLM results to tests

    if isinstance(data, dict):
        # Remove fields from the current dictionary
        for field in fields_to_remove:
            data.pop(field, None)

        # Recursively process child dictionaries
        for key, value in data.items():
            if isinstance(value, dict):
                data[key] = remove_redundant_fields(value)
            elif isinstance(value, list):
                data[key] = [
                    remove_redundant_fields(item) if isinstance(item, dict) else item
                    for item in value
                ]

    elif isinstance(data, list):
        # Recursively process child dictionaries in the list
        return [
            remove_redundant_fields(item) if isinstance(item, dict) else item
            for item in data
        ]

    return data


def get_robot_results_from_file_as_dict(
    file_path: Path,
    log_level: LogLevels,
    include_tags: list[str] | None = None,
    exclude_tags: list[str] | None = None,
) -> list[dict]:
    """Parses RF output.xml and returns test cases as dicts.

    Uses RF's native filtering via result.configure() - same as rebot.

    Args:
        file_path: Path to output.xml.
        log_level: Log level for parsing.
        include_tags: RF tag patterns to include (e.g., ['smoke*', 'critical']).
        exclude_tags: RF tag patterns to exclude (e.g., ['wip', 'bug-*']).

    Returns:
        List of test case dictionaries.
    """
    logger.debug(f"Getting robot results from {file_path}")
    result = ExecutionResult(file_path)

    # Use RF's native filtering (same as rebot --include/--exclude)
    suite_config = {}
    if include_tags:
        suite_config["include_tags"] = include_tags
    if exclude_tags:
        suite_config["exclude_tags"] = exclude_tags
    if suite_config:
        result.configure(suite_config=suite_config)
        logger.debug(f"Applied RF native filtering: {suite_config}")

    result.visit(UniqueNameResultVisitor())
    all_results = result.suite.to_dict()
    all_results = search_for_test_cases(all_results)
    all_results = remove_redundant_fields(all_results)
    return all_results
