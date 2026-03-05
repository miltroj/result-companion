from pathlib import Path

from robot.api import ExecutionResult

from result_companion.core.results.visitors import UniqueNameResultVisitor
from result_companion.core.utils.logging_config import logger


def extract_analyzable_items(
    suite: dict, parent_suites: list[dict] | None = None
) -> list[dict]:
    """Walks suite tree returning items to analyze.

    Suite setup fails -> returns suite itself (skip children).
    Suite setup passes -> recurses into sub-suites and tests.
    Propagates parent suite setups/teardowns as context.
    """
    if parent_suites is None:
        parent_suites = []

    if suite.get("setup", {}).get("status") == "FAIL":
        item = {k: v for k, v in suite.items() if k not in ("tests", "suites")}
        if parent_suites:
            item["suite_context"] = parent_suites
        return [item]

    suite_meta = {k: suite[k] for k in ("name", "setup", "teardown") if k in suite}
    chain = (
        parent_suites + [suite_meta]
        if suite_meta.get("setup") or suite_meta.get("teardown")
        else parent_suites
    )

    items = []
    for test in suite.get("tests", []):
        if chain:
            test["suite_context"] = chain
        items.append(test)
    for sub in suite.get("suites", []):
        items.extend(extract_analyzable_items(sub, chain))
    return items


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
    include_tags: list[str] | None = None,
    exclude_tags: list[str] | None = None,
) -> list[dict]:
    """Parses RF output.xml and returns test cases as dicts.

    Uses RF's native filtering via result.configure() - same as rebot.

    Args:
        file_path: Path to output.xml.
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
    all_results = extract_analyzable_items(all_results)
    all_results = remove_redundant_fields(all_results)
    return all_results
