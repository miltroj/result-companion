import uuid
from pathlib import Path

from robot.api import ExecutionResult, ResultVisitor

from result_companion.core.utils.log_levels import LogLevels
from result_companion.core.utils.logging_config import logger


def search_for_test_caseses(
    tests: dict | list, acumulated_tests: list = []
) -> list[dict]:
    # TODO: optimise this function
    if isinstance(tests, list):
        for el in tests:
            search_for_test_caseses(el, acumulated_tests)
        return acumulated_tests
    elif isinstance(tests, dict):
        if tests.get("tests", None):
            for el in tests["tests"]:
                acumulated_tests.append(el)
            return acumulated_tests
        elif tests.get("suites", None):
            return search_for_test_caseses(tests["suites"], acumulated_tests)


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


# TODO: workaround to fix potentail problem with exposing llm results to invalid test cases in log.html
def add_unique_sufix_for_test_cases_with_duplicated_names(
    data: list[dict], random_sufix: str | None = None
) -> list[dict]:
    all_test_cases_names = [test_case["name"] for test_case in data]
    unique_test_cases = set(all_test_cases_names)
    all_duplicated_test_cases_names = [
        test_case_name
        for test_case_name in all_test_cases_names
        if all_test_cases_names.count(test_case_name) > 1
    ]

    if len(unique_test_cases) == len(all_test_cases_names):
        return data

    for test_case in data:
        if test_case["name"] in all_duplicated_test_cases_names:
            if not random_sufix:
                random_sufix = str(uuid.uuid4())
            test_case["name"] = f"{test_case['name']}_{random_sufix}"
    return data


def get_robot_results_from_file_as_dict(
    file_path: Path, log_level: LogLevels
) -> list[dict]:
    logger.debug(f"Getting robot results from {file_path}")
    result = ExecutionResult(file_path)
    result.visit(ResultVisitor())
    all_results = result.suite.to_dict()
    all_results = search_for_test_caseses(all_results)
    all_results = remove_redundant_fields(all_results)
    all_results = add_unique_sufix_for_test_cases_with_duplicated_names(all_results)
    return all_results
