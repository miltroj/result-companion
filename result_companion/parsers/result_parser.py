from robot.api import ExecutionResult, ResultVisitor
from result_companion.parsers.cli_parser import LogLevel


def search_for_test_caseses(tests: dict | list, 
                            acumulated_tests: list = []) -> list[dict]:
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
    fields_to_remove = ['elapsed_time', 'lineno', 'owner', 'start_time', 'html', 'type', 'assign', "level", "timestamp"]

    if isinstance(data, dict):
        # Remove fields from the current dictionary
        for field in fields_to_remove:
            data.pop(field, None)

        # Recursively process child dictionaries
        for key, value in data.items():
            if isinstance(value, dict):
                data[key] = remove_redundant_fields(value)
            elif isinstance(value, list):
                data[key] = [remove_redundant_fields(item) if isinstance(item, dict) else item for item in value]

    elif isinstance(data, list):
        # Recursively process child dictionaries in the list
        return [remove_redundant_fields(item) if isinstance(item, dict) else item for item in data]

    return data


def get_robot_results_from_file_as_dict(file_path: str, log_level: LogLevel) -> dict:
    print(f"Getting robot results from {file_path}")
    result = ExecutionResult(file_path)
    result.visit(ResultVisitor())
    all_results = result.suite.to_dict()
    all_results = search_for_test_caseses(all_results)
    all_results = remove_redundant_fields(all_results)
    return all_results
