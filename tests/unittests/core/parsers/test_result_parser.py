from pathlib import Path
from unittest.mock import MagicMock, patch

from result_companion.core.parsers.result_parser import (
    extract_analyzable_items,
    get_robot_results_from_file_as_dict,
    remove_redundant_fields,
)

nested_suites = {
    "name": "E2E",
    "suites": [
        {
            "name": "Nested Suite",
            "suites": [
                {
                    "name": "Nested Suite Level 2",
                    "tests": [
                        {
                            "body": [
                                {
                                    "args": ("${test_var}",),
                                    "name": "Log",
                                    "owner": "BuiltIn",
                                    "status": "PASS",
                                }
                            ],
                            "name": "Nested Test Case",
                            "status": "PASS",
                        }
                    ],
                }
            ],
        },
        {
            "name": "Test Entrypoints",
            "tests": [
                {
                    "body": [
                        {
                            "args": ("${output_file_with_failures_to_analise}",),
                            "body": [
                                {
                                    "html": False,
                                    "level": "TRACE",
                                    "message": "Arguments: [ " "'output.xml' ]",
                                    "type": "MESSAGE",
                                },
                            ],
                            "name": "File Should Exist",
                            "status": "PASS",
                        },
                    ],
                    "message": "error here",
                    "name": "Ollama Local Model Run Should Succede",
                    "status": "FAIL",
                }
            ],
        },
    ],
}


def test_extract_analyzable_items_returns_tests_from_nested_suites():
    expected_result = [
        {
            "body": [
                {
                    "args": ("${test_var}",),
                    "name": "Log",
                    "owner": "BuiltIn",
                    "status": "PASS",
                }
            ],
            "name": "Nested Test Case",
            "status": "PASS",
        },
        {
            "body": [
                {
                    "args": ("${output_file_with_failures_to_analise}",),
                    "body": [
                        {
                            "html": False,
                            "level": "TRACE",
                            "message": "Arguments: [ " "'output.xml' ]",
                            "type": "MESSAGE",
                        },
                    ],
                    "name": "File Should Exist",
                    "status": "PASS",
                },
            ],
            "message": "error here",
            "name": "Ollama Local Model Run Should Succede",
            "status": "FAIL",
        },
    ]

    result = extract_analyzable_items(nested_suites)
    assert len(result) == 2
    assert result == expected_result


def test_extract_analyzable_items_returns_tests_when_no_child_suites():
    test_suite = {
        "tests": [
            {
                "body": [
                    {
                        "args": ("${test_var}",),
                        "name": "Log",
                        "owner": "BuiltIn",
                        "status": "PASS",
                    }
                ],
                "name": "Nested Test Case",
                "status": "PASS",
            },
            {
                "body": [
                    {
                        "args": ("${output_file_with_failures_to_analise}",),
                        "body": [
                            {
                                "html": False,
                                "level": "TRACE",
                                "message": "Arguments: [ " "'output.xml' ]",
                                "type": "MESSAGE",
                            },
                        ],
                        "name": "File Should Exist",
                        "status": "PASS",
                    },
                ],
                "message": "error here",
                "name": "Ollama Local Model Run Should Succede",
                "status": "FAIL",
            },
        ]
    }

    results = extract_analyzable_items(test_suite)
    assert len(results) == 2
    assert results == [
        {
            "body": [
                {
                    "args": ("${test_var}",),
                    "name": "Log",
                    "owner": "BuiltIn",
                    "status": "PASS",
                }
            ],
            "name": "Nested Test Case",
            "status": "PASS",
        },
        {
            "body": [
                {
                    "args": ("${output_file_with_failures_to_analise}",),
                    "body": [
                        {
                            "html": False,
                            "level": "TRACE",
                            "message": "Arguments: [ " "'output.xml' ]",
                            "type": "MESSAGE",
                        },
                    ],
                    "name": "File Should Exist",
                    "status": "PASS",
                },
            ],
            "message": "error here",
            "name": "Ollama Local Model Run Should Succede",
            "status": "FAIL",
        },
    ]


def test_extract_analyzable_items_returns_suite_without_tests_when_setup_fails():
    suite = {
        "name": "Broken Suite",
        "setup": {"name": "Suite Setup", "status": "FAIL", "message": "auth expired"},
        "tests": [{"name": "Should Not Appear", "status": "FAIL"}],
        "suites": [],
    }

    result = extract_analyzable_items(suite)

    assert len(result) == 1
    assert result[0]["name"] == "Broken Suite"
    assert "tests" not in result[0]
    assert "suites" not in result[0]
    assert result[0]["setup"]["status"] == "FAIL"


suite_with_mixed_leaf_results = {
    "name": "Integration-Tests",
    "status": "FAIL",
    "suites": [
        {
            "name": "Etl",
            "id": "s1-s1",
            "status": "FAIL",
            "suites": [
                {
                    "name": "Snowflake-Execution-Passing",
                    "id": "s1-s1-s1",
                    "status": "PASS",
                    "setup": {"name": "Suite Setup", "status": "PASS"},
                    "tests": [
                        {
                            "name": "Config File Should Exist Passing Test",
                            "status": "PASS",
                        },
                        {"name": "Query Snowflake Passing Test", "status": "PASS"},
                    ],
                },
                {
                    "name": "Databricks-Execution-Setup-Fails",
                    "id": "s1-s1-s2",
                    "status": "FAIL",
                    "setup": {
                        "name": "Setup Databricks",
                        "status": "FAIL",
                        "message": "token expired",
                    },
                    "tests": [
                        {"name": "Delta Write Test Failing Test", "status": "FAIL"},
                    ],
                },
                {
                    "name": "Sagemaker-Execution-Passing",
                    "id": "s1-s1-s3",
                    "status": "PASS",
                    "tests": [
                        {"name": "Sagemaker Run Passing Test", "status": "PASS"},
                    ],
                },
            ],
        }
    ],
}


def test_extract_analyzable_items_returns_only_root_suite_when_root_setup_fails():
    suite = {
        **suite_with_mixed_leaf_results,
        "setup": {"name": "Global Setup", "status": "FAIL", "message": "infra down"},
    }

    result = extract_analyzable_items(suite)

    assert len(result) == 1
    assert result[0]["name"] == "Integration-Tests"
    assert result[0]["setup"]["status"] == "FAIL"
    assert "tests" not in result[0]
    assert "suites" not in result[0]
    assert result[0]["setup"]["message"] == "infra down"


def test_extract_analyzable_items_returns_suite_not_tests_when_leaf_setup_fails():
    result = extract_analyzable_items(suite_with_mixed_leaf_results)

    names = [item["name"] for item in result]
    assert len(result) == 4
    assert "Config File Should Exist Passing Test" in names
    assert "Query Snowflake Passing Test" in names
    assert "Databricks-Execution-Setup-Fails" in names
    assert "Sagemaker Run Passing Test" in names
    assert "Delta Write Test Failing Test" not in names

    delta_suite = next(
        r for r in result if r["name"] == "Databricks-Execution-Setup-Fails"
    )
    assert delta_suite["setup"]["status"] == "FAIL"
    assert "tests" not in delta_suite
    assert delta_suite["setup"]["message"] == "token expired"

    snowflake_test = next(
        r for r in result if r["name"] == "Config File Should Exist Passing Test"
    )
    assert "suite_context" in snowflake_test
    assert snowflake_test["suite_context"][0]["name"] == "Snowflake-Execution-Passing"
    assert snowflake_test["suite_context"][0]["setup"]["status"] == "PASS"


def test_extract_analyzable_items_propagates_two_levels_of_suite_context():
    suite = {
        "name": "Root",
        "status": "FAIL",
        "setup": {"name": "Root Setup", "status": "PASS"},
        "teardown": {"name": "Root Teardown", "status": "PASS"},
        "suites": [
            {
                "name": "Mid",
                "status": "FAIL",
                "setup": {"name": "Mid Setup", "status": "PASS"},
                "suites": [
                    {
                        "name": "Leaf-OK",
                        "status": "FAIL",
                        "setup": {"name": "Leaf Setup", "status": "PASS"},
                        "tests": [
                            {"name": "Test-A", "status": "FAIL", "message": "boom"}
                        ],
                    },
                    {
                        "name": "Leaf-Broken",
                        "status": "FAIL",
                        "setup": {
                            "name": "Leaf Setup",
                            "status": "FAIL",
                            "message": "db down",
                        },
                        "tests": [{"name": "Test-B", "status": "FAIL"}],
                    },
                ],
            }
        ],
    }

    result = extract_analyzable_items(suite)

    expected = [
        {
            "name": "Test-A",
            "status": "FAIL",
            "message": "boom",
            "suite_context": [
                {
                    "name": "Root",
                    "setup": {"name": "Root Setup", "status": "PASS"},
                    "teardown": {"name": "Root Teardown", "status": "PASS"},
                },
                {"name": "Mid", "setup": {"name": "Mid Setup", "status": "PASS"}},
                {"name": "Leaf-OK", "setup": {"name": "Leaf Setup", "status": "PASS"}},
            ],
        },
        {
            "name": "Leaf-Broken",
            "id": None,
            "status": "FAIL",
            "setup": {"name": "Leaf Setup", "status": "FAIL", "message": "db down"},
            "suite_context": [
                {
                    "name": "Root",
                    "setup": {"name": "Root Setup", "status": "PASS"},
                    "teardown": {"name": "Root Teardown", "status": "PASS"},
                },
                {"name": "Mid", "setup": {"name": "Mid Setup", "status": "PASS"}},
            ],
        },
    ]

    assert len(result) == 2
    assert result[0] == expected[0]
    assert result[1]["name"] == "Leaf-Broken"
    assert result[1]["setup"]["status"] == "FAIL"
    assert result[1]["suite_context"] == expected[1]["suite_context"]


def test_remove_redundant_fields_strips_robot_internal_fields():
    data = {
        "body": [],
        "message": "error here",
        "name": "Ollama Local Model Run Should Succede",
        "status": "FAIL",
        "elapsed_time": "0:00:00.000",
        "lineno": 1,
        "owner": "BuiltIn",
        "start_time": "2021-10-01 12:00:00",
        "html": False,
        "type": "MESSAGE",
        "assign": "output_file_with_failures_to_analise",
        "level": "TRACE",
        "timestamp": "2021-10-01 12:00:00",
    }
    result = remove_redundant_fields(data)
    assert "elapsed_time" not in result
    assert "lineno" not in result
    assert "owner" not in result
    assert "start_time" not in result
    assert "html" not in result
    assert "type" not in result
    assert "assign" not in result
    assert "level" not in result
    assert "timestamp" not in result
    assert result == {
        "body": [],
        "message": "error here",
        "name": "Ollama Local Model Run Should Succede",
        "status": "FAIL",
    }


@patch("result_companion.core.parsers.result_parser.ExecutionResult")
@patch("result_companion.core.parsers.result_parser.UniqueNameResultVisitor")
def test_get_robot_results_passes_tags_to_rf_configure(mock_visitor, mock_exec_result):
    """Verifies include/exclude tags are passed to RF's result.configure()."""
    mock_result = MagicMock()
    mock_result.suite.to_dict.return_value = {
        "tests": [{"name": "Test1", "status": "FAIL", "tags": ["smoke"]}]
    }
    mock_exec_result.return_value = mock_result

    get_robot_results_from_file_as_dict(
        file_path=Path("fake.xml"),
        include_tags=["smoke*", "critical"],
        exclude_tags=["wip"],
    )

    mock_result.configure.assert_called_once_with(
        suite_config={"include_tags": ["smoke*", "critical"], "exclude_tags": ["wip"]}
    )
