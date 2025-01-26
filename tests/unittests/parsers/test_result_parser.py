from result_companion.parsers.result_parser import search_for_test_caseses, remove_redundant_fields


neasted_suites = {
    "name": "E2E",
    "suites": [
        {
            "name": "Neasted Suite",
            "suites": [
                {
                    "name": "Test Neasted",
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
                            "name": "Test Neasted Test Case",
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


def test_search_for_all_tests_recursively():
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
                "name": "Test Neasted Test Case",
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
            }
    ]

    result = search_for_test_caseses(neasted_suites, acumulated_tests=[])
    assert len(result) == 2
    assert result == expected_result


def test_should_search_for_tests_where_there_are_no_test_suites():
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
                "name": "Test Neasted Test Case",
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

    results = search_for_test_caseses(test_suite, acumulated_tests=[])
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
            "name": "Test Neasted Test Case",
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


def test_removing_redundant_fields():
    data = {
            "body": [
            ],
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
        "body": [
        ],
        "message": "error here",
        "name": "Ollama Local Model Run Should Succede",
        "status": "FAIL",
    }