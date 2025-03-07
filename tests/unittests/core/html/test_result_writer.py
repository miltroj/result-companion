from pathlib import Path
from unittest.mock import create_autospec, patch

import pytest
from robot.model.testcase import TestCase
from robot.model.testsuite import TestSuite

from result_companion.core.html.result_writer import create_base_llm_result_log
from result_companion.core.results.visitors import UniqueNameResultVisitor


class DummyResult:
    """A dummy ExecutionResult-like object for testing."""

    def __init__(self, suite):
        self.suite = suite

    def visit(self, visitor: UniqueNameResultVisitor):
        # Simulate traversing the suite: call start_test on each test and then end_suite on the suite.
        for test in self.suite.tests:
            visitor.start_test(test)
        visitor.end_suite(self.suite)


@pytest.fixture
def dummy_result():
    test1 = create_autospec(TestCase)
    test1.name = "Test 1"
    test1.id = "T1"

    test2 = create_autospec(TestCase)
    test2.name = "Test 2"
    test2.id = "T2"
    suite = TestSuite("Dummy Suite")
    suite.tests = [test1, test2]
    return DummyResult(suite)


def test_write_log_with_llm_result_as_html_log_file(dummy_result):
    dummy_input = Path("dummy_input.xml")
    dummy_output = Path("dummy_output.html")

    with patch(
        "result_companion.core.html.result_writer.ExecutionResult",
        return_value=dummy_result,
    ) as mock_exec_result:
        with patch(
            "result_companion.core.html.result_writer.LLMResultWriter.write_results"
        ) as mock_write_results:

            # Call the function under test.
            create_base_llm_result_log(dummy_input, dummy_output)

            mock_exec_result.assert_called_once_with(dummy_input)
            mock_write_results.assert_called_once()
            args, kwargs = mock_write_results.call_args
            # Check that the "log" keyword argument is equal to dummy_output.
            assert kwargs.get("log") == dummy_output

    tests = dummy_result.suite.tests
    assert tests[0].name == "Test 1"
    assert tests[1].name == "Test 2"
