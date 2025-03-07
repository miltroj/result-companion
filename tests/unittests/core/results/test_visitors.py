from unittest.mock import create_autospec

from robot.model.testcase import TestCase
from robot.model.testsuite import TestSuite

from result_companion.core.results.visitors import UniqueNameResultVisitor


def test_rename_duplicate_test_names():
    """Test that the visitor renames tests with duplicate names by appending their ID."""
    visitor = UniqueNameResultVisitor()

    test1 = create_autospec(TestCase)
    test1.name = "Duplicate Test"
    test1.id = "s1-t1"

    test2 = create_autospec(TestCase)
    test2.name = "Duplicate Test"
    test2.id = "s1-t2"

    test3 = create_autospec(TestCase)
    test3.name = "Unique Test"
    test3.id = "s1-t3"

    test4 = create_autospec(TestCase)
    test4.name = "Duplicate Test"
    test4.id = "s2-t1"

    # Create real test suites
    main_suite = TestSuite(name="Main Suite")
    child_suite = TestSuite(name="Child Suite")

    # Add tests to suites
    main_suite.tests = [test1, test2, test3]
    child_suite.tests = [test4]
    main_suite.suites = [child_suite]

    visitor.start_test(test1)
    visitor.start_test(test2)
    visitor.start_test(test3)
    visitor.start_test(test4)
    visitor.end_suite(main_suite)

    assert test1.name == "Duplicate Test s1-t1"
    assert test2.name == "Duplicate Test s1-t2"
    assert test3.name == "Unique Test"  # Should remain unchanged
    assert test4.name == "Duplicate Test s2-t1"
