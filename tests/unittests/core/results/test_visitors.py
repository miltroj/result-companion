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


def test_no_keyerror_when_end_suite_called_bottom_up_for_nested_suites():
    """Regression: end_suite called bottom-up by RF renames a test in child suite,
    then parent's end_suite iterates the same child tests — must not KeyError on the renamed name.
    """
    visitor = UniqueNameResultVisitor()

    test_in_parent = create_autospec(TestCase)
    test_in_parent.name = "Duplicate Test"
    test_in_parent.id = "s1-t1"

    test_in_child = create_autospec(TestCase)
    test_in_child.name = "Duplicate Test"
    test_in_child.id = "s1-s2-t1"

    child_suite = TestSuite(name="Child Suite")
    child_suite.tests = [test_in_child]
    child_suite.suites = []

    parent_suite = TestSuite(name="Parent Suite")
    parent_suite.tests = [test_in_parent]
    parent_suite.suites = [child_suite]

    visitor.start_test(test_in_parent)
    visitor.start_test(test_in_child)

    visitor.end_suite(child_suite)  # renames test_in_child → "Duplicate Test s1-s2-t1"
    visitor.end_suite(parent_suite)

    assert test_in_parent.name == "Duplicate Test s1-t1"
    assert test_in_child.name == "Duplicate Test s1-s2-t1"
