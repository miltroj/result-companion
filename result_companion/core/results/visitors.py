from robot.api import ResultVisitor

from result_companion.core.utils.logging_config import logger


# TODO: workaround to fix potentail problem with exposing llm results to invalid test cases in log.html
class UniqueNameResultVisitor(ResultVisitor):
    """Custom visitor that ensures unique test case names by appending IDs to duplicates."""

    def __init__(self):
        super().__init__()
        self.test_names = {}

    def start_test(self, test):
        """Called when a test is encountered during traversal."""
        if test.name in self.test_names:
            self.test_names[test.name] += 1
        else:
            self.test_names[test.name] = 1

    def end_suite(self, suite):
        """Called when suite processing is complete."""
        for test in suite.tests:
            if test.name in self.test_names and self.test_names[test.name] > 1:
                logger.debug(f"Renaming test '{test.name}' to '{test.name} {test.id}'")
                test.name = f"{test.name} {test.id}"

        for child_suite in suite.suites:
            for test in child_suite.tests:
                if test.name in self.test_names and self.test_names[test.name] > 1:
                    logger.debug(
                        f"Renaming test '{test.name}' to '{test.name} {test.id}'"
                    )
                    test.name = f"{test.name} {test.id}"
