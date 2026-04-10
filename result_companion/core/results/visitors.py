from robot.api import ResultVisitor

from result_companion.core.utils.logging_config import logger


# TODO: workaround to fix potentail problem with exposing llm results to invalid test cases in log.html
class UniqueNameResultVisitor(ResultVisitor):
    """Ensures unique test and suite names by appending IDs to duplicates."""

    def __init__(self):
        super().__init__()
        self._test_names: dict[str, int] = {}
        self._suite_names: dict[str, int] = {}

    def start_test(self, test) -> None:
        """Counts test name occurrences."""
        self._test_names[test.name] = self._test_names.get(test.name, 0) + 1

    def start_suite(self, suite) -> None:
        """Counts suite name occurrences."""
        self._suite_names[suite.name] = self._suite_names.get(suite.name, 0) + 1

    def end_suite(self, suite) -> None:
        """Renames duplicates — suite itself, then its direct tests."""
        if self._suite_names.get(suite.name, 0) > 1:
            logger.debug(f"Renaming suite '{suite.name}' to '{suite.name} {suite.id}'")
            suite.name = f"{suite.name} {suite.id}"

        for test in suite.tests:
            if self._test_names.get(test.name, 0) > 1:
                logger.debug(f"Renaming test '{test.name}' to '{test.name} {test.id}'")
                test.name = f"{test.name} {test.id}"

        for child_suite in suite.suites:
            for test in child_suite.tests:
                if self._test_names.get(test.name, 0) > 1:
                    logger.debug(
                        f"Renaming test '{test.name}' to '{test.name} {test.id}'"
                    )
                    test.name = f"{test.name} {test.id}"
