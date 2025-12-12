"""Pure functions for filtering Robot Framework test results."""

from dataclasses import dataclass
from fnmatch import fnmatchcase


@dataclass
class TestFilter:
    """Test filter configuration matching Robot Framework conventions.

    Attributes:
        include_tags: Tag patterns to include (RF-style: 'tag*', 'tagANDother').
        exclude_tags: Tag patterns to exclude.
        include_passing: Include tests with PASS status.
    """

    include_tags: list[str]
    exclude_tags: list[str]
    include_passing: bool

    def __post_init__(self):
        self.include_tags = self.include_tags or []
        self.exclude_tags = self.exclude_tags or []


def tag_matches_pattern(tag: str, pattern: str) -> bool:
    """Checks if tag matches RF-style pattern.

    Args:
        tag: Tag name to check.
        pattern: Pattern with wildcards (e.g., 'smoke*', 'critical').

    Returns:
        True if tag matches pattern.
    """
    return fnmatchcase(tag.lower(), pattern.lower())


def matches_test_tags(
    test_tags: list[str], include: list[str], exclude: list[str]
) -> bool:
    """Checks if test tags match include/exclude patterns (RF logic).

    Args:
        test_tags: List of tags from test case.
        include: Include patterns (empty = include all).
        exclude: Exclude patterns.

    Returns:
        True if test should be included.
    """
    if not test_tags:
        test_tags = []

    # Exclude takes precedence
    if exclude:
        for pattern in exclude:
            if any(tag_matches_pattern(tag, pattern) for tag in test_tags):
                return False

    # Include (empty means include all)
    if not include:
        return True

    return any(
        tag_matches_pattern(tag, pattern) for pattern in include for tag in test_tags
    )


def filter_tests(test_cases: list[dict], test_filter: TestFilter) -> list[dict]:
    """Filters test cases by tags and status.

    Args:
        test_cases: List of test case dictionaries.
        test_filter: Filter configuration.

    Returns:
        Filtered list of test cases.
    """
    filtered = []

    for test_case in test_cases:
        # Status filter
        if test_case.get("status") == "PASS" and not test_filter.include_passing:
            continue

        # Tag filter
        test_tags = test_case.get("tags", [])
        if not matches_test_tags(
            test_tags, test_filter.include_tags, test_filter.exclude_tags
        ):
            continue

        filtered.append(test_case)

    return filtered
