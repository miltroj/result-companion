from result_companion.core.results.test_filter import (
    TestFilter,
    filter_tests,
    matches_test_tags,
    tag_matches_pattern,
)


def make_test(name: str, tags: list[str], status: str = "FAIL") -> dict:
    """Factory for test case data."""
    return {"name": name, "tags": tags, "status": status}


def test_tag_matches_pattern_exact():
    assert tag_matches_pattern("smoke", "smoke")
    assert tag_matches_pattern("SMOKE", "smoke")


def test_tag_matches_pattern_wildcard():
    assert tag_matches_pattern("smoke-login", "smoke*")
    assert tag_matches_pattern("critical", "crit*")
    assert not tag_matches_pattern("regression", "smoke*")


def test_matches_test_tags_include_empty_means_all():
    result = matches_test_tags(["smoke", "critical"], [], [])
    assert result is True


def test_matches_test_tags_include_pattern_matches():
    result = matches_test_tags(["smoke", "regression"], ["smoke"], [])
    assert result is True


def test_matches_test_tags_include_pattern_no_match():
    result = matches_test_tags(["regression"], ["smoke"], [])
    assert result is False


def test_matches_test_tags_exclude_takes_precedence():
    result = matches_test_tags(["smoke", "wip"], ["smoke"], ["wip"])
    assert result is False


def test_filter_tests_include_tags_only_failures():
    tests = [
        make_test("T1", ["smoke"], "FAIL"),
        make_test("T2", ["regression"], "FAIL"),
        make_test("T3", ["smoke"], "PASS"),
    ]
    test_filter = TestFilter(
        include_tags=["smoke"], exclude_tags=[], include_passing=False
    )

    result = filter_tests(tests, test_filter)

    assert len(result) == 1
    assert result[0]["name"] == "T1"


def test_filter_tests_exclude_tags():
    tests = [
        make_test("T1", ["smoke"], "FAIL"),
        make_test("T2", ["smoke", "wip"], "FAIL"),
    ]
    test_filter = TestFilter(
        include_tags=[], exclude_tags=["wip"], include_passing=False
    )

    result = filter_tests(tests, test_filter)

    assert len(result) == 1
    assert result[0]["name"] == "T1"


def test_filter_tests_include_passing():
    tests = [
        make_test("T1", ["smoke"], "PASS"),
        make_test("T2", ["smoke"], "FAIL"),
    ]
    test_filter = TestFilter(
        include_tags=["smoke"], exclude_tags=[], include_passing=True
    )

    result = filter_tests(tests, test_filter)

    assert len(result) == 2


def test_filter_tests_no_filters_returns_failures_only():
    tests = [
        make_test("T1", [], "PASS"),
        make_test("T2", [], "FAIL"),
    ]
    test_filter = TestFilter(include_tags=[], exclude_tags=[], include_passing=False)

    result = filter_tests(tests, test_filter)

    assert len(result) == 1
    assert result[0]["name"] == "T2"


def test_filter_tests_wildcard_patterns():
    tests = [
        make_test("T1", ["smoke-login"], "FAIL"),
        make_test("T2", ["smoke-logout"], "FAIL"),
        make_test("T3", ["regression"], "FAIL"),
    ]
    test_filter = TestFilter(
        include_tags=["smoke*"], exclude_tags=[], include_passing=False
    )

    result = filter_tests(tests, test_filter)

    assert len(result) == 2
    assert result[0]["name"] == "T1"
    assert result[1]["name"] == "T2"


def test_filter_tests_no_tags_included_when_include_empty():
    tests = [
        make_test("T1", [], "FAIL"),
        make_test("T2", ["smoke"], "FAIL"),
    ]
    test_filter = TestFilter(include_tags=[], exclude_tags=[], include_passing=False)

    result = filter_tests(tests, test_filter)

    assert len(result) == 2


def test_filter_tests_no_tags_excluded_when_include_specified():
    tests = [
        make_test("T1", [], "FAIL"),
        make_test("T2", ["smoke"], "FAIL"),
    ]
    test_filter = TestFilter(
        include_tags=["smoke"], exclude_tags=[], include_passing=False
    )

    result = filter_tests(tests, test_filter)

    assert len(result) == 1
    assert result[0]["name"] == "T2"
