"""Tests for result_companion.core.chunking.rf_results."""

from __future__ import annotations

import hashlib

import pytest
from robot.api import ExecutionResult
from robot.errors import DataError
from robot.result.model import Keyword as RFKeyword
from robot.result.model import Message as RFMessage
from robot.result.model import TestCase as RFTestCase
from robot.result.model import TestSuite as RFTestSuite

from result_companion.core.chunking.chunking import render_lines_to_text
from result_companion.core.chunking.rf_results import (
    ALL_FIELDS,
    ContextAwareRobotResults,
    RenderContext,
    TestLines,
    _control_path_segment,
    _embedded_image_id,
    _iter_tests_with_context,
    _join_parts,
    _render_body_item,
    _render_common_fields,
    _render_keyword,
    _render_message,
    _render_suite,
    _render_test,
    get_rc_robot_results,
)
from result_companion.core.chunking.utils import Chunking
from result_companion.core.results.visitors import UniqueNameResultVisitor

_MINIMAL_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<robot generator="Robot 7.1" rpa="false" schemaversion="5">
<suite id="s1" name="Suite">
  <test id="s1-t1" name="Passing Test">
    <tag>smoke</tag>
    <status status="PASS" start="2026-01-01T00:00:00.000000" elapsed="0.001"/>
  </test>
  <test id="s1-t2" name="Failing Test">
    <tag>critical</tag>
    <status status="FAIL" start="2026-01-01T00:00:00.001000" elapsed="0.001"/>
  </test>
  <status status="FAIL" start="2026-01-01T00:00:00.000000" elapsed="0.002"/>
</suite>
<statistics>
  <total><stat pass="1" fail="1" skip="0">All Tests</stat></total>
  <tag><stat pass="1" fail="0" skip="0">smoke</stat></tag>
  <tag><stat pass="0" fail="1" skip="0">critical</stat></tag>
  <suite><stat id="s1" pass="1" fail="1" skip="0">Suite</stat></suite>
</statistics>
<errors/>
</robot>
"""

_SCREENSHOT_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<robot generator="Robot 7.1" rpa="false" schemaversion="5">
<suite id="s1" name="Suite">
  <test id="s1-t1" name="Failing Test">
    <kw name="Capture Page Screenshot">
      <msg time="2026-01-01T00:00:00.000000" level="INFO" html="true">&lt;img src="data:image/png;base64,aGVs bG8="&gt;</msg>
      <status status="PASS" start="2026-01-01T00:00:00.000000" elapsed="0.001"/>
    </kw>
    <status status="FAIL" start="2026-01-01T00:00:00.001000" elapsed="0.001">boom</status>
  </test>
  <status status="FAIL" start="2026-01-01T00:00:00.000000" elapsed="0.002"/>
</suite>
<statistics><total><stat pass="0" fail="1" skip="0">All Tests</stat></total><tag/><suite><stat id="s1" pass="0" fail="1" skip="0">Suite</stat></suite></statistics>
<errors/>
</robot>
"""

_BROWSER_WRAPPED_SCREENSHOT_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<robot generator="Robot 7.1" rpa="false" schemaversion="5">
<suite id="s1" name="Suite">
  <test id="s1-t1" name="Failing Test">
    <kw name="Take Screenshot">
      <msg time="2026-01-01T00:00:00.000000" level="INFO" html="true">&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td colspan="3"&gt;&lt;img src="data:image/png;base64,aGVs bG8="&gt;</msg>
      <status status="PASS" start="2026-01-01T00:00:00.000000" elapsed="0.001"/>
    </kw>
    <status status="FAIL" start="2026-01-01T00:00:00.001000" elapsed="0.001">boom</status>
  </test>
  <status status="FAIL" start="2026-01-01T00:00:00.000000" elapsed="0.002"/>
</suite>
<statistics><total><stat pass="0" fail="1" skip="0">All Tests</stat></total><tag/><suite><stat id="s1" pass="0" fail="1" skip="0">Suite</stat></suite></statistics>
<errors/>
</robot>
"""

_PASSING_SCREENSHOT_XML = _SCREENSHOT_XML.replace(
    'name="Failing Test"', 'name="Passing Test"'
).replace('status="FAIL"', 'status="PASS"')

_NESTED_SUITE_SETUP_SCREENSHOT_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<robot generator="Robot 7.1" rpa="false" schemaversion="5">
<suite id="s1" name="Root">
  <kw name="Capture Page Screenshot" type="SETUP">
    <msg time="2026-01-01T00:00:00.000000" level="INFO" html="true">&lt;img src="data:image/png;base64,aGVsbG8="&gt;</msg>
    <status status="PASS" start="2026-01-01T00:00:00.000000" elapsed="0.001"/>
  </kw>
  <suite id="s1-s1" name="Child">
    <test id="s1-s1-t1" name="Failing Test">
      <status status="FAIL" start="2026-01-01T00:00:00.001000" elapsed="0.001">boom</status>
    </test>
    <status status="FAIL" start="2026-01-01T00:00:00.000000" elapsed="0.002"/>
  </suite>
  <status status="FAIL" start="2026-01-01T00:00:00.000000" elapsed="0.003"/>
</suite>
<statistics><total><stat pass="0" fail="1" skip="0">All Tests</stat></total><tag/><suite><stat id="s1" pass="0" fail="1" skip="0">Root</stat><stat id="s1-s1" pass="0" fail="1" skip="0">Child</stat></suite></statistics>
<errors/>
</robot>
"""

_DUPLICATE_TEST_SCREENSHOTS_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<robot generator="Robot 7.1" rpa="false" schemaversion="5">
<suite id="s1" name="Root">
  <suite id="s1-s1" name="A">
    <test id="s1-s1-t1" name="Same Test">
      <kw name="Capture Page Screenshot">
        <msg level="INFO" html="true">&lt;img src="data:image/png;base64,b25l"&gt;</msg>
        <msg level="INFO" html="true">&lt;img src="data:image/png;base64,dHdv"&gt;</msg>
        <status status="PASS" start="2026-01-01T00:00:00.000000" elapsed="0.001"/>
      </kw>
      <status status="FAIL" start="2026-01-01T00:00:00.001000" elapsed="0.001">boom</status>
    </test>
    <status status="FAIL" start="2026-01-01T00:00:00.000000" elapsed="0.002"/>
  </suite>
  <suite id="s1-s2" name="B">
    <test id="s1-s2-t1" name="Same Test">
      <kw name="Capture Page Screenshot">
        <msg level="INFO" html="true">&lt;img src="data:image/png;base64,dGhyZWU="&gt;</msg>
        <msg level="INFO" html="true">&lt;img src="data:image/png;base64,Zm91cg=="&gt;</msg>
        <status status="PASS" start="2026-01-01T00:00:00.002000" elapsed="0.001"/>
      </kw>
      <status status="FAIL" start="2026-01-01T00:00:00.003000" elapsed="0.001">boom</status>
    </test>
    <status status="FAIL" start="2026-01-01T00:00:00.002000" elapsed="0.002"/>
  </suite>
  <status status="FAIL" start="2026-01-01T00:00:00.000000" elapsed="0.004"/>
</suite>
<statistics><total><stat pass="0" fail="2" skip="0">All Tests</stat></total><tag/><suite><stat id="s1" pass="0" fail="2" skip="0">Root</stat></suite></statistics>
<errors/>
</robot>
"""

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeKeyword:
    """Duck-typed keyword for tests that don't require isinstance checks."""

    def __init__(
        self,
        name: str = "Keyword",
        status: str = "PASS",
        type: str = "kw",
        args: tuple = (),
        assign: tuple = (),
        body: tuple = (),
        doc: str = "",
        tags: tuple = (),
        elapsed_time=None,
        lineno=None,
    ):
        self.name = name
        self.status = status
        self.type = type
        self.args = args
        self.assign = assign
        self.body = body
        self.doc = doc
        self.tags = tags
        self.elapsed_time = elapsed_time
        self.lineno = lineno


class FakeTest:
    """Duck-typed test case."""

    def __init__(
        self,
        name: str = "My Test",
        status: str = "FAIL",
        doc: str = "",
        tags: tuple = (),
        has_setup: bool = False,
        setup=None,
        has_teardown: bool = False,
        teardown=None,
        body: tuple = (),
        elapsed_time=None,
        lineno=None,
        owner=None,
    ):
        self.name = name
        self.status = status
        self.doc = doc
        self.tags = tags
        self.has_setup = has_setup
        self.setup = setup
        self.has_teardown = has_teardown
        self.teardown = teardown
        self.body = body
        self.elapsed_time = elapsed_time
        self.lineno = lineno
        self.owner = owner


class FakeSuite:
    """Duck-typed test suite.

    Hits ContextAwareRobotResults.__init__ ``else`` branch (not a Path/str
    and not robot.result.model.TestSuite), so ``self._suite = source.suite``.
    """

    def __init__(
        self,
        name: str = "Suite",
        doc: str = "",
        has_setup: bool = False,
        setup=None,
        has_teardown: bool = False,
        teardown=None,
        tests: tuple | list = (),
        suites: tuple | list = (),
    ):
        self.name = name
        self.doc = doc
        self.has_setup = has_setup
        self.setup = setup
        self.has_teardown = has_teardown
        self.teardown = teardown
        self.tests = tests
        self.suites = suites

    @property
    def suite(self) -> FakeSuite:
        """Required by ContextAwareRobotResults else-branch."""
        return self

    @property
    def all_tests(self):
        """Recursively yields all tests (mirrors RF TestSuite.all_tests)."""
        yield from self.tests
        for child in self.suites:
            yield from child.all_tests


class FakeControlItem:
    """Non-Keyword, non-Message item with a body (control structure)."""

    def __init__(self, body: list = None):
        self.body = body or []


class FakeChunkingStrategy:
    """Minimal chunking strategy that returns text as a single chunk."""

    def apply(self, lines: list[tuple[int, str]]) -> tuple[list[str], Chunking]:
        text = render_lines_to_text(lines)
        stats = Chunking(
            chunk_size=0,
            number_of_chunks=1,
            raw_text_len=len(text),
            tokens_from_raw_text=1,
            tokenized_chunks=1,
        )
        return [text], stats


# ---------------------------------------------------------------------------
# _join_parts
# ---------------------------------------------------------------------------


class TestJoinParts:
    def test_all_parts_joined_with_dash(self):
        assert _join_parts("A", "B", "C") == "A - B - C"

    def test_none_parts_skipped(self):
        assert _join_parts("A", None, "C") == "A - C"

    def test_all_none_returns_empty_string(self):
        assert _join_parts(None, None) == ""

    def test_single_part_has_no_dash(self):
        assert _join_parts("Only") == "Only"


# ---------------------------------------------------------------------------
# _render_message
# ---------------------------------------------------------------------------


class TestRenderMessage:
    def test_returns_empty_when_message_not_in_fields(self):
        msg = RFMessage(message="log text", level="INFO")
        assert _render_message(msg, depth=0, fields=frozenset()) == []

    def test_renders_plain_message(self):
        msg = RFMessage(message="hello", level="INFO")
        result = _render_message(msg, depth=1, fields=frozenset({"message"}))
        assert result == [(1, "hello")]

    def test_prepends_level_bracket_when_level_in_fields(self):
        msg = RFMessage(message="hello", level="WARN")
        result = _render_message(msg, depth=0, fields=frozenset({"message", "level"}))
        assert result == [(0, "[WARN] hello")]

    def test_prepends_timestamp_when_present_and_in_fields(self):
        msg = RFMessage(message="hello", level="INFO", timestamp="2024-01-01T12:00:00")
        result = _render_message(
            msg, depth=0, fields=frozenset({"message", "level", "timestamp"})
        )
        assert len(result) == 1
        _, text = result[0]
        assert "[INFO]" in text and "hello" in text and "2024" in text

    def test_skips_timestamp_prefix_when_timestamp_is_none(self):
        msg = RFMessage(message="hello", level="INFO", timestamp=None)
        result = _render_message(
            msg, depth=0, fields=frozenset({"message", "level", "timestamp"})
        )
        assert result == [(0, "[INFO] hello")]

    def test_non_html_image_tag_is_stripped_without_placeholder(self):
        images = []
        context = RenderContext(
            suite_path=("Suite",),
            test_name="Test",
            include_images=True,
            collected_images=images,
        )
        msg = RFMessage(
            message='before <img src="data:image/png;base64,aGVsbG8="> after',
            level="INFO",
            html=False,
        )

        result = _render_message(
            msg, depth=0, fields=frozenset({"message"}), context=context
        )

        assert images == []
        assert result == [(0, "before  after")]


# ---------------------------------------------------------------------------
# _render_body_item
# ---------------------------------------------------------------------------


class TestRenderBodyItem:
    def test_message_dispatches_to_render_message(self):
        msg = RFMessage(message="log", level="INFO")
        result = _render_body_item(msg, depth=0, fields=frozenset({"message"}))
        assert result == [(0, "log")]

    def test_keyword_dispatches_to_render_keyword(self):
        kw = RFKeyword(name="My Kw", status="PASS")
        result = _render_body_item(kw, depth=0, fields=frozenset({"name", "status"}))
        assert any("My Kw" in text for _, text in result)

    def test_control_structure_with_body_recurses_into_children(self):
        inner_msg = RFMessage(message="inner", level="INFO")
        ctrl = FakeControlItem(body=[inner_msg])
        result = _render_body_item(ctrl, depth=1, fields=frozenset({"message"}))
        assert result == [(1, "inner")]

    def test_control_structure_with_empty_body_returns_empty(self):
        ctrl = FakeControlItem(body=[])
        result = _render_body_item(ctrl, depth=0, fields=frozenset({"message"}))
        assert result == []

    def test_control_structure_adds_child_image_path_then_restores_context(self):
        images = []
        context = RenderContext(
            suite_path=("Suite",),
            test_name="Test",
            keyword_path=("Keyword",),
            include_images=True,
            collected_images=images,
        )
        msg = RFMessage(
            message='<img src="data:image/png;base64,aGVsbG8=">',
            level="INFO",
            html=True,
        )
        ctrl = FakeControlItem(body=[msg])
        ctrl.type = "IF"
        ctrl.name = "branch"

        result = _render_body_item(
            ctrl, depth=1, fields=frozenset({"message"}), context=context
        )

        assert result == [(1, "[SCREENSHOT] embedded image/png screenshot #1")]
        assert len(images) == 1
        assert images[0].keyword_path == ("Keyword", "IF:branch")
        assert context.keyword_path == ("Keyword",)


def test_control_path_segment_uses_available_control_labels():
    named_item = type("ControlItem", (), {"type": "IF", "name": "branch"})()
    typed_item = type("ControlItem", (), {"type": "FOR", "name": None})()
    named_only_item = type("ControlItem", (), {"type": None, "name": "branch"})()

    assert _control_path_segment(named_item) == "IF:branch"
    assert _control_path_segment(typed_item) == "FOR"
    assert _control_path_segment(named_only_item) == "branch"


# ---------------------------------------------------------------------------
# _render_common_fields
# ---------------------------------------------------------------------------


class TestRenderCommonFields:
    def test_renders_all_common_fields_when_present(self):
        kw = FakeKeyword(
            doc="my doc", tags=["smoke"], elapsed_time="0:00:01", lineno=42
        )
        fields = frozenset({"doc", "tags", "elapsed_time", "lineno"})

        result = _render_common_fields(kw, depth=1, fields=fields)
        texts = [t for _, t in result]

        assert any("0:00:01" in t for t in texts)
        assert any("42" in t for t in texts)
        assert any("my doc" in t for t in texts)
        assert any("smoke" in t for t in texts)

    def test_returns_empty_when_all_values_absent(self):
        kw = FakeKeyword(doc="", tags=[], elapsed_time=None, lineno=None)
        result = _render_common_fields(kw, depth=0, fields=ALL_FIELDS)
        assert result == []

    def test_elapsed_time_zero_is_rendered_not_dropped(self):
        from datetime import timedelta

        kw = FakeKeyword(elapsed_time=timedelta(0))
        result = _render_common_fields(kw, depth=0, fields=frozenset({"elapsed_time"}))
        assert len(result) == 1
        assert "elapsed" in result[0][1]


# ---------------------------------------------------------------------------
# _render_keyword
# ---------------------------------------------------------------------------


class TestRenderKeyword:
    def test_renders_name_and_status_header(self):
        kw = FakeKeyword(name="Connect DB", status="PASS")
        result = _render_keyword(kw, depth=0, fields=frozenset({"name", "status"}))
        assert result[0] == (0, "Keyword: Connect DB - PASS")

    def test_uses_type_title_when_type_in_fields(self):
        kw = FakeKeyword(name="Suite Setup", type="setup")
        result = _render_keyword(kw, depth=0, fields=frozenset({"name", "type"}))
        assert result[0][1].startswith("Setup:")

    def test_falls_back_to_keyword_label_when_type_not_in_fields(self):
        kw = FakeKeyword(name="Suite Setup", type="setup")
        result = _render_keyword(kw, depth=0, fields=frozenset({"name"}))
        assert result[0][1].startswith("Keyword:")

    def test_renders_args_line(self):
        kw = FakeKeyword(name="Kw", args=("arg1", "arg2"))
        result = _render_keyword(kw, depth=0, fields=frozenset({"name", "args"}))
        texts = [t for _, t in result]
        assert any("arg1" in t and "arg2" in t for t in texts)

    def test_renders_assign_line(self):
        kw = FakeKeyword(name="Kw", assign=("${result}",))
        result = _render_keyword(kw, depth=0, fields=frozenset({"name", "assign"}))
        texts = [t for _, t in result]
        assert any("${result}" in t for t in texts)

    def test_name_omitted_when_not_in_fields(self):
        kw = FakeKeyword(name="Secret", status="PASS")
        result = _render_keyword(kw, depth=0, fields=frozenset({"status"}))
        assert "Secret" not in result[0][1]


# ---------------------------------------------------------------------------
# _render_test
# ---------------------------------------------------------------------------


class TestRenderTest:
    def test_renders_name_and_status_header(self):
        test = FakeTest(name="Login Test", status="FAIL")
        result = _render_test(test, depth=0, fields=frozenset({"name", "status"}))
        assert result[0] == (0, "Test: Login Test - FAIL")

    def test_renders_setup_and_teardown_keywords(self):
        setup = FakeKeyword(name="Suite Setup", type="setup")
        teardown = FakeKeyword(name="Suite Teardown", type="teardown")
        test = FakeTest(
            has_setup=True, setup=setup, has_teardown=True, teardown=teardown
        )
        fields = frozenset({"name", "status", "setup", "teardown"})

        result = _render_test(test, depth=0, fields=fields)
        texts = [t for _, t in result]

        assert any("Suite Setup" in t for t in texts)
        assert any("Suite Teardown" in t for t in texts)

    def test_body_items_with_setup_teardown_type_are_skipped(self):
        # Items already handled via has_setup/has_teardown must not appear twice.
        setup_item = FakeKeyword(name="Should Not Appear", type="setup")
        test = FakeTest(body=[setup_item])
        fields = frozenset({"name", "status"})

        result = _render_test(test, depth=0, fields=fields)
        texts = [t for _, t in result]

        assert not any("Should Not Appear" in t for t in texts)

    def test_renders_owner(self):
        test = FakeTest(name="T", status="PASS", owner="alice")
        result = _render_test(
            test, depth=0, fields=frozenset({"name", "status", "owner"})
        )
        texts = [t for _, t in result]
        assert any("alice" in t for t in texts)


# ---------------------------------------------------------------------------
# _render_suite
# ---------------------------------------------------------------------------


class TestRenderSuite:
    def test_renders_suite_name_header(self):
        suite = FakeSuite(name="My Suite")
        result = _render_suite(suite, depth=0, fields=frozenset({"name"}))
        assert result[0] == (0, "Suite: My Suite")

    def test_renders_doc_when_present(self):
        suite = FakeSuite(name="S", doc="suite description")
        result = _render_suite(suite, depth=0, fields=frozenset({"name", "doc"}))
        texts = [t for _, t in result]
        assert any("suite description" in t for t in texts)

    def test_renders_tests_under_suite(self):
        test = FakeTest(name="T1", status="FAIL")
        suite = FakeSuite(name="S", tests=[test])
        result = _render_suite(suite, depth=0, fields=frozenset({"name", "status"}))
        texts = [t for _, t in result]
        assert any("T1" in t for t in texts)

    def test_renders_child_suites_recursively(self):
        child = FakeSuite(name="Child")
        suite = FakeSuite(name="Parent", suites=[child])
        result = _render_suite(suite, depth=0, fields=frozenset({"name"}))
        texts = [t for _, t in result]
        assert any("Parent" in t for t in texts)
        assert any("Child" in t for t in texts)

    def test_suite_teardown_rendered_after_tests(self):
        teardown = FakeKeyword(name="Suite Tear", type="teardown")
        test = FakeTest(name="T", status="PASS")
        suite = FakeSuite(name="S", has_teardown=True, teardown=teardown, tests=[test])
        fields = frozenset({"name", "status", "teardown"})

        result = _render_suite(suite, depth=0, fields=fields)
        texts = [t for _, t in result]
        teardown_idx = next(i for i, t in enumerate(texts) if "Suite Tear" in t)
        test_idx = next(i for i, t in enumerate(texts) if "Test:" in t)
        assert teardown_idx > test_idx


# ---------------------------------------------------------------------------
# _iter_tests_with_context
# ---------------------------------------------------------------------------


class TestIterTestsWithContext:
    def test_yields_test_with_suite_name_in_context(self):
        test = FakeTest(name="T1", status="FAIL")
        suite = FakeSuite(name="Root", tests=[test])

        results = list(
            _iter_tests_with_context(suite, [], 0, frozenset({"name", "status"}))
        )

        assert len(results) == 1
        name, status, lines = results[0]
        assert name == "T1"
        assert status == "FAIL"
        texts = [t for _, t in lines]
        assert any("Suite: Root" in t for t in texts)
        assert any("Test: T1 - FAIL" in t for t in texts)

    def test_suite_setup_prepended_to_each_test_context(self):
        setup = FakeKeyword(name="Suite Setup", type="setup", status="PASS")
        test = FakeTest(name="T", status="PASS")
        suite = FakeSuite(name="S", has_setup=True, setup=setup, tests=[test])
        fields = frozenset({"name", "status", "setup"})

        _, _, lines = list(_iter_tests_with_context(suite, [], 0, fields))[0]
        texts = [t for _, t in lines]

        assert any("Suite Setup" in t for t in texts)

    def test_nested_suite_tests_include_outer_suite_name(self):
        inner_test = FakeTest(name="Inner T", status="PASS")
        inner_suite = FakeSuite(name="Inner", tests=[inner_test])
        outer_suite = FakeSuite(name="Outer", suites=[inner_suite])

        results = list(
            _iter_tests_with_context(outer_suite, [], 0, frozenset({"name", "status"}))
        )

        assert len(results) == 1
        _, _, lines = results[0]
        texts = [t for _, t in lines]
        assert any("Outer" in t for t in texts)
        assert any("Inner" in t for t in texts)

    def test_yields_nothing_for_empty_suite(self):
        suite = FakeSuite(name="Empty")
        assert list(_iter_tests_with_context(suite, [], 0, frozenset({"name"}))) == []

    def test_failing_suite_setup_yields_suite_as_single_unit_not_individual_tests(self):
        failing_setup = FakeKeyword(
            name="Suite Setup Fails", status="FAIL", type="setup"
        )
        suite_with_failing_setup = FakeSuite(
            name="Suite With Failing Setup",
            has_setup=True,
            setup=failing_setup,
            tests=[
                FakeTest(name=f"Never Reached {i}", status="FAIL", body=())
                for i in range(4)
            ],
        )
        normal_suite = FakeSuite(
            name="Normal Suite",
            tests=[FakeTest(name="Normal Test", status="FAIL")],
        )
        root = FakeSuite(name="Root", suites=[normal_suite, suite_with_failing_setup])

        results = list(
            _iter_tests_with_context(
                root, [], 0, frozenset({"name", "status", "setup"})
            )
        )

        names = [name for name, _, _ in results]
        assert len(names) == 2
        assert "Normal Test" in names
        assert "Suite With Failing Setup" in names

    def test_duplicate_suite_names_with_failing_setups_get_unique_names(self):
        setup_a = RFKeyword(name="Setup", type="SETUP")
        setup_a.status = "FAIL"
        setup_a.elapsed_time = "0:00:00.001000"
        setup_b = RFKeyword(name="Setup", type="SETUP")
        setup_b.status = "FAIL"
        setup_b.elapsed_time = "0:00:00.001000"
        child1 = RFTestSuite(name="Shared Name")
        child1.setup = setup_a
        child1.tests.append(RFTestCase(name="Test A", status="FAIL"))
        child2 = RFTestSuite(name="Shared Name")
        child2.setup = setup_b
        child2.tests.append(RFTestCase(name="Test B", status="FAIL"))

        root = RFTestSuite(name="Root")
        root.suites.extend([child1, child2])
        root.visit(UniqueNameResultVisitor())
        results = ContextAwareRobotResults(root)

        assert results._suite.name == "Root"
        assert results._suite.has_setup is False
        assert results.test_names == ["Shared Name", "Shared Name s1-s2"]
        assert results._suite.suites[0].name == "Shared Name"
        assert results._suite.suites[1].name == "Shared Name s1-s2"
        assert list(results.as_texts()) == [
            (
                "Shared Name",
                "Suite: Root\n    Suite: Shared Name\n        Setup: Setup - FAIL\n            elapsed: 0:00:00.001000",
            ),
            (
                "Shared Name s1-s2",
                "Suite: Root\n    Suite: Shared Name s1-s2\n        Setup: Setup - FAIL\n            elapsed: 0:00:00.001000",
            ),
        ]

    def test_root_setup_failure_with_nested_suites_collapses_to_single_unit(self):
        setup = RFKeyword(name="Setup", type="SETUP")
        setup.status = "FAIL"
        setup.elapsed_time = "0:00:00.001000"
        child1 = RFTestSuite(name="Shared Name")
        # Tests which shouldn't be visible - below setup, inherits setup failure
        child1.tests.append(
            RFTestCase(name="Test A - Should not be visible!", status="FAIL")
        )
        child2 = RFTestSuite(name="Shared Name")
        child2.tests.append(
            RFTestCase(name="Test B - Should not be visible!", status="FAIL")
        )

        root = RFTestSuite(name="Root")
        root.setup = setup
        root.suites.extend([child1, child2])

        results = ContextAwareRobotResults(root)

        assert results._suite.has_setup is True
        assert len(results._suite.suites) == 2
        assert results.test_names == ["Root"]
        assert list(results.as_texts()) == [
            (
                "Root",
                "Suite: Root\n    Setup: Setup - FAIL\n        elapsed: 0:00:00.001000",
            ),
        ]

    def test_root_suite_setup_failure_collapses_all_tests_into_single_unit(
        self, tmp_path
    ):
        xml = tmp_path / "output.xml"
        xml.write_text(
            """\
<?xml version="1.0" encoding="UTF-8"?>
<robot generator="Robot 7.1" rpa="false" schemaversion="5">
<suite id="s1" name="Root">
  <kw name="Setup" type="SETUP"><status status="FAIL" start="2026-01-01T00:00:00.000000" elapsed="0.001">boom</status></kw>
  <test id="s1-t1" name="Test A"><status status="FAIL" start="2026-01-01T00:00:00.001000" elapsed="0.001">Parent suite setup failed: boom</status></test>
  <test id="s1-t2" name="Test B"><status status="FAIL" start="2026-01-01T00:00:00.002000" elapsed="0.001">Parent suite setup failed: boom</status></test>
  <status status="FAIL" start="2026-01-01T00:00:00.000000" elapsed="0.003"/>
</suite>
<statistics><total><stat pass="0" fail="2" skip="0">All Tests</stat></total><tag/><suite><stat id="s1" pass="0" fail="2" skip="0">Root</stat></suite></statistics>
<errors/>
</robot>
"""
        )
        results = ContextAwareRobotResults(xml)

        assert results.test_names == ["Root"]
        assert ["Test A", "Test B"] not in results.test_names
        assert list(results.as_texts()) == [
            (
                "Root",
                "Suite: Root\n    Setup: Setup - FAIL\n        elapsed: 0:00:00.001000",
            ),
        ]

    def test_failing_suite_setup_collapses_even_when_setup_field_excluded(self):
        failing_setup = FakeKeyword(name="Suite Setup", status="FAIL", type="setup")
        suite = FakeSuite(
            name="Bad Suite",
            has_setup=True,
            setup=failing_setup,
            tests=[FakeTest(name="Never Reached", status="FAIL")],
        )

        results = list(
            _iter_tests_with_context(suite, [], 0, frozenset({"name", "status"}))
        )

        assert len(results) == 1
        name, status, _ = results[0]
        assert name == "Bad Suite"
        assert status == "FAIL"

    def test_failing_suite_setup_includes_own_teardown_in_collapsed_output(self):
        failing_setup = FakeKeyword(name="Suite Setup", status="FAIL", type="setup")
        teardown = FakeKeyword(name="Suite Teardown", status="PASS", type="teardown")
        suite = FakeSuite(
            name="Bad Suite",
            has_setup=True,
            setup=failing_setup,
            has_teardown=True,
            teardown=teardown,
            tests=[FakeTest(name="Never Reached", status="FAIL")],
        )
        fields = frozenset({"name", "status", "setup", "teardown"})

        _, _, lines = list(_iter_tests_with_context(suite, [], 0, fields))[0]
        texts = [t for _, t in lines]

        assert any("Suite Teardown" in t for t in texts)

    def test_failing_suite_setup_includes_ancestor_teardowns_in_collapsed_output(self):
        outer_teardown = FakeKeyword(name="Outer Tear", status="PASS", type="teardown")
        failing_setup = FakeKeyword(name="Inner Setup", status="FAIL", type="setup")
        inner_suite = FakeSuite(
            name="Inner",
            has_setup=True,
            setup=failing_setup,
            tests=[FakeTest(name="Never Reached", status="FAIL")],
        )
        outer_suite = FakeSuite(
            name="Outer",
            has_teardown=True,
            teardown=outer_teardown,
            suites=[inner_suite],
        )
        fields = frozenset({"name", "status", "setup", "teardown"})

        results = list(_iter_tests_with_context(outer_suite, [], 0, fields))
        _, _, lines = results[0]
        texts = [t for _, t in lines]

        assert any("Outer Tear" in t for t in texts)

    def test_passing_suite_setup_still_iterates_individual_tests(self):
        passing_setup = FakeKeyword(name="Suite Setup", status="PASS", type="setup")
        suite = FakeSuite(
            name="Suite",
            has_setup=True,
            setup=passing_setup,
            tests=[FakeTest(name="Test A"), FakeTest(name="Test B")],
        )

        results = list(
            _iter_tests_with_context(
                suite, [], 0, frozenset({"name", "status", "setup"})
            )
        )

        assert [n for n, _, _ in results] == ["Test A", "Test B"]

    def test_suite_teardown_appended_after_test_body_not_before(self):
        teardown = FakeKeyword(name="Suite Tear", type="teardown", status="FAIL")
        test = FakeTest(name="T", status="PASS")
        suite = FakeSuite(name="S", has_teardown=True, teardown=teardown, tests=[test])
        fields = frozenset({"name", "status", "teardown"})

        _, _, lines = list(_iter_tests_with_context(suite, [], 0, fields))[0]
        texts = [t for _, t in lines]
        teardown_idx = next(i for i, t in enumerate(texts) if "Suite Tear" in t)
        test_idx = next(i for i, t in enumerate(texts) if "Test:" in t)

        assert teardown_idx > test_idx

    def test_nested_suite_tests_include_ancestor_teardowns_after_test_body(self):
        outer_teardown = FakeKeyword(name="Outer Tear", type="teardown", status="FAIL")
        inner_teardown = FakeKeyword(name="Inner Tear", type="teardown", status="PASS")
        inner_test = FakeTest(name="T", status="PASS")
        inner_suite = FakeSuite(
            name="Inner", has_teardown=True, teardown=inner_teardown, tests=[inner_test]
        )
        outer_suite = FakeSuite(
            name="Outer",
            has_teardown=True,
            teardown=outer_teardown,
            suites=[inner_suite],
        )
        fields = frozenset({"name", "status", "teardown"})

        _, _, lines = list(_iter_tests_with_context(outer_suite, [], 0, fields))[0]
        texts = [t for _, t in lines]
        test_idx = next(i for i, t in enumerate(texts) if "Test:" in t)
        inner_tear_idx = next(i for i, t in enumerate(texts) if "Inner Tear" in t)
        outer_tear_idx = next(i for i, t in enumerate(texts) if "Outer Tear" in t)

        assert test_idx < inner_tear_idx < outer_tear_idx


# ---------------------------------------------------------------------------
# TestLines
# ---------------------------------------------------------------------------


class TestTestLines:
    def test_str_renders_depth_indented_text(self):
        tl = TestLines(name="T", lines=[(0, "Suite: S"), (1, "Test: T - PASS")])
        assert str(tl) == "Suite: S\n    Test: T - PASS"


# ---------------------------------------------------------------------------
# ContextAwareRobotResults
# ---------------------------------------------------------------------------


class TestContextAwareRobotResults:
    def _make(self, tests=(), suites=()):
        suite = FakeSuite(name="Root", tests=list(tests), suites=list(suites))
        return ContextAwareRobotResults(suite)

    def test_iter_yields_test_name_and_test_lines(self):
        results = self._make(tests=[FakeTest(name="T1", status="FAIL")])

        items = list(results)

        assert len(items) == 1
        name, tl = items[0]
        assert name == "T1"
        assert isinstance(tl, TestLines)

    def test_as_texts_yields_name_and_rendered_string(self):
        results = self._make(tests=[FakeTest(name="T1", status="FAIL")])

        pairs = list(results.as_texts())

        assert len(pairs) == 1
        name, text = pairs[0]
        assert name == "T1"
        assert "T1" in text

    def test_exclude_passing_skips_pass_tests(self):
        results = self._make(
            tests=[
                FakeTest(name="Pass", status="PASS"),
                FakeTest(name="Fail", status="FAIL"),
            ]
        )
        results.exclude_passing()

        assert [n for n, _ in results] == ["Fail"]

    def test_exclude_passing_also_skips_skip_status_tests(self):
        results = self._make(
            tests=[
                FakeTest(name="Skip", status="SKIP"),
                FakeTest(name="Fail", status="FAIL"),
            ]
        )
        results.exclude_passing()

        assert [n for n, _ in results] == ["Fail"]

    def test_exclude_passing_false_keeps_all_tests(self):
        results = self._make(tests=[FakeTest(name="Pass", status="PASS")])
        results.exclude_passing(False)

        assert [n for n, _ in results] == ["Pass"]

    def test_include_fields_replaces_entire_field_set(self):
        results = self._make()
        results.include_fields(["name", "status"])
        assert results._fields == frozenset({"name", "status"})

    def test_exclude_fields_removes_from_active_set(self):
        results = self._make()
        before = results._fields
        results.exclude_fields(["doc", "tags"])
        assert results._fields == before - frozenset({"doc", "tags"})

    def test_has_chunking_is_false_by_default(self):
        assert self._make().has_chunking is False

    def test_has_chunking_is_true_after_set_chunking(self):
        results = self._make()
        results.set_chunking(FakeChunkingStrategy())
        assert results.has_chunking is True

    def test_render_chunks_raises_when_no_strategy_set(self):
        results = self._make(tests=[FakeTest()])
        with pytest.raises(ValueError, match="set_chunking"):
            list(results.render_chunks())

    def test_render_chunks_yields_name_chunks_stats_status(self):
        results = self._make(tests=[FakeTest(name="T1", status="FAIL")])
        results.set_chunking(FakeChunkingStrategy())

        items = list(results.render_chunks())

        assert len(items) == 1
        test_name, chunks, chunk_stats, test_status = items[0]
        assert test_name == "T1"
        assert test_status == "FAIL"
        assert isinstance(chunks, list)

    def test_render_chunks_status_comes_from_model_not_rendered_text(self):
        # Status comes from test.status directly, not parsed from rendered lines.
        results = self._make(tests=[FakeTest(name="T", status="FAIL")])
        results.include_fields([])
        results.set_chunking(FakeChunkingStrategy())

        _, _, _, test_status = list(results.render_chunks())[0]

        assert test_status == "FAIL"

    def test_total_test_count_counts_all_tests_ignoring_exclude_passing(self):
        tests = [FakeTest(name="P", status="PASS"), FakeTest(name="F", status="FAIL")]
        results = self._make(tests=tests)
        results.exclude_passing()

        assert results.total_test_count == 2

    def test_test_names_returns_all_names_by_default(self):
        tests = [FakeTest(name="A", status="FAIL"), FakeTest(name="B", status="PASS")]
        results = self._make(tests=tests)
        assert results.test_names == ["A", "B"]

    def test_test_names_respects_exclude_passing_filter(self):
        tests = [FakeTest(name="A", status="FAIL"), FakeTest(name="B", status="PASS")]
        results = self._make(tests=tests)
        results.exclude_passing()
        assert results.test_names == ["A"]

    def test_source_hash_is_stable_across_calls(self):
        results = self._make(tests=[FakeTest(name="T", status="FAIL")])
        assert results.source_hash == results.source_hash

    def test_source_hash_is_12_chars(self):
        assert len(self._make().source_hash) == 12

    def test_source_hash_for_non_file_source_uses_rendered_fallback(self):
        results = self._make(tests=[FakeTest(name="T", status="FAIL")])
        expected = hashlib.sha256(str(results).encode()).hexdigest()[:12]

        assert results.source_hash == expected

    def test_source_hash_for_non_file_source_ignores_field_filters(self):
        results = self._make(tests=[FakeTest(name="T", status="FAIL")])
        source_hash = results.source_hash

        results.include_fields([])

        assert results.source_hash == source_hash

    def test_str_renders_suite_and_test_content(self):
        test = FakeTest(name="T", status="FAIL")
        suite = FakeSuite(name="S", tests=[test])
        results = ContextAwareRobotResults(suite)

        text = str(results)

        assert "Suite: S" in text
        assert "Test: T" in text


# ---------------------------------------------------------------------------
# ContextAwareRobotResults — source paths
# ---------------------------------------------------------------------------


class TestContextAwareRobotResultsSourcePaths:
    def test_path_source_parses_xml_and_sets_result(self, tmp_path):
        xml = tmp_path / "output.xml"
        xml.write_text(_MINIMAL_XML)

        results = ContextAwareRobotResults(xml)

        assert results._result is not None
        assert results.total_test_count == 2

    def test_path_source_hash_uses_raw_file_sha256(self, tmp_path):
        xml = tmp_path / "output.xml"
        xml.write_text(_MINIMAL_XML)
        expected = hashlib.sha256(_MINIMAL_XML.encode()).hexdigest()[:12]

        results = ContextAwareRobotResults(xml)

        assert results.source_hash == expected

    def test_path_source_hash_ignores_analysis_filters(self, tmp_path):
        xml = tmp_path / "output.xml"
        xml.write_text(_MINIMAL_XML)
        results = ContextAwareRobotResults(xml)
        source_hash = results.source_hash

        results.include_tags(["critical"])
        results.exclude_fields(["status"])
        results.exclude_passing()

        assert results.source_hash == source_hash

    def test_execution_result_source_uses_suite_directly(self, tmp_path):
        xml = tmp_path / "output.xml"
        xml.write_text(_MINIMAL_XML)
        execution_result = ExecutionResult(xml)

        results = ContextAwareRobotResults(execution_result)

        assert results._result is execution_result
        assert results.total_test_count == 2

    def test_test_suite_source_sets_result_to_none(self):
        suite = RFTestSuite(name="Empty")

        results = ContextAwareRobotResults(suite)

        assert results._result is None
        assert results._suite is suite


# ---------------------------------------------------------------------------
# ContextAwareRobotResults — embedded images
# ---------------------------------------------------------------------------


class TestContextAwareRobotResultsEmbeddedImages:
    def test_embedded_image_id_uses_render_context_and_payload(self):
        context = RenderContext(
            suite_path=("Root", "Suite"),
            test_name="Failing Test",
            keyword_path=("Capture Page Screenshot",),
        )
        expected = hashlib.sha256(
            "\0".join(
                (
                    "Root",
                    "Suite",
                    "Failing Test",
                    "Capture Page Screenshot",
                    "2",
                    "1",
                    "aGVsbG8=",
                )
            ).encode()
        ).hexdigest()[:24]

        image_id = _embedded_image_id(context, 2, 1, "aGVsbG8=")

        assert image_id == expected

    def test_embedded_image_tags_are_stripped_by_default(self, tmp_path):
        xml = tmp_path / "output.xml"
        xml.write_text(_SCREENSHOT_XML)

        text = list(ContextAwareRobotResults(xml).as_texts())[0][1]

        assert "aGVs" not in text
        assert "[SCREENSHOT]" not in text
        assert "Capture Page Screenshot" in text

    def test_include_embedded_images_renders_placeholder_under_keyword(self, tmp_path):
        xml = tmp_path / "output.xml"
        xml.write_text(_SCREENSHOT_XML)
        results = ContextAwareRobotResults(xml).include_embedded_images()

        images = results.collect_embedded_images()
        text = list(results.as_texts())[0][1]

        assert len(images) == 1
        assert images[0].test_identity == ("Suite", "Failing Test")
        assert images[0].keyword_path == ("Capture Page Screenshot",)
        assert images[0].data_base64 == "aGVsbG8="
        assert "Keyword: Capture Page Screenshot - PASS" in text
        assert "    [SCREENSHOT] embedded image/png screenshot #1" in text
        assert text.index("Capture Page Screenshot") < text.index("[SCREENSHOT]")

    def test_parent_suite_setup_screenshot_is_rendered_for_child_tests(self, tmp_path):
        xml = tmp_path / "output.xml"
        xml.write_text(_NESTED_SUITE_SETUP_SCREENSHOT_XML)
        results = ContextAwareRobotResults(xml).include_embedded_images()

        images = results.collect_embedded_images()
        text = list(results.as_texts())[0][1]

        assert len(images) == 1
        assert images[0].test_name == "Failing Test"
        assert images[0].test_identity == ("Root", "Child", "Failing Test")
        assert images[0].keyword_path == ("Capture Page Screenshot",)
        assert "[SCREENSHOT] embedded image/png screenshot #1" in text

    def test_collect_embedded_images_caps_per_test_identity(self, tmp_path):
        xml = tmp_path / "output.xml"
        xml.write_text(_DUPLICATE_TEST_SCREENSHOTS_XML)
        results = ContextAwareRobotResults(xml)

        all_images = results.collect_embedded_images()
        capped_images = results.collect_embedded_images(max_per_test=1)

        assert [image.data_base64 for image in all_images] == [
            "b25l",
            "dHdv",
            "dGhyZWU=",
            "Zm91cg==",
        ]
        assert [image.data_base64 for image in capped_images] == [
            "b25l",
            "dGhyZWU=",
        ]
        assert capped_images[0].test_identity != capped_images[1].test_identity

    def test_screenshot_only_message_does_not_render_empty_info_line(self, tmp_path):
        xml = tmp_path / "output.xml"
        xml.write_text(_SCREENSHOT_XML)

        text = list(ContextAwareRobotResults(xml).include_embedded_images().as_texts())[
            0
        ][1]

        assert "[INFO]" not in text
        assert "[SCREENSHOT]" in text

    def test_browser_screenshot_wrapper_html_is_removed(self, tmp_path):
        xml = tmp_path / "output.xml"
        xml.write_text(_BROWSER_WRAPPED_SCREENSHOT_XML)

        text = list(ContextAwareRobotResults(xml).include_embedded_images().as_texts())[
            0
        ][1]

        assert "[SCREENSHOT] embedded image/png screenshot #1" in text
        assert "</td>" not in text
        assert "<tr" not in text
        assert "colspan" not in text
        assert "[INFO]" not in text

    def test_attach_image_texts_renders_ocr_lines_next_to_placeholder(self, tmp_path):
        xml = tmp_path / "output.xml"
        xml.write_text(_SCREENSHOT_XML)
        results = ContextAwareRobotResults(xml)
        image = results.collect_embedded_images()[0]

        text = list(
            results.attach_image_texts({image.id: "Login\nPassword"}).as_texts()
        )[0][1]

        assert "[SCREENSHOT] embedded image/png screenshot #1" in text
        assert "[SCREENSHOT_OCR] Login" in text
        assert "[SCREENSHOT_OCR] Password" in text

    def test_collect_embedded_images_respects_exclude_passing(self, tmp_path):
        xml = tmp_path / "output.xml"
        xml.write_text(_PASSING_SCREENSHOT_XML)
        results = ContextAwareRobotResults(xml).exclude_passing()

        assert results.collect_embedded_images() == []

    def test_source_hash_ignores_attached_image_texts(self, tmp_path):
        xml = tmp_path / "output.xml"
        xml.write_text(_SCREENSHOT_XML)
        results = ContextAwareRobotResults(xml)
        image = results.collect_embedded_images()[0]
        source_hash = results.source_hash

        results.attach_image_texts({image.id: "Login"})

        assert results.source_hash == source_hash

    def test_strips_embedded_base64_from_full_suite_rendering(self, tmp_path):
        xml = tmp_path / "output.xml"
        xml.write_text(_SCREENSHOT_XML)

        text = str(ContextAwareRobotResults(xml))

        assert "aGVs" not in text
        assert "<img" not in text


# ---------------------------------------------------------------------------
# _apply_config / include_tags / exclude_tags
# ---------------------------------------------------------------------------


class TestApplyConfig:
    def test_raise_type_error_on_filtering_on_TestSuite_object(self):
        results = ContextAwareRobotResults(RFTestSuite(name="S"))
        with pytest.raises(
            TypeError,
            match="Source is TestSuite, not ExecutionResult, TAG filtering is not available!",
        ):
            results.include_tags(["smoke"])

    def test_raises_value_error_on_data_error(self, tmp_path, monkeypatch):
        xml = tmp_path / "output.xml"
        xml.write_text(_MINIMAL_XML)
        results = ContextAwareRobotResults(xml)

        monkeypatch.setattr(
            results._result,
            "configure",
            lambda **_: (_ for _ in ()).throw(DataError("no match")),
        )

        with pytest.raises(ValueError, match="Tag filter"):
            results.include_tags(["nonexistent"])

    def test_include_tags_filters_to_matching_tests(self, tmp_path):
        xml = tmp_path / "output.xml"
        xml.write_text(_MINIMAL_XML)
        results = ContextAwareRobotResults(xml)

        results.include_tags(["smoke"])

        assert results.total_test_count == 1
        assert results.test_names == ["Passing Test"]

    def test_exclude_tags_removes_matching_tests(self, tmp_path):
        xml = tmp_path / "output.xml"
        xml.write_text(_MINIMAL_XML)
        results = ContextAwareRobotResults(xml)

        results.exclude_tags(["smoke"])

        assert results.total_test_count == 1
        assert results.test_names == ["Failing Test"]


# ---------------------------------------------------------------------------
# get_rc_robot_results — all options
# ---------------------------------------------------------------------------


class TestGetRcRobotResults:
    def test_all_options_applied(self, tmp_path):
        xml = tmp_path / "output.xml"
        xml.write_text(_MINIMAL_XML)

        results = get_rc_robot_results(
            file_path=xml,
            include_tags=["critical"],
            exclude_fields=["doc"],
            exclude_passing=True,
            chunking_strategy=FakeChunkingStrategy(),
        )

        assert results.has_chunking
        assert "doc" not in results._fields
        assert results.test_names == ["Failing Test"]
