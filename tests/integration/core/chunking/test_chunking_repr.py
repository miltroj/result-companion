from __future__ import annotations

import pytest

from result_companion.core.chunking.chunking import ChunkingStrategy
from result_companion.core.chunking.rf_results import (
    ContextAwareRobotResults,
    get_rc_robot_results,
)
from result_companion.core.parsers.config import TokenizerModel, TokenizerTypes

RENDER_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<robot generator="Robot 7.1" rpa="false" schemaversion="5">
<suite id="s1" name="My Application">
  <suite id="s1-s1" name="Data Pipeline">
    <test id="s1-s1-t1" name="Processes Data And Writes Output">
      <kw name="Connect To Database">
        <msg time="2026-01-01T00:00:00.000000" level="WARN">DeprecationWarning: Call to deprecated create function.</msg>
        <msg time="2026-01-01T00:00:00.001000" level="WARN">DeprecationWarning: Call to deprecated create function.</msg>
        <msg time="2026-01-01T00:00:00.002000" level="WARN">DeprecationWarning: Call to deprecated create function.</msg>
        <status status="PASS" start="2026-01-01T00:00:00.000000" elapsed="0.003"/>
      </kw>
      <kw name="Write Output">
        <msg time="2026-01-01T00:00:00.003000" level="INFO">Writing 1000 rows to output table.</msg>
        <status status="FAIL" start="2026-01-01T00:00:00.003000" elapsed="0.001"/>
      </kw>
      <status status="FAIL" start="2026-01-01T00:00:00.000000" elapsed="0.004"/>
    </test>
    <status status="FAIL" start="2026-01-01T00:00:00.000000" elapsed="0.004"/>
  </suite>
  <status status="FAIL" start="2026-01-01T00:00:00.000000" elapsed="0.004"/>
</suite>
<statistics>
  <total><stat pass="0" fail="1" skip="0">All Tests</stat></total>
  <tag/>
  <suite><stat id="s1" pass="0" fail="1" skip="0">My Application</stat></suite>
</statistics>
<errors/>
</robot>
"""

TAGGED_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<robot generator="Robot 7.1" rpa="false" schemaversion="5">
<suite id="s1" name="Tests">
  <test id="s1-t1" name="Smoke Test"><tag>smoke</tag><status status="FAIL"/></test>
  <test id="s1-t2" name="WIP Test"><tag>wip</tag><status status="FAIL"/></test>
  <test id="s1-t3" name="Regression"><tag>regression</tag><status status="FAIL"/></test>
  <test id="s1-t4" name="Smoke WIP"><tag>smoke</tag><tag>wip</tag><status status="FAIL"/></test>
  <status status="FAIL"/>
</suite>
<statistics>
  <total><stat pass="0" fail="4" skip="0">All Tests</stat></total>
  <tag/>
  <suite><stat id="s1" pass="0" fail="4" skip="0">Tests</stat></suite>
</statistics>
<errors/>
</robot>
"""

SUITE_SETUP_FAIL_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<robot generator="Robot 7.1" rpa="false" schemaversion="5">
<suite id="s1" name="Root">
  <kw name="Setup" type="SETUP"><status status="FAIL" start="2026-01-01T00:00:00.000000" elapsed="0.001">boom</status></kw>
  <test id="s1-t1" name="Test A"><status status="FAIL" start="2026-01-01T00:00:00.001000" elapsed="0.001">Parent suite setup failed: boom</status></test>
  <test id="s1-t2" name="Test B"><status status="FAIL" start="2026-01-01T00:00:00.002000" elapsed="0.001">Parent suite setup failed: boom</status></test>
  <status status="FAIL" start="2026-01-01T00:00:00.000000" elapsed="0.003"/>
</suite>
<statistics>
  <total><stat pass="0" fail="2" skip="0">All Tests</stat></total>
  <tag/>
  <suite><stat id="s1" pass="0" fail="2" skip="0">Root</stat></suite>
</statistics>
<errors/>
</robot>
"""

# Ollama tokenizer = len(text)//4 — deterministic, no external deps.
# Budget of 30 tokens forces ~3 chunks while keeping chunk_size large enough
# for breadcrumb lines (Suite/Test header) to fit in each continuation chunk.
_CHUNKED_STRATEGY = ChunkingStrategy(
    tokenizer_config=TokenizerModel(
        tokenizer=TokenizerTypes.OLLAMA, max_content_tokens=30
    ),
    system_prompt="",
)


@pytest.fixture()
def render_xml(tmp_path):
    """Write RENDER_XML to a temp file and return its path."""
    p = tmp_path / "output.xml"
    p.write_text(RENDER_XML)
    return p


@pytest.fixture()
def tagged_xml(tmp_path):
    """Write TAGGED_XML to a temp file and return its path."""
    p = tmp_path / "tagged.xml"
    p.write_text(TAGGED_XML)
    return p


@pytest.fixture()
def suite_setup_fail_xml(tmp_path):
    """Write SUITE_SETUP_FAIL_XML to a temp file and return its path."""
    p = tmp_path / "suite_setup_fail.xml"
    p.write_text(SUITE_SETUP_FAIL_XML)
    return p


def test_render_pipeline_produces_readable_indented_text(render_xml):
    results = ContextAwareRobotResults(render_xml).include_fields(
        ["name", "status", "message"]
    )
    _, text = next(results.as_texts())

    expected = (
        "Suite: My Application\n"
        "    Suite: Data Pipeline\n"
        "        Test: Processes Data And Writes Output - FAIL\n"
        "            Keyword: Connect To Database - PASS\n"
        "                DeprecationWarning: Call to deprecated create function. (repeats ×3)\n"
        "            Keyword: Write Output - FAIL\n"
        "                Writing 1000 rows to output table."
    )
    assert text == expected


def test_chunking_injects_ancestor_breadcrumbs_into_continuation_chunk(render_xml):
    results = (
        ContextAwareRobotResults(render_xml)
        .include_fields(["name", "status", "message"])
        .set_chunking(_CHUNKED_STRATEGY)
    )

    chunks_by_test = {name: chunks for name, chunks, _, _ in results.render_chunks()}
    chunks = chunks_by_test["Processes Data And Writes Output"]

    assert len(chunks) > 1, "Increase text or reduce token budget to force multi-chunk"
    for chunk in chunks[1:]:
        assert "Suite: My Application" in chunk
        assert "Data Pipeline" in chunk


def test_get_rc_robot_results_filters_by_include_tags(tagged_xml):
    results = get_rc_robot_results(
        file_path=tagged_xml, include_tags=["smoke"], exclude_passing=False
    )
    assert results.test_names == ["Smoke Test", "Smoke WIP"]


def test_get_rc_robot_results_filters_by_exclude_tags(tagged_xml):
    results = get_rc_robot_results(
        file_path=tagged_xml, exclude_tags=["wip"], exclude_passing=False
    )
    assert results.test_names == ["Smoke Test", "Regression"]


# Important logical test for suite setup failure collapsing underlying tests into a single unit - no need to analyze individual tests
def test_get_rc_robot_results_collapses_when_suite_setup_fails(suite_setup_fail_xml):
    results = get_rc_robot_results(file_path=suite_setup_fail_xml)
    assert results.total_test_count == 2
    assert results.test_names == ["Root"]
    name, text = next(results.as_texts())
    assert name == "Root"
    assert "Setup: Setup - FAIL" in text
    assert "Test:" not in text
    assert list(results.as_texts()) == [
        (
            "Root",
            "Suite: Root\n    Setup: Setup - FAIL\n        elapsed: 0:00:00.001000",
        )
    ]
