from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
PLUGIN_ROOT = ROOT / "examples" / "plugins" / "pytest_junit"
SAMPLE_JUNIT = PLUGIN_ROOT / "sample_junit.xml"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from result_companion_pytest_junit.plugin import PytestJUnitPlugin  # noqa: E402

from result_companion.api import analyze  # noqa: E402
from result_companion.core.chunking.chunking import ChunkingStrategy  # noqa: E402
from result_companion.core.parsers.config import DefaultConfigModel  # noqa: E402
from result_companion.core.plugins.base import ParseOptions  # noqa: E402


def make_config() -> DefaultConfigModel:
    """Creates minimal dry-run analysis config."""
    return DefaultConfigModel(
        version=1.0,
        llm_config={
            "question_prompt": "Find failure cause.",
            "prompt_template": "{question}\n{context}",
            "chunking": {
                "chunk_analysis_prompt": "Analyze: {text}",
                "final_synthesis_prompt": "Synthesize: {summary}",
            },
            "summary_prompt_template": "Summary:\n{analyses}",
        },
        llm_factory={
            "model": "openai/gpt-4",
            "api_key": "sk-test",
        },
        tokenizer={
            "tokenizer": "openai_tokenizer",
            "max_content_tokens": 1000,
        },
        test_filter={
            "include_tags": [],
            "exclude_tags": [],
            "include_passing": False,
        },
    )


def test_can_parse_recognizes_pytest_junit_xml():
    plugin = PytestJUnitPlugin()

    result = plugin.can_parse(SAMPLE_JUNIT)

    assert result is True


def test_parse_filters_passing_and_renders_failure_chunk():
    plugin = PytestJUnitPlugin()
    options = ParseOptions(exclude_passing=True)

    results = plugin.parse(SAMPLE_JUNIT, options)
    results.set_chunking(ChunkingStrategy.build(max_content_tokens=1000))
    payloads = list(results.render_chunks())

    assert results.total_test_count == 3
    assert results.test_names == ["tests.test_math.test_divides_by_zero"]
    assert len(payloads) == 1
    assert payloads[0].test_name == "tests.test_math.test_divides_by_zero"
    assert payloads[0].status == "FAIL"
    assert "ZeroDivisionError: division by zero" in payloads[0].chunks[0]


def test_parse_rejects_tag_filters():
    plugin = PytestJUnitPlugin()
    options = ParseOptions(include_tags=["smoke"])

    with pytest.raises(ValueError, match="does not support tag filters"):
        plugin.parse(SAMPLE_JUNIT, options)


def test_analyze_dryrun_with_pytest_junit_plugin_returns_failure_report():
    plugin = PytestJUnitPlugin()
    config = make_config()

    result = analyze(
        output=SAMPLE_JUNIT,
        config=config,
        dryrun=True,
        result_format="pytest-junit",
        parser_plugins=(plugin,),
    )

    assert result.test_names == ["tests.test_math.test_divides_by_zero"]
    report = result.llm_results["tests.test_math.test_divides_by_zero"]
    assert "## **[DRYRUN]**  tests.test_math.test_divides_by_zero" in report
    assert "*No LLM analysis in dryrun mode.*" in report
