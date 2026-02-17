from types import SimpleNamespace
from unittest.mock import patch

import pytest

from result_companion.core.parsers.config import DefaultConfigModel
from result_companion.core.results.text_report import (
    render_text_report,
    summarize_failures_with_llm,
)


def test_render_text_report_includes_summary_and_failed_tests():
    """Builds concise report with summary and per-test sections."""
    report = render_text_report(
        llm_results={"test_a": "analysis A", "test_b": "analysis B"},
        failed_test_names=["test_a", "test_b"],
        overall_summary="High-level failure summary.",
    )

    assert "Result Companion - Failure Summary" in report
    assert "Failed tests analyzed: 2" in report
    assert "High-level failure summary." in report
    assert "- test_a" in report
    assert "Per-test Analysis:" in report
    assert "analysis B" in report


@pytest.mark.asyncio
async def test_summarize_failures_uses_prompt_template_from_config():
    """Uses configurable failure summary prompt template."""
    config = DefaultConfigModel(
        version=1.0,
        llm_config={
            "question_prompt": "question prompt",
            "prompt_template": "my_template {question} {context}",
            "failure_summary_prompt_template": "CI summary only:\n{analyses}",
            "chunking": {
                "chunk_analysis_prompt": "Analyze: {text}",
                "final_synthesis_prompt": "Synthesize: {summary}",
            },
        },
        llm_factory={
            "model": "openai/gpt-4",
            "api_key": "sk-test",
        },
        tokenizer={
            "tokenizer": "openai_tokenizer",
            "max_content_tokens": 1000,
        },
    )

    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="global summary"))]
    )
    llm_results = {"test_a": "details A"}

    with (
        patch(
            "result_companion.core.results.text_report.load_config",
            return_value=config,
        ),
        patch(
            "result_companion.core.results.text_report._smart_acompletion",
            return_value=fake_response,
        ) as mocked_completion,
    ):
        summary = await summarize_failures_with_llm(
            llm_results=llm_results,
            model_name="openai/gpt-4",
            config=None,
        )

    assert summary == "global summary"
    messages = mocked_completion.call_args.kwargs["messages"]
    assert messages[0]["content"].startswith("CI summary only:")
    assert "### test_a" in messages[0]["content"]
