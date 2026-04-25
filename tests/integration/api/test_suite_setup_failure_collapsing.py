"""Integration test: suite-setup failure collapses children into single analysis unit."""

from pathlib import Path

from result_companion.api import analyze
from result_companion.core.parsers.config import (
    ChunkingPromptsModel,
    DefaultConfigModel,
    LLMConfigModel,
    LLMFactoryModel,
    TokenizerModel,
)

FIXTURE = Path(__file__).parent / "fixtures" / "output_nested_setups.xml"

EXPECTED_ANALYZED = {
    "Failing Setup",
    "Test Runs But Suite Teardown Will Fail",
    "Another Test That Runs Despite Teardown Failure",
    "Deep Test That Fails",
    "Test That Fails After Suite Setup",
}


def _make_config() -> DefaultConfigModel:
    return DefaultConfigModel(
        version=1,
        llm_config=LLMConfigModel(
            question_prompt="What failed?",
            prompt_template="Q: {question}\nC: {context}",
            summary_prompt_template="Summary:\n{analyses}",
            chunking=ChunkingPromptsModel(
                chunk_analysis_prompt="Analyze: {text}",
                final_synthesis_prompt="Synthesize: {summary}",
            ),
        ),
        llm_factory=LLMFactoryModel(model="openai/gpt-4", api_key="sk-test"),
        tokenizer=TokenizerModel(tokenizer="openai_tokenizer", max_content_tokens=1000),
    )


def test_analyze_dryrun_skips_suite_setup_failure_children():
    """Suite-setup failure collapses skipped children; only real failures are analyzed."""
    result = analyze(output=FIXTURE, config=_make_config(), dryrun=True)

    assert set(result.test_names) == EXPECTED_ANALYZED
    assert len(result.test_names) == 5
