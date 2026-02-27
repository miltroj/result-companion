from dataclasses import dataclass

import pytest

from result_companion.core.parsers.config import (
    ChunkingPromptsModel,
    DefaultConfigModel,
    LLMConfigModel,
    LLMFactoryModel,
    TokenizerModel,
)
from result_companion.core.results.text_report import (
    _build_overall_summary_prompt,
    summarize_failures_with_llm,
)


@dataclass
class FakeLiteLLMResponse:
    """Fake LiteLLM response for testing."""

    content: str

    @property
    def choices(self):
        """Returns fake choices list."""
        msg = type("Message", (), {"content": self.content})()
        return [type("Choice", (), {"message": msg})()]


def make_fake_acompletion(response: str):
    """Creates a fake acompletion function returning given response."""

    async def fake_acompletion(**kwargs):
        return FakeLiteLLMResponse(content=response)

    return fake_acompletion


def make_config(
    summary_prompt_template: str = "Summary:\n{analyses}",
) -> DefaultConfigModel:
    """Creates minimal test configuration."""
    return DefaultConfigModel(
        version=1,
        llm_config=LLMConfigModel(
            question_prompt="What failed?",
            prompt_template="Q: {question}\nC: {context}",
            summary_prompt_template=summary_prompt_template,
            chunking=ChunkingPromptsModel(
                chunk_analysis_prompt="Analyze: {text}",
                final_synthesis_prompt="Synthesize: {summary}",
            ),
        ),
        llm_factory=LLMFactoryModel(model="openai/gpt-4"),
        tokenizer=TokenizerModel(tokenizer="ollama_tokenizer", max_content_tokens=100),
    )


@pytest.fixture
def patch_smart_acompletion(monkeypatch):
    """Patches text_report._smart_acompletion with provided async callable."""

    def _apply(fake_acompletion):
        monkeypatch.setattr(
            "result_companion.core.results.text_report._smart_acompletion",
            fake_acompletion,
        )

    return _apply


class TestBuildOverallSummaryPrompt:
    """Tests for _build_overall_summary_prompt."""

    def test_single_result_formats_into_template(self):
        result = _build_overall_summary_prompt(
            {"test_login": "Timeout error"},
            "Summary:\n{analyses}",
        )

        assert "### test_login" in result
        assert "Timeout error" in result
        assert result.startswith("Summary:\n")

    def test_multiple_results_includes_all(self):
        results = {"test_a": "Error A", "test_b": "Error B"}

        prompt = _build_overall_summary_prompt(results, "{analyses}")

        assert "### test_a" in prompt
        assert "Error A" in prompt
        assert "### test_b" in prompt
        assert "Error B" in prompt

    def test_strips_whitespace_from_results(self):
        prompt = _build_overall_summary_prompt(
            {"test_x": "  padded result  \n"},
            "{analyses}",
        )

        assert "padded result" in prompt
        assert "  padded result  " not in prompt


class TestSummarizeFailuresWithLLM:
    """Tests for summarize_failures_with_llm."""

    @pytest.mark.asyncio
    async def test_empty_results_returns_none(self):
        result = await summarize_failures_with_llm({}, config=None)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_llm_response(self, patch_smart_acompletion):
        patch_smart_acompletion(make_fake_acompletion("Overall: 2 failures"))

        result = await summarize_failures_with_llm(
            {"test_a": "Error A", "test_b": "Error B"},
            make_config(),
        )

        assert result == "Overall: 2 failures"

    @pytest.mark.asyncio
    async def test_exception_returns_none(self, patch_smart_acompletion):
        async def failing_acompletion(**kwargs):
            raise RuntimeError("LLM unavailable")

        patch_smart_acompletion(failing_acompletion)

        result = await summarize_failures_with_llm(
            {"test_a": "Error"},
            make_config(),
        )

        assert result is None
