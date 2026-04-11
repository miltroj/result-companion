import json
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
    AnalyzeReport,
    _build_overall_summary_prompt,
    compute_source_hash,
    render_json_report,
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


class TestAnalyzeReport:
    """Tests for AnalyzeReport dataclass."""

    def test_roundtrip_json(self):
        report = AnalyzeReport(
            failed_test_count=2,
            analyzed_tests=["test_a", "test_b"],
            per_test_results={"test_a": "Error A", "test_b": "Error B"},
            overall_summary="Two failures.",
        )

        restored = AnalyzeReport.from_json(report.to_json())

        assert restored == report

    def test_to_text_matches_render_text_report(self):
        report = AnalyzeReport(
            failed_test_count=1,
            analyzed_tests=["test_x"],
            per_test_results={"test_x": "Timeout error"},
            overall_summary="Summary here",
        )

        text = report.to_text()

        assert "test_x" in text
        assert "Timeout error" in text
        assert "Summary here" in text

    def test_has_failures_true_when_tests_present(self):
        report = AnalyzeReport(failed_test_count=1, analyzed_tests=["t"])

        assert report.has_failures() is True

    def test_has_failures_false_when_empty(self):
        report = AnalyzeReport(failed_test_count=0, analyzed_tests=[])

        assert report.has_failures() is False

    def test_roundtrip_json_with_all_metadata(self):
        report = AnalyzeReport(
            failed_test_count=1,
            analyzed_tests=["test_a"],
            per_test_results={"test_a": "err"},
            overall_summary="sum",
            model="openai/gpt-4",
            source_file="output.xml",
            total_test_count=5,
            source_hash="abc123def456",
        )

        restored = AnalyzeReport.from_json(report.to_json())

        assert restored == report
        assert restored.model == "openai/gpt-4"
        assert restored.source_file == "output.xml"
        assert restored.total_test_count == 5
        assert restored.source_hash == "abc123def456"

    def test_from_json_defaults_missing_metadata_to_none(self):
        minimal_json = json.dumps(
            {
                "failed_test_count": 0,
                "analyzed_tests": [],
            }
        )

        report = AnalyzeReport.from_json(minimal_json)

        assert report.model is None
        assert report.source_file is None
        assert report.total_test_count is None
        assert report.source_hash is None


class TestComputeSourceHash:
    """Tests for compute_source_hash."""

    def test_returns_12_char_hex_string(self):
        result = compute_source_hash([{"name": "t1", "status": "FAIL"}])

        assert len(result) == 12
        assert all(c in "0123456789abcdef" for c in result)

    def test_same_input_produces_same_hash(self):
        data = [{"name": "t1", "status": "FAIL"}]

        assert compute_source_hash(data) == compute_source_hash(data)

    def test_different_input_produces_different_hash(self):
        a = [{"name": "t1", "status": "FAIL"}]
        b = [{"name": "t2", "status": "PASS"}]

        assert compute_source_hash(a) != compute_source_hash(b)


class TestRenderJsonReport:
    """Tests for render_json_report function."""

    def test_produces_valid_json_with_all_fields(self):
        result = render_json_report(
            llm_results={"test_1": "Error details"},
            analyzed_test_names=["test_1"],
            overall_summary="Root cause.",
        )
        parsed = json.loads(result)

        assert parsed["failed_test_count"] == 1
        assert parsed["analyzed_tests"] == ["test_1"]
        assert parsed["per_test_results"]["test_1"] == "Error details"
        assert parsed["overall_summary"] == "Root cause."

    def test_empty_results_produces_zero_count(self):
        result = render_json_report(
            llm_results={},
            analyzed_test_names=[],
            overall_summary=None,
        )
        parsed = json.loads(result)

        assert parsed["failed_test_count"] == 0
        assert parsed["overall_summary"] is None

    def test_includes_metadata_when_provided(self):
        result = render_json_report(
            llm_results={"t1": "err"},
            analyzed_test_names=["t1"],
            overall_summary=None,
            model="openai/gpt-4",
            source_file="output.xml",
            total_test_count=2,
            source_hash="abc123abc123",
        )
        parsed = json.loads(result)

        assert parsed["model"] == "openai/gpt-4"
        assert parsed["source_file"] == "output.xml"
        assert parsed["total_test_count"] == 2
        assert len(parsed["source_hash"]) == 12

    def test_metadata_none_when_no_test_cases_provided(self):
        result = render_json_report(
            llm_results={},
            analyzed_test_names=[],
            overall_summary=None,
        )
        parsed = json.loads(result)

        assert parsed["total_test_count"] is None
        assert parsed["source_hash"] is None


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
