from dataclasses import dataclass
from unittest.mock import patch

import pytest

from result_companion.core.analizers.factory_common import (
    _build_llm_params,
    _dryrun_result,
    _stats_header,
    analyze_test_case,
    execute_llm_and_get_results,
)
from result_companion.core.chunking.utils import Chunking
from result_companion.core.parsers.config import (
    ChunkingPromptsModel,
    DefaultConfigModel,
    LLMConfigModel,
    LLMFactoryModel,
    TokenizerModel,
)


@dataclass
class FakeLiteLLMChoice:
    """Fake LiteLLM choice for testing."""

    message: object


@dataclass
class FakeLiteLLMResponse:
    """Fake LiteLLM response for testing."""

    content: str

    @property
    def choices(self):
        """Returns fake choices list."""
        msg = type("Message", (), {"content": self.content})()
        choice = FakeLiteLLMChoice(message=msg)
        return [choice]


def make_fake_acompletion(response: str):
    """Creates a fake acompletion function that returns the given response."""

    async def fake_acompletion(**kwargs):
        return FakeLiteLLMResponse(content=response)

    return fake_acompletion


def make_config(
    model: str = "ollama_chat/test-model",
    max_content_tokens: int = 10,
) -> DefaultConfigModel:
    """Creates a test configuration."""
    return DefaultConfigModel(
        version=1,
        llm_config=LLMConfigModel(
            question_prompt="What failed?",
            prompt_template="Question: {question}\nContext: {context}",
            chunking=ChunkingPromptsModel(
                chunk_analysis_prompt="Analyze: {text}",
                final_synthesis_prompt="Synthesize: {summary}",
            ),
        ),
        llm_factory=LLMFactoryModel(
            model=model,
            api_base="http://localhost:11434",
        ),
        tokenizer=TokenizerModel(
            tokenizer="ollama_tokenizer",
            max_content_tokens=max_content_tokens,
        ),
    )


class TestBuildLLMParams:
    """Tests for _build_llm_params function."""

    def test_includes_model(self):
        """Test that model is included in params."""
        config = LLMFactoryModel(model="openai/gpt-4")

        params = _build_llm_params(config)

        assert params["model"] == "openai/gpt-4"

    def test_includes_api_base_when_set(self):
        """Test that api_base is included when provided."""
        config = LLMFactoryModel(
            model="ollama_chat/llama2",
            api_base="http://localhost:11434",
        )

        params = _build_llm_params(config)

        assert params["api_base"] == "http://localhost:11434"

    def test_includes_api_key_when_set(self):
        """Test that api_key is included when provided."""
        config = LLMFactoryModel(
            model="openai/gpt-4",
            api_key="sk-test123",
        )

        params = _build_llm_params(config)

        assert params["api_key"] == "sk-test123"

    def test_excludes_none_values(self):
        """Test that None values are not included."""
        config = LLMFactoryModel(model="openai/gpt-4")

        params = _build_llm_params(config)

        assert "api_base" not in params
        assert "api_key" not in params

    def test_includes_additional_parameters(self):
        """Test that additional parameters are merged."""
        config = LLMFactoryModel(
            model="openai/gpt-4",
            parameters={"temperature": 0.5, "max_tokens": 100},
        )

        params = _build_llm_params(config)

        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 100


class TestStatsHeader:
    """Tests for _stats_header function."""

    def test_returns_formatted_metadata(self):
        """Test stats header contains test info and stats."""
        chunk = Chunking(
            chunk_size=100,
            number_of_chunks=3,
            raw_text_len=5000,
            tokens_from_raw_text=1250,
            tokenized_chunks=3,
        )

        header = _stats_header("FAIL", chunk, dryrun=False)

        assert "FAIL" in header
        assert "Chunks: 3" in header
        assert "Tokens: ~1250" in header
        assert "[DRYRUN]" not in header

    def test_dryrun_mode(self):
        """Test stats header shows DRYRUN prefix when enabled."""
        chunk = Chunking(
            chunk_size=0,
            number_of_chunks=0,
            raw_text_len=1000,
            tokens_from_raw_text=250,
            tokenized_chunks=0,
        )

        header = _stats_header("PASS", chunk, dryrun=True)

        assert "[DRYRUN]" in header
        assert "Chunks: 0" in header

    def test_includes_test_name(self):
        """Test stats header includes test name."""
        chunk = Chunking(
            chunk_size=0,
            number_of_chunks=0,
            raw_text_len=100,
            tokens_from_raw_text=25,
            tokenized_chunks=0,
        )

        header = _stats_header("FAIL", chunk, name="My Test Case")

        assert "My Test Case" in header


class TestDryrunResult:
    """Tests for _dryrun_result function."""

    @pytest.mark.asyncio
    async def test_returns_placeholder(self):
        """Test dryrun returns placeholder message and test name."""
        test_case = {"name": "My Test", "status": "FAIL"}

        result, name, chunks = await _dryrun_result(test_case)

        assert name == "My Test"
        assert "No LLM analysis" in result
        assert chunks == []


class TestAnalyzeTestCase:
    """Tests for analyze_test_case function."""

    @pytest.mark.asyncio
    async def test_returns_llm_response(self):
        """Test that analyze_test_case returns LLM response."""
        test_case = {"name": "test_login", "status": "FAIL", "error": "timeout"}
        fake_acompletion = make_fake_acompletion("Root cause: timeout error")

        with patch(
            "result_companion.core.analizers.factory_common.acompletion",
            fake_acompletion,
        ):
            result, name, chunks = await analyze_test_case(
                test_case=test_case,
                question_prompt="What failed?",
                prompt_template="Q: {question}\nC: {context}",
                llm_params={"model": "test-model"},
            )

        assert result == "Root cause: timeout error"
        assert name == "test_login"
        assert chunks == []


class TestExecuteLLMAndGetResults:
    """Tests for execute_llm_and_get_results function."""

    @pytest.mark.asyncio
    async def test_processes_test_cases(self):
        """Test that execute_llm_and_get_results processes all test cases."""
        config = make_config(max_content_tokens=100000)  # High limit to avoid chunking
        test_cases = [
            {"name": "test1", "status": "FAIL"},
            {"name": "test2", "status": "FAIL"},
        ]
        fake_acompletion = make_fake_acompletion("Analysis result")

        with patch(
            "result_companion.core.analizers.factory_common.acompletion",
            fake_acompletion,
        ):
            results = await execute_llm_and_get_results(
                test_cases=test_cases,
                config=config,
            )

        assert "test1" in results
        assert "test2" in results
        assert "Analysis result" in results["test1"]
        assert "Analysis result" in results["test2"]

    @pytest.mark.asyncio
    async def test_dryrun_skips_llm_calls(self):
        """Test that dryrun mode skips actual LLM calls."""
        config = make_config()
        test_cases = [{"name": "test1", "status": "FAIL"}]

        results = await execute_llm_and_get_results(
            test_cases=test_cases,
            config=config,
            dryrun=True,
        )

        assert "test1" in results
        assert "No LLM analysis" in results["test1"]
        assert "DRYRUN" in results["test1"]

    @pytest.mark.asyncio
    async def test_includes_stats_header(self):
        """Test that results include stats header."""
        config = make_config(max_content_tokens=100000)
        test_cases = [{"name": "test_with_stats", "status": "FAIL"}]
        fake_acompletion = make_fake_acompletion("Analysis")

        with patch(
            "result_companion.core.analizers.factory_common.acompletion",
            fake_acompletion,
        ):
            results = await execute_llm_and_get_results(
                test_cases=test_cases,
                config=config,
            )

        result = results["test_with_stats"]
        assert "Status: FAIL" in result
        assert "Tokens:" in result
        assert "Analysis" in result
