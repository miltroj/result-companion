import pytest
from langchain_community.llms.fake import FakeListLLM
from langchain_core.prompts import ChatPromptTemplate

from result_companion.core.analizers.factory_common import (
    _dryrun_result,
    accumulate_llm_results_without_streaming,
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


@pytest.mark.asyncio
async def test_gather_llm_runs_and_get_results() -> None:
    config = DefaultConfigModel(
        version=1,
        llm_config=LLMConfigModel(
            question_prompt="question",
            prompt_template="template",
            chunking=ChunkingPromptsModel(
                chunk_analysis_prompt="Analyze: {text}",
                final_synthesis_prompt="Synthesize: {summary}",
            ),
        ),
        llm_factory=LLMFactoryModel(model_type="model_type"),
        tokenizer=TokenizerModel(tokenizer="ollama_tokenizer", max_content_tokens=10),
    )
    prompt = ChatPromptTemplate.from_template("template {context}")

    def mocked_model(x):
        return "llm generated result"

    # Simulate pre-filtered test cases (filtering now done in run_rc.py)
    test_cases = [
        {"name": "test2_failing", "status": "FAIL"},
    ]

    result = await execute_llm_and_get_results(
        test_cases=test_cases,
        config=config,
        prompt=prompt,
        model=mocked_model,
    )
    assert result == {"test2_failing": "llm generated result"}


@pytest.mark.asyncio
async def test_accumulate_llm_results_without_streaming():
    """Test non-streaming analysis builds chain and returns correct tuple."""
    test_case = {"name": "test_case_name", "status": "FAIL", "content": "test data"}
    question = "What failed?"
    prompt = ChatPromptTemplate.from_template("{question} {context}")

    fake_llm = FakeListLLM(responses=["Analysis result"], model="fake")

    result, name, chunks = await accumulate_llm_results_without_streaming(
        test_case, question, prompt, fake_llm
    )

    assert result == "Analysis result"
    assert name == "test_case_name"
    assert chunks == []


@pytest.mark.asyncio
async def test_dryrun_result_returns_debug_metadata():
    """Test dryrun returns test metadata without calling LLM."""
    test_case = {"name": "My Test Case", "status": "FAIL"}
    chunk = Chunking(
        chunk_size=0,
        number_of_chunks=2,
        raw_text_len=5000,
        tokens_from_raw_text=1250,
        tokenized_chunks=2,
    )

    result, name, chunks = await _dryrun_result(test_case, chunk)

    assert name == "My Test Case"
    assert "[DRYRUN]" in result
    assert "**Status**: FAIL" in result
    assert "**Chunks**: 2" in result
    assert "**Tokens**: 1250" in result
    assert chunks == []
