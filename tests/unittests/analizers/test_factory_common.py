import pytest
from langchain_core.prompts import ChatPromptTemplate

from result_companion.analizers.factory_common import execute_llm_and_get_results
from result_companion.parsers.config import (
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
            question_prompt="question", prompt_template="template"
        ),
        llm_factory=LLMFactoryModel(model_type="model_type"),
        tokenizer=TokenizerModel(tokenizer="ollama_tokenizer", max_content_tokens=10),
    )
    prompt = ChatPromptTemplate.from_template("template {context}")
    mocked_model = lambda x: "llm generated result"

    test_cases = [
        {"name": "test1_passing", "status": "PASS"},
        {"name": "test2_failing", "status": "FAIL"},
    ]

    result = await execute_llm_and_get_results(
        test_cases=test_cases,
        config=config,
        prompt=prompt,
        model=mocked_model,
        concurrency=1,
        include_passing=False,
    )
    assert result == {"test2_failing": "llm generated result"}
