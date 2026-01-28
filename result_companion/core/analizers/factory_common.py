import asyncio
from typing import Callable, Tuple

from langchain_anthropic import ChatAnthropic
from langchain_aws import BedrockLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama.llms import OllamaLLM
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from result_companion.core.chunking.chunking import (
    accumulate_llm_results_for_summarizaton_chain,
)
from result_companion.core.chunking.utils import (
    calculate_chunk_size,
    tokenizer_mappings,
)
from result_companion.core.parsers.config import DefaultConfigModel
from result_companion.core.utils.logging_config import get_progress_logger
from result_companion.core.utils.progress import run_tasks_with_progress

logger = get_progress_logger("Analyzer")

MODELS = Tuple[
    OllamaLLM
    | AzureChatOpenAI
    | BedrockLLM
    | ChatGoogleGenerativeAI
    | ChatOpenAI
    | ChatAnthropic,
    Callable,
]


async def accumulate_llm_results_without_streaming(
    test_case: dict,
    question_from_config_file: str,
    prompt: ChatPromptTemplate,
    model: MODELS,
) -> Tuple[str, str, list]:
    logger.info(
        f"### Test Case: {test_case['name']}, content length: {len(str(test_case))}"
    )
    chain = prompt | model | StrOutputParser()
    return (
        await chain.ainvoke(
            {"context": test_case, "question": question_from_config_file}, verbose=True
        ),
        test_case["name"],
        [],
    )


async def execute_llm_and_get_results(
    test_cases: list,
    config: DefaultConfigModel,
    prompt: ChatPromptTemplate,
    model: MODELS,
) -> dict:
    question_from_config_file = config.llm_config.question_prompt
    tokenizer = config.tokenizer
    test_case_concurrency = config.concurrency.test_case
    chunk_concurrency = config.concurrency.chunk
    chunk_analysis_prompt = config.llm_config.chunking.chunk_analysis_prompt
    final_synthesis_prompt = config.llm_config.chunking.final_synthesis_prompt

    llm_results = dict()
    corutines = []
    logger.info(
        f"Executing chain, {len(test_cases)=}, {test_case_concurrency=}, {chunk_concurrency=}"
    )

    for test_case in test_cases:
        raw_test_case_text = str(test_case)

        # First check if test fits in context with question_prompt (no chunking needed)
        chunk_check = calculate_chunk_size(
            raw_test_case_text, question_from_config_file, tokenizer
        )

        if chunk_check.chunk_size == 0:
            # Test fits in context - no chunking needed
            corutines.append(
                accumulate_llm_results_without_streaming(
                    test_case, question_from_config_file, prompt, model
                )
            )
        else:
            # Chunking needed - recalculate using chunk_analysis_prompt
            chunk = calculate_chunk_size(
                raw_test_case_text, chunk_analysis_prompt, tokenizer
            )
            tokenizer_func = tokenizer_mappings.get(tokenizer.tokenizer)
            corutines.append(
                accumulate_llm_results_for_summarizaton_chain(
                    test_case=test_case,
                    chunk_analysis_prompt=chunk_analysis_prompt,
                    final_synthesis_prompt=final_synthesis_prompt,
                    chunking_strategy=chunk,
                    llm=model,
                    chunk_concurrency=chunk_concurrency,
                    tokenizer=tokenizer_func,
                    max_content_tokens=tokenizer.max_content_tokens,
                )
            )

    semaphore = asyncio.Semaphore(test_case_concurrency)

    desc = f"Analyzing {len(test_cases)} test cases"
    results = await run_tasks_with_progress(corutines, semaphore=semaphore, desc=desc)

    for result, name, chunks in results:
        llm_results[name] = result

    return llm_results
