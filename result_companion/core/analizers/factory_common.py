import asyncio
from typing import Callable, Tuple

from langchain_aws import BedrockLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama.llms import OllamaLLM
from langchain_openai import AzureChatOpenAI

from result_companion.core.chunking.chunking import (
    accumulate_llm_results_for_summarizaton_chain,
)
from result_companion.core.chunking.utils import calculate_chunk_size
from result_companion.core.parsers.config import DefaultConfigModel

# Import and initialize progress logger
from result_companion.core.utils.progress import (
    get_progress_logger,
    run_tasks_with_progress,
)

# Initialize the singleton progress logger at module level
_progress_logger = get_progress_logger()

MODELS = Tuple[
    OllamaLLM | AzureChatOpenAI | BedrockLLM | ChatGoogleGenerativeAI, Callable
]


async def accumulate_llm_results_without_streaming(
    test_case: list, question_from_config_file: str, chain: RunnableSerializable
) -> Tuple[str, str, list]:
    # Use the module-level logger
    _progress_logger.info(
        f"### Test Case: {test_case['name']}, content length: {len(str(test_case))}"
    )
    return (
        await chain.ainvoke(
            {"context": test_case, "question": question_from_config_file}, verbose=True
        ),
        test_case["name"],
        [],
    )


def default_chain(prompt: ChatPromptTemplate, model: MODELS) -> RunnableSerializable:
    return prompt | model | StrOutputParser()


def compose_chain(prompt: ChatPromptTemplate, model: MODELS) -> RunnableSerializable:
    # TODO: create a propper chain
    return default_chain(prompt, model)


async def execute_llm_and_get_results(
    test_cases: list,
    config: DefaultConfigModel,
    prompt: ChatPromptTemplate,
    model: MODELS,
    concurrency: int = 1,
    include_passing: bool = True,
) -> dict:
    question_from_config_file = config.llm_config.question_prompt
    tokenizer = config.tokenizer

    llm_results = dict()
    corutines = []
    relevant_cases = []
    _progress_logger.info(f"Executing chain, {len(test_cases)=}, {concurrency=}")

    for test_case in test_cases:
        if test_case.get("status") == "PASS" and not include_passing:
            _progress_logger.debug(f"Skipping, passing tests {test_case['name']!r}!")
            continue

        relevant_cases.append(test_case)
        raw_test_case_text = str(test_case)
        chunk = calculate_chunk_size(
            raw_test_case_text, question_from_config_file, tokenizer
        )

        # TODO: zero chunk size seems magical
        if chunk.chunk_size == 0:
            chain = default_chain(prompt, model)
            corutines.append(
                accumulate_llm_results_without_streaming(
                    test_case, question_from_config_file, chain
                )
            )
        else:
            chain = compose_chain(prompt, model)
            corutines.append(
                accumulate_llm_results_for_summarizaton_chain(
                    test_case=test_case,
                    question_from_config_file=question_from_config_file,
                    chain=chain,
                    chunking_strategy=chunk,
                    llm=model,
                )
            )

    semaphore = asyncio.Semaphore(concurrency)  # Limit concurrency

    desc = f"Analyzing {len(relevant_cases)} test cases"
    results = await run_tasks_with_progress(corutines, semaphore, desc)

    for result, name, chunks in results:
        llm_results[name] = result

    return llm_results
