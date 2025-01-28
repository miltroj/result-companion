import asyncio
from typing import Callable, Tuple

from langchain_aws import BedrockLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_ollama.llms import OllamaLLM
from langchain_openai import AzureChatOpenAI

from result_companion.analizers.common import run_with_semaphore
from result_companion.chunking.chunking import (
    accumulate_llm_results_for_summarizaton_chain,
)
from result_companion.chunking.utils import calculate_chunk_size
from result_companion.parsers.config import LLMFactoryModel

MODELS = Tuple[OllamaLLM | AzureChatOpenAI | BedrockLLM, Callable]


async def accumulate_llm_results_without_streaming(
    test_case: list, question_from_config_file: str, chain: RunnableSerializable
) -> Tuple[str, str, list]:
    print(
        f"\n### Test Case: {test_case['name']}, content length: {len(str(test_case))}"
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
    config: LLMFactoryModel,
    prompt: ChatPromptTemplate,
    model: MODELS,
    concurrency: int = 1,
) -> dict:
    question_from_config_file = config.llm_config.question_prompt
    tokenizer = config.tokenizer

    llm_results = dict()
    corutines = []

    for test_case in test_cases:

        raw_test_case_text = str(test_case)
        chunk = calculate_chunk_size(
            raw_test_case_text, question_from_config_file, tokenizer
        )

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

    tasks = [run_with_semaphore(semaphore, coroutine) for coroutine in corutines]

    for result, name, chunks in await asyncio.gather(*tasks):
        llm_results[name] = result

    return llm_results
