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
from result_companion.core.chunking.utils import Chunking, calculate_chunk_size
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


def _stats_header(
    status: str, chunk: Chunking, dryrun: bool = False, name: str = ""
) -> str:
    """Returns markdown stats line for analysis."""
    chunks = chunk.number_of_chunks if chunk.requires_chunking else 0
    prefix = "**[DRYRUN]** " if dryrun else ""
    return f"""## {prefix} {name}

#### Status: {status} · Chunks: {chunks} · Tokens: ~{chunk.tokens_from_raw_text} · Raw length: {chunk.raw_text_len}

---

"""


async def _dryrun_result(test_case: dict) -> Tuple[str, str, list]:
    """Returns placeholder without calling LLM."""
    logger.info(
        f"### Test Case: {test_case['name']}, content length: {len(str(test_case))}"
    )
    return ("*No LLM analysis in dryrun mode.*", test_case["name"], [])


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
    dryrun: bool = False,
) -> dict:
    question_from_config_file = config.llm_config.question_prompt
    tokenizer = config.tokenizer
    test_case_concurrency = config.concurrency.test_case
    chunk_concurrency = config.concurrency.chunk
    chunk_analysis_prompt = config.llm_config.chunking.chunk_analysis_prompt
    final_synthesis_prompt = config.llm_config.chunking.final_synthesis_prompt

    llm_results = dict()
    corutines = []
    test_case_stats = {}  # name -> (chunk, status) for adding headers later
    logger.info(
        f"Executing chain, {len(test_cases)=}, {test_case_concurrency=}, {chunk_concurrency=}"
    )

    for test_case in test_cases:
        raw_test_case_text = str(test_case)
        chunk = calculate_chunk_size(
            raw_test_case_text, question_from_config_file, tokenizer
        )
        test_case_stats[test_case["name"]] = (chunk, test_case.get("status", "N/A"))

        if dryrun:
            corutines.append(_dryrun_result(test_case))
        elif not chunk.requires_chunking:
            corutines.append(
                accumulate_llm_results_without_streaming(
                    test_case, question_from_config_file, prompt, model
                )
            )
        else:
            corutines.append(
                accumulate_llm_results_for_summarizaton_chain(
                    test_case=test_case,
                    chunk_analysis_prompt=chunk_analysis_prompt,
                    final_synthesis_prompt=final_synthesis_prompt,
                    chunking_strategy=chunk,
                    llm=model,
                    chunk_concurrency=chunk_concurrency,
                )
            )

    semaphore = asyncio.Semaphore(test_case_concurrency)

    desc = f"Analyzing {len(test_cases)} test cases"
    results = await run_tasks_with_progress(corutines, semaphore=semaphore, desc=desc)

    for result, name, chunks in results:
        chunk, status = test_case_stats[name]
        header = _stats_header(status, chunk, dryrun, name)
        llm_results[name] = header + result
        logger.debug(f"Stored result for key: {name}")

    return llm_results
