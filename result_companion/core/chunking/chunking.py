import asyncio
from typing import Tuple

from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable

from result_companion.core.analizers.models import MODELS
from result_companion.core.chunking.utils import Chunking
from result_companion.core.utils.logging_config import get_progress_logger

logger = get_progress_logger("Chunking")


def build_sumarization_chain(
    prompt: PromptTemplate, model: MODELS
) -> RunnableSerializable:
    return prompt | model | StrOutputParser()


def split_text_into_chunks_using_text_splitter(
    text: str, chunk_size: int, overlap: int
) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text)


async def accumulate_llm_results_for_summarizaton_chain(
    test_case: dict,
    chunk_analysis_prompt: str,
    final_synthesis_prompt: str,
    chunking_strategy: Chunking,
    llm: MODELS,
    chunk_concurrency: int = 1,
) -> Tuple[str, str, list]:
    chunks = split_text_into_chunks_using_text_splitter(
        str(test_case), chunking_strategy.chunk_size, chunking_strategy.chunk_size // 10
    )
    return await summarize_test_case(
        test_case,
        chunks,
        llm,
        chunk_analysis_prompt,
        final_synthesis_prompt,
        chunk_concurrency,
    )


async def summarize_test_case(
    test_case: dict,
    chunks: list,
    llm: MODELS,
    chunk_analysis_prompt: str,
    final_synthesis_prompt: str,
    chunk_concurrency: int = 1,
) -> Tuple[str, str, list]:
    """Summarizes large test case by analyzing chunks and synthesizing results.

    Args:
        test_case: Test case dictionary with name and data.
        chunks: List of text chunks to analyze.
        llm: Language model instance.
        chunk_analysis_prompt: Template for analyzing chunks.
        final_synthesis_prompt: Template for final synthesis.
        chunk_concurrency: Chunks to process concurrently.

    Returns:
        Tuple of (final_analysis, test_name, chunks).
    """
    logger.info(f"### For test case {test_case['name']}, {len(chunks)=}")

    summarization_prompt = PromptTemplate(
        input_variables=["text"],
        template=chunk_analysis_prompt,
    )

    summarization_chain = build_sumarization_chain(summarization_prompt, llm)
    semaphore = asyncio.Semaphore(chunk_concurrency)
    test_name = test_case["name"]
    total_chunks = len(chunks)

    async def process_with_limit(chunk: str, chunk_idx: int) -> str:
        async with semaphore:
            logger.debug(
                f"[{test_name}] Processing chunk {chunk_idx + 1}/{total_chunks}, length {len(chunk)}"
            )
            return await summarization_chain.ainvoke({"text": chunk})

    chunk_tasks = [process_with_limit(chunk, i) for i, chunk in enumerate(chunks)]
    summaries = await asyncio.gather(*chunk_tasks)

    aggregated_summary = "\n\n---\n\n".join(
        [
            f"### Chunk {i+1}/{total_chunks}\n{summary}"
            for i, summary in enumerate(summaries)
        ]
    )

    final_prompt = PromptTemplate(
        input_variables=["summary"],
        template=final_synthesis_prompt,
    )

    final_analysis_chain = build_sumarization_chain(final_prompt, llm)
    final_result = await final_analysis_chain.ainvoke({"summary": aggregated_summary})

    return final_result, test_case["id"], chunks
