import asyncio
from typing import Callable, Tuple

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
    condense_prompt: str,
    chunking_strategy: Chunking,
    llm: MODELS,
    chunk_concurrency: int = 1,
    tokenizer: Callable[[str], int] | None = None,
    max_content_tokens: int = 50000,
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
        condense_prompt,
        chunk_concurrency,
        tokenizer,
        max_content_tokens,
    )


async def condense_summaries_if_needed(
    summaries: list[str],
    final_synthesis_prompt: str,
    condense_prompt: str,
    tokenizer: Callable[[str], int],
    max_tokens: int,
    llm: MODELS,
    depth: int = 0,
) -> str:
    """Recursively condenses summaries if they exceed token limit.

    Args:
        summaries: List of chunk summaries.
        final_synthesis_prompt: Template for final synthesis.
        condense_prompt: Template for condensing grouped summaries.
        tokenizer: Function to count tokens.
        max_tokens: Maximum allowed tokens.
        llm: Language model for condensing.
        depth: Current recursion depth (max 3).

    Returns:
        Aggregated summary that fits within limit.
    """
    logger.debug(f"### Condensing summaries: {len(summaries)=}")
    aggregated = "\n\n---\n\n".join(
        [f"### Chunk {i+1}/{len(summaries)}\n{s}" for i, s in enumerate(summaries)]
    )

    prompt_tokens = tokenizer(final_synthesis_prompt)
    available_tokens = max_tokens - prompt_tokens - 500

    # Guard: return early if limits don't make sense or max depth reached
    if available_tokens <= 0 or depth >= 3 or len(summaries) <= 1:
        logger.debug(
            f"### Returning early: {available_tokens=}, {depth=}, {len(summaries)=}"
        )
        return aggregated

    summary_tokens = tokenizer(aggregated)
    if summary_tokens <= available_tokens:
        logger.debug(
            f"### Returning early: {summary_tokens=} <= {available_tokens=}, {depth=}, {len(summaries)=}"
        )
        return aggregated

    logger.warning(
        f"Summaries exceed limit ({summary_tokens} > {available_tokens}). "
        f"Condensing {len(summaries)} summaries (depth={depth})..."
    )

    avg_summary_tokens = max(1, summary_tokens // len(summaries))
    summaries_per_group = max(2, available_tokens // avg_summary_tokens)

    # Guard: if grouping won't reduce count, return as-is
    if summaries_per_group >= len(summaries):
        logger.debug(
            f"### Returning early: {summaries_per_group=} >= {len(summaries)=}, {depth=}, {len(summaries)=}"
        )
        return aggregated

    groups = [
        summaries[i : i + summaries_per_group]
        for i in range(0, len(summaries), summaries_per_group)
    ]

    logger.info(f"Condensing into {len(groups)} groups (depth={depth})")
    logger.debug(f"### Groups: {groups}")

    prompt_template = PromptTemplate(input_variables=["text"], template=condense_prompt)
    condense_chain = build_sumarization_chain(prompt_template, llm)

    condensed = []
    for i, group in enumerate(groups):
        group_text = "\n\n".join(group)
        result = await condense_chain.ainvoke({"text": group_text})
        condensed.append(f"[Group {i+1}] {result}")

    return await condense_summaries_if_needed(
        condensed,
        final_synthesis_prompt,
        condense_prompt,
        tokenizer,
        max_tokens,
        llm,
        depth + 1,
    )


async def summarize_test_case(
    test_case: dict,
    chunks: list,
    llm: MODELS,
    chunk_analysis_prompt: str,
    final_synthesis_prompt: str,
    condense_prompt: str,
    chunk_concurrency: int = 1,
    tokenizer: Callable[[str], int] | None = None,
    max_content_tokens: int = 50000,
) -> Tuple[str, str, list]:
    """Summarizes large test case by analyzing chunks and synthesizing results.

    Args:
        test_case: Test case dictionary with name and data.
        chunks: List of text chunks to analyze.
        llm: Language model instance.
        chunk_analysis_prompt: Template for analyzing chunks.
        final_synthesis_prompt: Template for final synthesis.
        condense_prompt: Template for condensing summaries when exceeding limit.
        chunk_concurrency: Chunks to process concurrently.
        tokenizer: Function to count tokens (for limit checking).
        max_content_tokens: Maximum tokens for final synthesis.

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

    # Condense summaries if they exceed token limit (recursive, preserves info)
    if tokenizer is not None:
        aggregated_summary = await condense_summaries_if_needed(
            summaries,
            final_synthesis_prompt,
            condense_prompt,
            tokenizer,
            max_content_tokens,
            llm,
        )
    else:
        aggregated_summary = "\n\n---\n\n".join(
            [f"### Chunk {i+1}/{total_chunks}\n{s}" for i, s in enumerate(summaries)]
        )
    logger.debug(f"### Aggregated summary: {aggregated_summary}")
    final_prompt = PromptTemplate(
        input_variables=["summary"],
        template=final_synthesis_prompt,
    )

    final_analysis_chain = build_sumarization_chain(final_prompt, llm)
    final_result = await final_analysis_chain.ainvoke({"summary": aggregated_summary})

    return final_result, test_case["name"], chunks
