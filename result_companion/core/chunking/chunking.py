import asyncio
from typing import Any

from result_companion.core.analizers.llm_router import _smart_acompletion
from result_companion.core.chunking.utils import Chunking
from result_companion.core.utils.logging_config import get_progress_logger

logger = get_progress_logger("Chunking")


def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Splits text into overlapping chunks.

    Args:
        text: Text to split.
        chunk_size: Maximum size of each chunk.
        overlap: Number of characters to overlap between chunks.

    Returns:
        List of text chunks.
    """
    if chunk_size <= 0:
        return [text] if text else []

    if overlap >= chunk_size:
        overlap = chunk_size // 10

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])

        # Move start forward, accounting for overlap
        next_start = start + chunk_size - overlap
        if next_start <= start:
            next_start = start + chunk_size
        start = next_start

    return chunks


async def analyze_chunk(
    chunk: str,
    chunk_idx: int,
    total_chunks: int,
    test_name: str,
    chunk_analysis_prompt: str,
    llm_params: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> str:
    """Analyzes a single chunk using LiteLLM.

    Args:
        chunk: Text chunk to analyze.
        chunk_idx: Index of this chunk.
        total_chunks: Total number of chunks.
        test_name: Name of the test case.
        chunk_analysis_prompt: Prompt template with {text} placeholder.
        llm_params: Parameters for LiteLLM acompletion.
        semaphore: Semaphore for concurrency control.

    Returns:
        Analysis result for the chunk.
    """
    async with semaphore:
        logger.debug(
            f"[{test_name}] Processing chunk {chunk_idx + 1}/{total_chunks}, "
            f"length {len(chunk)}"
        )

        # Format the prompt with the chunk text
        formatted_prompt = chunk_analysis_prompt.format(text=chunk)
        messages = [{"role": "user", "content": formatted_prompt}]

        response = await _smart_acompletion(messages=messages, **llm_params)
        return response.choices[0].message.content


async def synthesize_summaries(
    aggregated_summary: str,
    final_synthesis_prompt: str,
    llm_params: dict[str, Any],
) -> str:
    """Synthesizes chunk summaries into final analysis.

    Args:
        aggregated_summary: Combined summaries from all chunks.
        final_synthesis_prompt: Prompt template with {summary} placeholder.
        llm_params: Parameters for LiteLLM acompletion.

    Returns:
        Final synthesized analysis.
    """
    formatted_prompt = final_synthesis_prompt.format(summary=aggregated_summary)
    messages = [{"role": "user", "content": formatted_prompt}]

    response = await _smart_acompletion(messages=messages, **llm_params)
    return response.choices[0].message.content


async def accumulate_llm_results_for_summarization(
    test_case: dict,
    chunk_analysis_prompt: str,
    final_synthesis_prompt: str,
    chunking_strategy: Chunking,
    llm_params: dict[str, Any],
    chunk_concurrency: int = 1,
) -> tuple[str, str, list]:
    """Summarizes large test case by analyzing chunks and synthesizing results.

    Args:
        test_case: Test case dictionary with name and data.
        chunk_analysis_prompt: Template for analyzing chunks (with {text}).
        final_synthesis_prompt: Template for final synthesis (with {summary}).
        chunking_strategy: Chunking configuration.
        llm_params: Parameters for LiteLLM acompletion.
        chunk_concurrency: Chunks to process concurrently.

    Returns:
        Tuple of (final_analysis, test_name, chunks).
    """
    overlap = chunking_strategy.chunk_size // 10
    chunks = split_text_into_chunks(
        str(test_case), chunking_strategy.chunk_size, overlap
    )

    test_name = test_case["name"]
    total_chunks = len(chunks)
    logger.info(f"### For test case {test_name}, {len(chunks)=}")

    semaphore = asyncio.Semaphore(chunk_concurrency)

    # Analyze all chunks concurrently (within semaphore limits)
    chunk_tasks = [
        analyze_chunk(
            chunk=chunk,
            chunk_idx=i,
            total_chunks=total_chunks,
            test_name=test_name,
            chunk_analysis_prompt=chunk_analysis_prompt,
            llm_params=llm_params,
            semaphore=semaphore,
        )
        for i, chunk in enumerate(chunks)
    ]
    summaries = await asyncio.gather(*chunk_tasks)

    # Aggregate summaries
    aggregated_summary = "\n\n---\n\n".join(
        [
            f"### Chunk {i+1}/{total_chunks}\n{summary}"
            for i, summary in enumerate(summaries)
        ]
    )

    # Synthesize final result
    final_result = await synthesize_summaries(
        aggregated_summary=aggregated_summary,
        final_synthesis_prompt=final_synthesis_prompt,
        llm_params=llm_params,
    )

    return final_result, test_name, chunks
