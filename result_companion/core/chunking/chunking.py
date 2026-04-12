import asyncio
from itertools import groupby
from typing import Any

from result_companion.core.analizers.llm_router import _smart_acompletion
from result_companion.core.utils.logging_config import get_progress_logger

logger = get_progress_logger("Chunking")

_INDENT = "    "


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


def _indent(depth: int, text: str) -> str:
    """Returns text prefixed with depth levels of indentation."""
    return f"{_INDENT * depth}{text}"


def _render_rf_keywords(body: list[dict], depth: int) -> list[tuple[int, str]]:
    """Recursively renders Robot Framework keyword/message body items as (depth, text) pairs."""
    lines: list[tuple[int, str]] = []
    for item in body:
        if item.get("type") == "MESSAGE":
            lines.append((depth, item["message"]))
            continue
        name = item.get("name", "?")
        status = item.get("status", "")
        kind = item.get("type", "KEYWORD").title()
        lines.append((depth, f"{kind}: {name} - {status}"))
        args = item.get("args", [])
        if args:
            lines.append((depth + 1, f"args: {', '.join(str(a) for a in args)}"))
        lines.extend(_render_rf_keywords(item.get("body", []), depth + 1))
    return lines


def deduplicate_consecutive_lines(
    lines: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    """Collapses consecutive identical lines into a single annotated line.

    Args:
        lines: List of (depth, text) pairs.

    Returns:
        Deduplicated list where runs of identical lines become ``text (×N)``.
    """
    result = []
    for (depth, text), group in groupby(lines):
        count = sum(1 for _ in group)
        result.append((depth, text if count == 1 else f"{text} (repeats ×{count})"))
    return result


def render_lines_to_text(lines: list[tuple[int, str]]) -> str:
    """Joins (depth, text) pairs into an indented string."""
    return "\n".join(_indent(d, t) for d, t in lines)


def render_rf_test_structure(test_case: dict) -> list[tuple[int, str]]:
    """Renders RF suite ancestry → test case → keywords as (depth, text) lines."""
    suite_context = test_case.get("suite_context", [])
    suite_lines = [(i, f"Suite: {ctx['name']}") for i, ctx in enumerate(suite_context)]
    depth = len(suite_context)
    test_line = (depth, f"Test: {test_case['name']} - {test_case.get('status', '')}")
    return (
        suite_lines
        + [test_line]
        + _render_rf_keywords(test_case.get("body", []), depth + 1)
    )


def _collect_ancestor_context_at(
    lines: list[tuple[int, str]], at_idx: int
) -> list[str]:
    """Collects the suite→test→keyword ancestor chain for the line at at_idx.

    Walks backwards through rendered lines, picking exactly one line per depth
    level above the target line, building the full nesting context (suite name,
    test name, parent keyword) needed to make each chunk self-contained.

    Correctness relies on depth-first rendering: in such a list a parent at depth D
    always precedes all its children at depth D+1, so the first backward hit at each
    depth level is always the direct ancestor, never a sibling's subtree.
    """
    # Start one level above the target line and walk up, collecting exactly
    # one representative line per depth level until we reach the root (depth 0).
    target = lines[at_idx][0] - 1
    ancestors: list[tuple[int, str]] = []
    for i in range(at_idx - 1, -1, -1):
        if lines[i][0] == target:
            ancestors.insert(0, lines[i])
            target -= 1
            if target < 0:
                break
    return [_indent(depth, text) for depth, text in ancestors]


def _split_long_line(
    text: str, depth: int, breadcrumbs: list[str], chunk_size: int
) -> list[str]:
    """Splits a single line exceeding chunk_size into breadcrumb-prefixed chunks.

    Args:
        text: Line content (without indentation).
        depth: Indentation depth of the line.
        breadcrumbs: Pre-formatted ancestor lines prepended to each piece.
        chunk_size: Maximum characters per chunk.

    Returns:
        List of chunks, each starting with breadcrumbs context.
    """
    # Fixed chars per chunk: all breadcrumb lines + marker line + piece line indentation.
    # chunk_size // 3 guards against breadcrumbs consuming almost the full budget,
    # which would cause extremely small pieces and a near-infinite loop.
    fixed_chars = (
        sum(len(b) + 1 for b in breadcrumbs)
        + len(_indent(depth, "{...}"))
        + 1
        + len(_INDENT) * depth
    )
    available_chars = max(chunk_size - fixed_chars, chunk_size // 3)
    pieces = [
        text[i : i + available_chars] for i in range(0, len(text), available_chars)
    ]
    return [
        "\n".join(breadcrumbs + [_indent(depth, "{...}"), _indent(depth, p)])
        for p in pieces
    ]


def chunk_rf_test_lines(lines: list[tuple[int, str]], chunk_size: int) -> list[str]:
    """Splits RF test structure lines into context-aware chunks.

    Each chunk starts with the suite→test→keyword ancestor context so the LLM
    can interpret the chunk without seeing previous chunks.

    Args:
        lines: List of (depth, text) pairs from _render_rf_test_structure.
        chunk_size: Maximum characters per chunk.

    Returns:
        List of text chunks, each self-contained with ancestor context.
    """
    if not lines:
        return []

    if chunk_size <= 0:
        return [render_lines_to_text(lines)]

    total = sum(len(_indent(d, t)) + 1 for d, t in lines)
    if total <= chunk_size:
        return ["\n".join(_indent(d, t) for d, t in lines)]

    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for idx, (depth, text) in enumerate(lines):
        line = _indent(depth, text)
        line_len = len(line) + 1
        breadcrumbs = _collect_ancestor_context_at(lines, idx)

        # Edge case: a single keyword log line is longer than the whole chunk budget.
        # Fill the current chunk with as much of this line as fits before flushing,
        # so the current chunk reaches chunk_size instead of being emitted half-empty.
        # Then split the remainder into breadcrumb-prefixed sub-chunks.
        if line_len > chunk_size:
            if current:
                chunks.append("\n".join(current))
                current, current_size = [], 0
            pieces = _split_long_line(text, depth, breadcrumbs, chunk_size)
            chunks.extend(pieces[:-1])
            current = pieces[-1].splitlines() if pieces else []
            current_size = sum(len(s) + 1 for s in current)
            continue

        # Normal case: this line would overflow the current chunk.
        # Flush it and start the next chunk with ancestor context + "{...}" marker
        # so the LLM knows it is reading a continuation, not the start of the test.
        if current_size + line_len > chunk_size and current:
            chunks.append("\n".join(current))
            current = breadcrumbs + [_indent(depth, "{...}")]
            current_size = sum(len(piece) + 1 for piece in current)

        current.append(line)
        current_size += line_len

    if current:
        chunks.append("\n".join(current))

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
    test_name: str,
    chunks: list[str],
    chunk_analysis_prompt: str,
    final_synthesis_prompt: str,
    llm_params: dict[str, Any],
    chunk_concurrency: int = 1,
) -> tuple[str, str, list]:
    """Summarizes large test case by analyzing chunks and synthesizing results.

    Args:
        test_name: Name of the test case.
        chunks: Pre-computed text chunks.
        chunk_analysis_prompt: Template for analyzing chunks (with {text}).
        final_synthesis_prompt: Template for final synthesis (with {summary}).
        llm_params: Parameters for LiteLLM acompletion.
        chunk_concurrency: Chunks to process concurrently.

    Returns:
        Tuple of (final_analysis, test_name, chunks).
    """
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
