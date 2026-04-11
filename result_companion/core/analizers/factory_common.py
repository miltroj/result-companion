import asyncio
from typing import Any

from result_companion.core.analizers.llm_router import _smart_acompletion
from result_companion.core.chunking.chunking import (
    accumulate_llm_results_for_summarization,
    chunk_rf_test_lines,
)
from result_companion.core.chunking.rf_chunker import ChunkableResult
from result_companion.core.chunking.utils import Chunking, calculate_chunk_size
from result_companion.core.parsers.config import DefaultConfigModel, LLMFactoryModel
from result_companion.core.utils.logging_config import get_progress_logger
from result_companion.core.utils.progress import run_tasks_with_progress

logger = get_progress_logger("Analyzer")


def _stats_header(
    status: str, chunk: Chunking, dryrun: bool = False, name: str = ""
) -> str:
    """Returns markdown stats line for analysis.

    Args:
        status: Test case status (PASS/FAIL).
        chunk: Chunking information.
        dryrun: Whether this is a dryrun.
        name: Test case name.

    Returns:
        Formatted markdown header string.
    """
    chunks = chunk.number_of_chunks if chunk.requires_chunking else 0
    prefix = "**[DRYRUN]** " if dryrun else ""
    return f"""## {prefix} {name}

#### Status: {status} · Chunks: {chunks} · Tokens: ~{chunk.tokens_from_raw_text} · Raw length: {chunk.raw_text_len}

---

"""


def _build_llm_params(llm_factory: LLMFactoryModel) -> dict[str, Any]:
    """Builds LiteLLM parameters from config.

    Args:
        llm_factory: LLM factory configuration.

    Returns:
        Dictionary of parameters for acompletion().
    """
    params = {"model": llm_factory.model}

    if llm_factory.api_base:
        params["api_base"] = llm_factory.api_base

    if llm_factory.api_key:
        params["api_key"] = llm_factory.api_key

    params.update(llm_factory.parameters)

    return params


async def _dryrun_result(test_name: str, rendered_text: str) -> tuple[str, str, list]:
    """Returns placeholder without calling LLM.

    Args:
        test_name: Test case name.
        rendered_text: Rendered test case text.

    Returns:
        Tuple of (result, test_name, chunks).
    """
    logger.info(f"### Test Case: {test_name}, content length: {len(rendered_text)}")
    return ("*No LLM analysis in dryrun mode.*", test_name, [])


async def analyze_test_case(
    test_name: str,
    rendered_text: str,
    question_prompt: str,
    prompt_template: str,
    llm_params: dict[str, Any],
) -> tuple[str, str, list]:
    """Analyzes a single test case using LiteLLM.

    Args:
        test_name: Test case name.
        rendered_text: Rendered test case text.
        question_prompt: The analysis question/prompt.
        prompt_template: Template for formatting the prompt.
        llm_params: Parameters for LiteLLM acompletion.

    Returns:
        Tuple of (result, test_name, chunks).
    """
    logger.info(f"### Test Case: {test_name}, content length: {len(rendered_text)}")

    formatted_prompt = prompt_template.format(
        question=question_prompt,
        context=rendered_text,
    )

    messages = [{"role": "user", "content": formatted_prompt}]
    response = await _smart_acompletion(messages=messages, **llm_params)
    return (response.choices[0].message.content, test_name, [])


async def execute_llm_and_get_results(
    chunkable: ChunkableResult,
    config: DefaultConfigModel,
    include_passing: bool = False,
    dryrun: bool = False,
    quiet: bool = False,
) -> dict:
    """Executes LLM analysis on tests from ChunkableResult and returns results.

    Args:
        chunkable: ChunkableResult to iterate tests from.
        config: Parsed configuration.
        include_passing: Whether to include PASS tests.
        dryrun: If True, skip actual LLM calls.
        quiet: If True, suppress progress output.

    Returns:
        Dictionary mapping test case names to analysis results.
    """
    question_prompt = config.llm_config.question_prompt
    prompt_template = config.llm_config.prompt_template
    tokenizer = config.tokenizer
    test_case_concurrency = config.concurrency.test_case
    chunk_concurrency = config.concurrency.chunk
    chunk_analysis_prompt = config.llm_config.chunking.chunk_analysis_prompt
    final_synthesis_prompt = config.llm_config.chunking.final_synthesis_prompt

    llm_params = _build_llm_params(config.llm_factory)

    llm_results = {}
    coroutines = []
    test_case_stats = {}

    tests = list(chunkable.iter_tests(include_passing=include_passing))

    logger.info(
        f"Executing analysis, {len(tests)=}, {test_case_concurrency=}, {chunk_concurrency=}"
    )

    for test_name, test_status, lines in tests:
        rendered_text = "\n".join("    " * depth + text for depth, text in lines)
        chunk = calculate_chunk_size(rendered_text, question_prompt, tokenizer)
        test_case_stats[test_name] = (chunk, test_status)

        if dryrun:
            coroutines.append(_dryrun_result(test_name, rendered_text))
        elif not chunk.requires_chunking:
            coroutines.append(
                analyze_test_case(
                    test_name=test_name,
                    rendered_text=rendered_text,
                    question_prompt=question_prompt,
                    prompt_template=prompt_template,
                    llm_params=llm_params,
                )
            )
        else:
            chunks = chunk_rf_test_lines(lines, chunk.chunk_size)
            coroutines.append(
                accumulate_llm_results_for_summarization(
                    test_name=test_name,
                    chunks=chunks,
                    chunk_analysis_prompt=chunk_analysis_prompt,
                    final_synthesis_prompt=final_synthesis_prompt,
                    llm_params=llm_params,
                    chunk_concurrency=chunk_concurrency,
                )
            )

    semaphore = asyncio.Semaphore(test_case_concurrency)

    desc = f"Analyzing {len(tests)} test cases"
    results = await run_tasks_with_progress(
        coroutines,
        semaphore=semaphore,
        desc=desc,
        disable_progress=quiet,
    )

    for result, name, chunks in results:
        chunk, status = test_case_stats[name]
        header = _stats_header(status, chunk, dryrun, name)
        llm_results[name] = header + result

    return llm_results
