from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from result_companion.core.analizers.llm_router import _smart_acompletion
from result_companion.core.chunking.chunking import (
    accumulate_llm_results_for_summarization,
)
from result_companion.core.chunking.utils import Chunking
from result_companion.core.parsers.config import DefaultConfigModel, LLMFactoryModel
from result_companion.core.utils.logging_config import get_llm_io_logger, get_progress_logger
from result_companion.core.utils.progress import run_tasks_with_progress

if TYPE_CHECKING:
    from result_companion.core.chunking.rf_results import ContextAwareRobotResults

logger = get_progress_logger("Analyzer")
llm_io = get_llm_io_logger()


def _stats_header(
    test_status: str,
    chunk_stats: Chunking,
    dryrun: bool = False,
    test_name: str = "",
) -> str:
    """Returns markdown stats line for analysis.

    Args:
        test_status: Test case status (PASS/FAIL).
        chunk_stats: Chunking information.
        dryrun: Whether this is a dryrun.
        test_name: Test case name.

    Returns:
        Formatted markdown header string.
    """
    chunks = chunk_stats.number_of_chunks if chunk_stats.requires_chunking else 0
    prefix = "**[DRYRUN]** " if dryrun else ""
    return f"""## {prefix} {test_name}

#### Status: {test_status} · Chunks: {chunks} · Tokens: ~{chunk_stats.tokens_from_raw_text} · Raw length: {chunk_stats.raw_text_len}

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

    # Merge any additional parameters
    params.update(llm_factory.parameters)

    return params


async def _dryrun_result(test_name: str) -> tuple[str, str, list]:
    """Returns placeholder without calling LLM.

    Args:
        test_name: Test case name.

    Returns:
        Tuple of (result, test_name, chunks).
    """
    logger.info(f"### Test Case: {test_name} [dryrun]")
    return ("*No LLM analysis in dryrun mode.*", test_name, [])


async def analyze_test_case(
    test_name: str,
    test_case_text: str,
    question_prompt: str,
    prompt_template: str,
    llm_params: dict[str, Any],
) -> tuple[str, str, list]:
    """Analyzes a single test case using LiteLLM.

    Args:
        test_name: Name of the test case.
        test_case_text: Rendered test case text.
        question_prompt: The analysis question/prompt.
        prompt_template: Template for formatting the prompt.
        llm_params: Parameters for LiteLLM acompletion.

    Returns:
        Tuple of (result, test_name, chunks).
    """
    logger.info(f"### Test Case: {test_name}, content length: {len(test_case_text)}")

    formatted_prompt = prompt_template.format(
        question=question_prompt,
        context=test_case_text,
    )

    response = await _smart_acompletion(
        messages=[{"role": "user", "content": formatted_prompt}], **llm_params
    )
    content = response.choices[0].message.content
    llm_io.debug(
        f"\n{'='*60}\n[{test_name}] (single chunk)\n"
        f"--- PROMPT ---\n{formatted_prompt}\n"
        f"--- RESPONSE ---\n{content}\n"
    )
    return (content, test_name, [])


async def execute_llm_and_get_results(
    results: ContextAwareRobotResults,
    config: DefaultConfigModel,
    dryrun: bool = False,
    quiet: bool = False,
) -> dict:
    """Executes LLM analysis on pre-configured ContextAwareRobotResults.

    Args:
        results: Configured ContextAwareRobotResults with chunking strategy set.
        config: Parsed configuration.
        dryrun: If True, skip actual LLM calls.
        quiet: If True, suppress progress output.

    Returns:
        Dictionary mapping test case names to analysis results.
    """
    question_prompt = config.llm_config.question_prompt
    prompt_template = config.llm_config.prompt_template
    test_case_concurrency = config.concurrency.test_case
    chunk_concurrency = config.concurrency.chunk
    chunk_analysis_prompt = config.llm_config.chunking.chunk_analysis_prompt
    final_synthesis_prompt = config.llm_config.chunking.final_synthesis_prompt
    llm_params = _build_llm_params(config.llm_factory)

    coroutines = []
    test_case_stats: dict[str, tuple[Chunking, str]] = {}

    for test_name, chunks, chunk_stats, test_status in results.render_chunks():
        test_case_stats[test_name] = (chunk_stats, test_status)

        if dryrun:
            coroutines.append(_dryrun_result(test_name))
        elif len(chunks) == 1:
            coroutines.append(
                analyze_test_case(
                    test_name=test_name,
                    test_case_text=chunks[0],
                    question_prompt=question_prompt,
                    prompt_template=prompt_template,
                    llm_params=llm_params,
                )
            )
        else:
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

    test_count = len(coroutines)
    logger.info(
        f"Executing analysis, {test_count=}, {test_case_concurrency=}, {chunk_concurrency=}"
    )

    semaphore = asyncio.Semaphore(test_case_concurrency)
    task_results = await run_tasks_with_progress(
        coroutines,
        semaphore=semaphore,
        desc=f"Analyzing {test_count} test cases",
        disable_progress=quiet,
    )

    llm_results = {}
    for result, name, _chunks in task_results:
        chunk_stats, test_status = test_case_stats[name]
        header = _stats_header(test_status, chunk_stats, dryrun, name)
        llm_results[name] = header + result

    return llm_results
