import asyncio
from typing import Any

from result_companion.core.analizers.llm_router import _smart_acompletion
from result_companion.core.chunking.chunking import (
    accumulate_llm_results_for_summarization,
    chunk_rf_test_lines,
    deduplicate_consecutive_lines,
    render_lines_to_text,
    render_rf_test_structure,
)
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
    return (response.choices[0].message.content, test_name, [])


async def execute_llm_and_get_results(
    test_cases: list,
    config: DefaultConfigModel,
    dryrun: bool = False,
    quiet: bool = False,
) -> dict:
    """Executes LLM analysis on test cases and returns results.

    Args:
        test_cases: List of test case dictionaries.
        config: Parsed configuration.
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
    test_case_stats = {}  # name -> (chunk, status) for adding headers later

    logger.info(
        f"Executing analysis, {len(test_cases)=}, {test_case_concurrency=}, {chunk_concurrency=}"
    )

    for test_case in test_cases:
        lines = deduplicate_consecutive_lines(render_rf_test_structure(test_case))
        rendered_text = render_lines_to_text(lines)
        logger.debug(
            f"Test case {test_case['name']} full rendered text:\n{rendered_text}"
        )
        chunk_info = calculate_chunk_size(rendered_text, question_prompt, tokenizer)
        chunks = chunk_rf_test_lines(lines, chunk_info.chunk_size)
        test_case_stats[test_case["name"]] = (
            chunk_info,
            test_case.get("status", "N/A"),
        )

        if dryrun:
            coroutines.append(_dryrun_result(test_case["name"]))
        elif len(chunks) == 1:
            coroutines.append(
                analyze_test_case(
                    test_name=test_case["name"],
                    test_case_text=chunks[0],
                    question_prompt=question_prompt,
                    prompt_template=prompt_template,
                    llm_params=llm_params,
                )
            )
        else:
            coroutines.append(
                accumulate_llm_results_for_summarization(
                    test_name=test_case["name"],
                    chunks=chunks,
                    chunk_analysis_prompt=chunk_analysis_prompt,
                    final_synthesis_prompt=final_synthesis_prompt,
                    llm_params=llm_params,
                    chunk_concurrency=chunk_concurrency,
                )
            )

    semaphore = asyncio.Semaphore(test_case_concurrency)

    desc = f"Analyzing {len(test_cases)} test cases"
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
