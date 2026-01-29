import asyncio
from typing import Tuple

from langchain_core.runnables import RunnableSerializable


async def accumulate_llm_results(
    test_case: list, question_from_config_file: str, chain: RunnableSerializable
) -> Tuple[str, str]:
    print(
        f"\n### Test Case: {test_case['name']}, content length: {len(str(test_case))}"
    )
    result = []
    async for chunk in chain.astream(
        {"context": test_case, "question": question_from_config_file}, verbose=True
    ):
        result.append(chunk)
        print(chunk, end="", flush=True)
    return "".join(result), test_case["id"]


async def run_llm_based_analysis_and_stream_results(
    test_cases: list, question_from_config_file: str, chain: RunnableSerializable
) -> dict:

    llm_results = dict()
    for test_case in test_cases:
        llm_results[test_case["id"]], _ = await accumulate_llm_results(
            test_case, question_from_config_file, chain
        )

    return llm_results


async def run_with_semaphore(semaphore: asyncio.Semaphore, coroutine: any) -> any:
    async with semaphore:
        return await coroutine


async def run_llm_api_calls_based_analysis_and_stream_results(
    test_cases: list, question_from_config_file: str, chain: RunnableSerializable
) -> dict:
    llm_results = dict()
    corutines = []

    for test_case in test_cases:
        corutines.append(
            accumulate_llm_results(test_case, question_from_config_file, chain)
        )

    semaphore = asyncio.Semaphore(1)  # Limit concurrency

    tasks = [run_with_semaphore(semaphore, coroutine) for coroutine in corutines]

    for result, name in await asyncio.gather(*tasks):
        llm_results[name] = result

    return llm_results
