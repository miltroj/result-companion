import asyncio
from typing import Any, List, TypeVar

from tqdm import tqdm

T = TypeVar("T")


async def run_tasks_with_progress(
    coroutines: List,
    semaphore: asyncio.Semaphore = None,
    desc: str = "Processing tasks",
) -> List[Any]:
    """Run coroutines with progress bar tracking.

    Args:
        coroutines: List of coroutines to run.
        semaphore: Optional semaphore to limit concurrency.
        desc: Description to display in the progress bar.

    Returns:
        List of results from the coroutines.
    """
    if not coroutines:
        return []

    if semaphore is None:
        semaphore = asyncio.Semaphore(1)

    async def run_with_semaphore(coro):
        async with semaphore:
            return await coro

    tasks = [asyncio.create_task(run_with_semaphore(coro)) for coro in coroutines]
    results = [None] * len(tasks)
    task_to_index = {task: i for i, task in enumerate(tasks)}
    pending = set(tasks)

    with tqdm(
        total=len(tasks),
        desc=desc,
        position=0,
        leave=True,
        dynamic_ncols=True,
    ) as pbar:
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                results[task_to_index[task]] = task.result()
                pbar.update(1)

    return results
