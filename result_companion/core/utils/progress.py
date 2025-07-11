import asyncio
import logging
import sys
from typing import Any, Callable, List, Optional, TypeVar

from tqdm import tqdm

T = TypeVar("T")


# Create a custom handler to ensure log messages appear above the progress bar
class TqdmLoggingHandler(logging.Handler):
    """
    Custom logging handler that ensures logs are displayed above the tqdm progress bar.
    It writes directly to sys.stdout (bypassing the progress bar).
    """

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


# Global progress bar that can be accessed across the application
_current_progress_bar: Optional[tqdm] = None


def get_current_progress_bar() -> Optional[tqdm]:
    """Get the current active progress bar instance."""
    return _current_progress_bar


def setup_progress_logging(logger_name="RC"):
    """
    Configure logging to work with tqdm progress bars.
    This ensures log messages appear above the progress bar.

    Args:
        logger_name: Name of the logger to configure
    """
    logger = logging.getLogger(logger_name)

    # Remove existing stdout handlers to prevent duplicate output
    for handler in list(logger.handlers):
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            logger.removeHandler(handler)

    # Add our custom handler
    tqdm_handler = TqdmLoggingHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    tqdm_handler.setFormatter(formatter)
    logger.addHandler(tqdm_handler)

    return logger


class AsyncTaskProgress:
    """
    A class to track progress of asynchronous tasks with a progress bar.
    The progress bar always stays at the bottom of the console output.
    """

    def __init__(self, total: int, desc: str = "Processing tasks"):
        """
        Initialize the progress tracker.

        Args:
            total: Total number of tasks to track
            desc: Description to display in the progress bar
        """
        self.total = total
        self.desc = desc
        self.progress_bar = None
        self.completed = 0
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        """Set up the progress bar when entering the context."""
        global _current_progress_bar

        # Create a progress bar that stays at the bottom of the output
        self.progress_bar = tqdm(
            total=self.total,
            desc=self.desc,
            position=0,  # Position 0 ensures it stays at the bottom
            leave=True,  # Keep the bar after completion
            dynamic_ncols=True,  # Adjust width based on terminal size
            file=sys.stdout,  # Output to stdout
            miniters=1,  # Update on each iteration
        )

        # Store reference to current progress bar
        _current_progress_bar = self.progress_bar

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up the progress bar when exiting the context."""
        global _current_progress_bar

        if self.progress_bar:
            self.progress_bar.close()
            _current_progress_bar = None

    async def update(self, n: int = 1):
        """Update the progress bar by n steps."""
        async with self._lock:
            self.completed += n
            if self.progress_bar:
                self.progress_bar.update(n)


async def track_async_tasks(
    tasks: List[asyncio.Task], desc: str = "Processing tasks"
) -> List[Any]:
    """
    Track the progress of a list of asyncio tasks with a progress bar.

    Args:
        tasks: List of asyncio tasks to track
        desc: Description to display in the progress bar

    Returns:
        List of task results in the same order as the tasks
    """
    pending = set(tasks)
    results = [None] * len(tasks)

    async with AsyncTaskProgress(len(tasks), desc=desc) as progress:
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                # Find the task's index in the original list
                for i, original_task in enumerate(tasks):
                    if task == original_task:
                        results[i] = task.result()
                        break

                await progress.update(1)

    return results


async def run_tasks_with_progress(
    coroutines: List[Callable[[], T]],
    semaphore: asyncio.Semaphore = None,
    desc: str = "Processing tasks",
) -> List[T]:
    """
    Run a list of coroutines with a semaphore and track their progress.

    Args:
        coroutines: List of coroutines to run
        semaphore: Optional semaphore to limit concurrency
        desc: Description to display in the progress bar

    Returns:
        List of results from the coroutines
    """
    if semaphore is None:
        semaphore = asyncio.Semaphore(1)

    async def run_with_semaphore(coro):
        async with semaphore:
            return await coro

    tasks = [asyncio.create_task(run_with_semaphore(coro)) for coro in coroutines]
    return await track_async_tasks(tasks, desc=desc)
