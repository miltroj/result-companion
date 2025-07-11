import asyncio
import logging
import sys
from typing import Any, Callable, List, Optional, TypeVar

from tqdm import tqdm

T = TypeVar("T")


# Track global log level to ensure consistency across loggers
_global_log_level = logging.INFO


def set_progress_log_level(level) -> None:
    """
    Set the log level for all progress loggers (current and future).

    Args:
        level: Log level to set (can be string or int level)
    """
    global _global_log_level

    # Convert string to log level if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    _global_log_level = level

    # Update all existing loggers
    root_logger = logging.getLogger()
    for logger in [root_logger] + list(root_logger.manager.loggerDict.values()):
        if hasattr(logger, "handlers"):  # Only actual loggers, not PlaceHolders
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)


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


# Global variable for tracking the current progress bar
_current_progress_bar: Optional[tqdm] = None


def get_current_progress_bar() -> Optional[tqdm]:
    """Get the current active progress bar instance."""
    return _current_progress_bar


class ProgressLogger:
    """
    A logger configured to work with tqdm progress bars.
    This class replaces the singleton pattern with a dependency injection approach.
    """

    def __init__(self, logger_name: str = "RC"):
        """
        Create and configure a logger that works with tqdm progress bars.

        Args:
            logger_name: Name of the logger to configure
        """
        self.logger = self._setup_logger(logger_name)

    def _setup_logger(self, logger_name: str) -> logging.Logger:
        """
        Configure a logger to work with tqdm progress bars.
        This ensures log messages appear above the progress bar.

        Args:
            logger_name: Name of the logger to configure

        Returns:
            The configured logger
        """
        logger = logging.getLogger(logger_name)

        # Apply global log level to ensure consistency
        logger.setLevel(_global_log_level)

        # Check if we've already added our handler
        has_tqdm_handler = False
        for handler in logger.handlers:
            if isinstance(handler, TqdmLoggingHandler):
                has_tqdm_handler = True
                # Make sure the handler has the right level
                handler.setLevel(_global_log_level)

        # Only add a new handler if needed
        if not has_tqdm_handler:
            # Remove existing stdout handlers to prevent duplicate output
            for handler in list(logger.handlers):
                if (
                    isinstance(handler, logging.StreamHandler)
                    and handler.stream == sys.stdout
                ):
                    logger.removeHandler(handler)

            # Add our custom handler
            tqdm_handler = TqdmLoggingHandler()
            tqdm_handler.setLevel(_global_log_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            tqdm_handler.setFormatter(formatter)
            logger.addHandler(tqdm_handler)

        return logger

    def debug(self, msg, *args, **kwargs):
        """Log a debug message."""
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """Log an info message."""
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Log a warning message."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Log an error message."""
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """Log a critical message."""
        self.logger.critical(msg, *args, **kwargs)

    def set_level(self, level):
        """Set the log level for this logger instance."""
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)

        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)


class AsyncTaskProgress:
    """
    A class to track progress of asynchronous tasks with a progress bar.
    The progress bar always stays at the bottom of the console output.
    """

    def __init__(
        self,
        total: int,
        desc: str = "Processing tasks",
        logger: Optional[ProgressLogger] = None,
    ):
        """
        Initialize the progress tracker.

        Args:
            total: Total number of tasks to track
            desc: Description to display in the progress bar
            logger: Optional logger instance for logging during progress tracking
        """
        self.total = total
        self.desc = desc
        self.progress_bar = None
        self.completed = 0
        self._lock = asyncio.Lock()
        self.logger = logger or ProgressLogger()

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

    def debug(self, msg, *args, **kwargs):
        """Log a debug message above the progress bar."""
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """Log an info message above the progress bar."""
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Log a warning message above the progress bar."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Log an error message above the progress bar."""
        self.logger.error(msg, *args, **kwargs)


async def track_async_tasks(
    tasks: List[asyncio.Task],
    desc: str = "Processing tasks",
    logger: Optional[ProgressLogger] = None,
) -> List[Any]:
    """
    Track the progress of a list of asyncio tasks with a progress bar.

    Args:
        tasks: List of asyncio tasks to track
        desc: Description to display in the progress bar
        logger: Optional logger instance for logging during task tracking

    Returns:
        List of task results in the same order as the tasks
    """
    pending = set(tasks)
    results = [None] * len(tasks)

    async with AsyncTaskProgress(len(tasks), desc=desc, logger=logger) as progress:
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
    logger: Optional[ProgressLogger] = None,
) -> List[T]:
    """
    Run a list of coroutines with a semaphore and track their progress.

    Args:
        coroutines: List of coroutines to run
        semaphore: Optional semaphore to limit concurrency
        desc: Description to display in the progress bar
        logger: Optional logger instance for logging during task execution

    Returns:
        List of results from the coroutines
    """
    if semaphore is None:
        semaphore = asyncio.Semaphore(1)

    async def run_with_semaphore(coro):
        async with semaphore:
            return await coro

    tasks = [asyncio.create_task(run_with_semaphore(coro)) for coro in coroutines]
    return await track_async_tasks(tasks, desc=desc, logger=logger)
