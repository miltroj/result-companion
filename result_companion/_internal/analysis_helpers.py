"""Shared analysis helpers for API and CLI entrypoints."""

from typing import Optional

from result_companion.core.analizers.local.ollama_runner import ollama_on_init_strategy
from result_companion.core.parsers.config import DefaultConfigModel
from result_companion.core.utils.logging_config import logger


def run_provider_init_strategies(model_name: str) -> None:
    """Runs provider-specific initialization based on LiteLLM model prefix.

    Args:
        model_name: LiteLLM model identifier (e.g., ollama_chat/llama2).
    """
    if not model_name.startswith("ollama"):
        return

    parts = model_name.split("/")
    model_short = parts[1].split(":")[0] if len(parts) > 1 else model_name.split(":")[0]
    logger.debug(f"Running Ollama init strategy for model: {model_short}")
    ollama_on_init_strategy(model_name=model_short)


def apply_concurrency_overrides(
    config: DefaultConfigModel,
    test_case_concurrency: Optional[int],
    chunk_concurrency: Optional[int],
) -> None:
    """Mutates config concurrency settings with CLI/caller overrides."""
    if test_case_concurrency is not None:
        config.concurrency.test_case = test_case_concurrency
    if chunk_concurrency is not None:
        config.concurrency.chunk = chunk_concurrency


def resolve_tags(
    caller_tags: Optional[list[str]],
    config_tags: list[str],
) -> Optional[list[str]]:
    """Returns caller tags if provided, else config tags, else None."""
    return caller_tags or config_tags or None
