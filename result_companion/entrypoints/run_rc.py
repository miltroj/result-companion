import asyncio
import time
from pathlib import Path
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_aws import BedrockLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama.llms import OllamaLLM
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import ValidationError

from result_companion.core.analizers.factory_common import execute_llm_and_get_results
from result_companion.core.analizers.local.ollama_runner import ollama_on_init_strategy
from result_companion.core.analizers.models import MODELS
from result_companion.core.analizers.remote.copilot import ChatCopilot
from result_companion.core.html.html_creator import create_llm_html_log
from result_companion.core.parsers.config import LLMFactoryModel, load_config
from result_companion.core.parsers.result_parser import (
    get_robot_results_from_file_as_dict,
)
from result_companion.core.utils.log_levels import LogLevels
from result_companion.core.utils.logging_config import logger, set_global_log_level


def init_llm_with_strategy_factory(
    config: LLMFactoryModel,
    concurrency: int = 1,
) -> MODELS:
    """Creates LLM model with optional init strategy."""
    model_type = config.model_type
    parameters = dict(config.parameters)

    model_classes = {
        "OllamaLLM": (OllamaLLM, ollama_on_init_strategy),
        "AzureChatOpenAI": (AzureChatOpenAI, None),
        "BedrockLLM": (BedrockLLM, None),
        "ChatGoogleGenerativeAI": (ChatGoogleGenerativeAI, None),
        "ChatOpenAI": (ChatOpenAI, None),
        "ChatAnthropic": (ChatAnthropic, None),
        "ChatCopilot": (ChatCopilot, None),
    }

    # Runtime overrides: model needs concurrency-derived params
    runtime_overrides = {
        "ChatCopilot": {"pool_size": concurrency},
    }

    if model_type not in model_classes:
        raise ValueError(
            f"Unsupported model type: {model_type} not in {model_classes.keys()}"
        )

    if model_type in runtime_overrides:
        parameters.update(runtime_overrides[model_type])

    model_class, strategy = model_classes[model_type]
    try:
        return model_class(**parameters), strategy
    except (TypeError, ValidationError) as e:
        raise ValueError(
            f"Invalid parameters for {model_type}: {parameters}, while available parameters are: {model_class.__init__.__annotations__}"
        ) from e


async def _main(
    output: Path,
    log_level: LogLevels,
    config: Optional[Path],
    report: Optional[str],
    include_passing: bool,
    test_case_concurrency: Optional[int] = None,
    chunk_concurrency: Optional[int] = None,
    include_tags: Optional[list[str]] = None,
    exclude_tags: Optional[list[str]] = None,
    dryrun: bool = False,
) -> bool:
    set_global_log_level(str(log_level))

    logger.info("Starting Result Companion!")
    start = time.time()
    # TODO: move to testable method
    parsed_config = load_config(config)

    if test_case_concurrency is not None:
        parsed_config.concurrency.test_case = test_case_concurrency
    if chunk_concurrency is not None:
        parsed_config.concurrency.chunk = chunk_concurrency

    # Merge CLI tags with config (CLI takes precedence)
    final_include = include_tags or parsed_config.test_filter.include_tags or None
    final_exclude = exclude_tags or parsed_config.test_filter.exclude_tags or None

    # Use RF's native filtering (same as rebot --include/--exclude)
    # TODO: set output log level from config or cli
    test_cases = get_robot_results_from_file_as_dict(
        file_path=output,
        log_level=LogLevels.DEBUG,
        include_tags=final_include,
        exclude_tags=final_exclude,
    )

    # Filter passing tests (RF doesn't have this natively)
    should_include_passing = (
        include_passing or parsed_config.test_filter.include_passing
    )
    if not should_include_passing:
        test_cases = [t for t in test_cases if t.get("status") != "PASS"]

    logger.info(f"Filtered to {len(test_cases)} test cases")

    question_from_config_file = parsed_config.llm_config.question_prompt
    template = parsed_config.llm_config.prompt_template
    model, model_init_strategy = init_llm_with_strategy_factory(
        parsed_config.llm_factory,
        concurrency=parsed_config.concurrency.test_case,
    )

    if model_init_strategy:
        logger.debug(
            f"Using init strategy: {model_init_strategy} with parameters: {parsed_config.llm_factory.strategy.parameters}"
        )
        model_init_strategy(**parsed_config.llm_factory.strategy.parameters)

    logger.debug(f"Prompt template: {template}")
    logger.debug(f"Question loaded {question_from_config_file=}")
    prompt_template = ChatPromptTemplate.from_template(template)

    llm_results = await execute_llm_and_get_results(
        test_cases,
        parsed_config,
        prompt_template,
        model,
        dryrun=dryrun,
    )

    report_path = report if report else "rc_log.html"
    if llm_results:
        model_info = {
            "model": parsed_config.llm_factory.parameters.get(
                "model", parsed_config.llm_factory.model_type
            )
        }
        create_llm_html_log(
            input_result_path=output,
            llm_output_path=report_path,
            llm_results=llm_results,
            model_info=model_info,
        )
        logger.info(f"Report created: {Path(report_path).resolve()}")

    stop = time.time()
    logger.debug(f"Execution time: {stop - start}")
    return True


def run_rc(
    output: Path,
    log_level: LogLevels,
    config: Optional[Path],
    report: Optional[str],
    include_passing: bool,
    test_case_concurrency: Optional[int] = None,
    chunk_concurrency: Optional[int] = None,
    include_tags: Optional[list[str]] = None,
    exclude_tags: Optional[list[str]] = None,
    dryrun: bool = False,
) -> bool:
    try:
        return asyncio.run(
            _main(
                output=output,
                log_level=log_level,
                config=config,
                report=report,
                include_passing=include_passing,
                test_case_concurrency=test_case_concurrency,
                chunk_concurrency=chunk_concurrency,
                include_tags=include_tags,
                exclude_tags=exclude_tags,
                dryrun=dryrun,
            )
        )
    except Exception:
        # logging unhandled exceptions to file from asyncio.run
        logger.critical("Unhandled exception", exc_info=True)
        raise
