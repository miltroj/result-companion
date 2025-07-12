import asyncio
import time
from pathlib import Path
from typing import Optional

from langchain_aws import BedrockLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama.llms import OllamaLLM
from langchain_openai import AzureChatOpenAI
from pydantic import ValidationError

from result_companion.core.analizers.factory_common import execute_llm_and_get_results
from result_companion.core.analizers.local.ollama_runner import ollama_on_init_strategy
from result_companion.core.analizers.models import MODELS
from result_companion.core.html.html_creator import create_llm_html_log
from result_companion.core.parsers.config import LLMFactoryModel, load_config
from result_companion.core.parsers.result_parser import (
    get_robot_results_from_file_as_dict,
)
from result_companion.core.utils.log_levels import LogLevels
from result_companion.core.utils.logging_config import (
    log_uncaught_exceptions,
    logger,
    set_global_log_level,
)
from result_companion.core.utils.progress import ProgressLogger

log_uncaught_exceptions(logger)


def init_llm_with_strategy_factory(
    config: LLMFactoryModel,
) -> MODELS:
    model_type = config.model_type
    parameters = config.parameters

    model_classes = {
        "OllamaLLM": (OllamaLLM, ollama_on_init_strategy),
        "AzureChatOpenAI": (AzureChatOpenAI, None),
        "BedrockLLM": (BedrockLLM, None),
        "ChatGoogleGenerativeAI": (ChatGoogleGenerativeAI, None),
    }

    if model_type not in model_classes:
        raise ValueError(
            f"Unsupported model type: {model_type} not in {model_classes.keys()}"
        )

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
) -> bool:
    # Set the log level once for all loggers through our unified system
    set_global_log_level(str(log_level))

    # Create a progress logger for the main entrypoint
    progress_logger = ProgressLogger("RC")

    progress_logger.info("Starting Result Companion!")
    start = time.time()
    # TODO: move to testable method
    parsed_config = load_config(config)
    # TODO: set output log level
    test_cases = get_robot_results_from_file_as_dict(
        file_path=output, log_level=LogLevels.TRACE
    )

    question_from_config_file = parsed_config.llm_config.question_prompt
    template = parsed_config.llm_config.prompt_template
    model, model_init_strategy = init_llm_with_strategy_factory(
        parsed_config.llm_factory
    )

    if model_init_strategy:
        progress_logger.debug(
            f"Using init strategy: {model_init_strategy} with parameters: {parsed_config.llm_factory.strategy.parameters}"
        )
        model_init_strategy(**parsed_config.llm_factory.strategy.parameters)

    progress_logger.debug(f"Prompt template: {template}")
    progress_logger.debug(f"Question loaded {question_from_config_file=}")
    prompt_template = ChatPromptTemplate.from_template(template)

    llm_results = await execute_llm_and_get_results(
        test_cases,
        parsed_config,
        prompt_template,
        model,
        include_passing=include_passing,
    )

    if report:
        create_llm_html_log(
            input_result_path=output,
            llm_output_path=report,
            llm_results=llm_results,
        )
    stop = time.time()
    progress_logger.debug(f"Execution time: {stop - start}")
    return True


def run_rc(
    output: Path,
    log_level: LogLevels,
    config: Optional[Path],
    report: Optional[str],
    include_passing: bool,
) -> bool:
    return asyncio.run(
        _main(
            output=output,
            log_level=log_level,
            config=config,
            report=report,
            include_passing=include_passing,
        )
    )
