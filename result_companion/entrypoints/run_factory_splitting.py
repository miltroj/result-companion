import asyncio
import time
from sys import argv

from langchain_aws import BedrockLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_openai import AzureChatOpenAI
from pydantic import ValidationError

from result_companion.analizers.factory_common import execute_llm_and_get_results
from result_companion.analizers.local.ollama_runner import ollama_on_init_strategy
from result_companion.analizers.models import MODELS
from result_companion.html.html_creator import create_llm_html_log
from result_companion.parsers.cli_parser import OutputLogLevel, parse_args
from result_companion.parsers.config import LLMFactoryModel, load_config
from result_companion.parsers.result_parser import get_robot_results_from_file_as_dict
from result_companion.utils.logging_config import (
    log_uncaught_exceptions,
    set_global_log_level,
    setup_logging,
)
from result_companion.utils.utils import file_exists

logger = setup_logging("RC")
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


async def _main(args=argv[1:], file_exists=file_exists) -> bool:
    arguments = parse_args(file_exists=file_exists).parse_args(args)
    set_global_log_level(log_level=arguments.log_level)
    logger.info(f"Starting Result Companion!")
    logger.debug(arguments)
    start = time.time()
    # TODO: move to testable method
    config = load_config(arguments)
    # TODO: set output log level
    test_cases = get_robot_results_from_file_as_dict(
        file_path=arguments.output, log_level=OutputLogLevel.TRACE
    )

    question_from_config_file = config.llm_config.question_prompt
    template = config.llm_config.prompt_template
    model, model_init_strategy = init_llm_with_strategy_factory(config.llm_factory)

    if model_init_strategy:
        logger.debug(
            f"Using init strategy: {model_init_strategy} with parameters: {config.llm_factory.parameters}"
        )
        model_init_strategy(**config.llm_factory.strategy.parameters)

    logger.debug(f"Prompt template: {template}")
    logger.debug(f"Question loaded {question_from_config_file=}")
    prompt_template = ChatPromptTemplate.from_template(template)

    llm_results = await execute_llm_and_get_results(
        test_cases, config, prompt_template, model, arguments.include_passing
    )

    if arguments.report:
        create_llm_html_log(
            input_result_path=arguments.output,
            llm_output_path=arguments.report,
            llm_results=llm_results,
        )
    stop = time.time()
    logger.debug(f"Execution time: {stop - start}")
    return True


def main(args=argv[1:]):
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
