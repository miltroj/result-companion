import asyncio
import time
import typer

from pathlib import Path
from typing import Optional

from langchain_aws import BedrockLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_openai import AzureChatOpenAI
from pydantic import ValidationError

from result_companion.analizers.factory_common import execute_llm_and_get_results
from result_companion.analizers.local.ollama_runner import ollama_on_init_strategy
from result_companion.analizers.models import MODELS
from result_companion.html.html_creator import create_llm_html_log
from result_companion.utils.log_levels import LogLevels
from result_companion.parsers.config import LLMFactoryModel, load_config
from result_companion.parsers.result_parser import get_robot_results_from_file_as_dict
from result_companion.utils.logging_config import (
    log_uncaught_exceptions,
    set_global_log_level,
    setup_logging,
)

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


async def _main(output: Path, log_level: LogLevels, config: Optional[Path], report: Optional[str], diff: Optional[Path], include_passing: bool) -> bool:
    set_global_log_level(log_level=str(log_level))
    logger.info(f"Starting Result Companion!")
    start = time.time()
    # TODO: move to testable method
    parsed_config = load_config(config)
    # TODO: set output log level
    test_cases = get_robot_results_from_file_as_dict(
        file_path=output, log_level=LogLevels.TRACE
    )

    question_from_config_file = parsed_config.llm_config.question_prompt
    template = parsed_config.llm_config.prompt_template
    model, model_init_strategy = init_llm_with_strategy_factory(parsed_config.llm_factory)

    if model_init_strategy:
        logger.debug(
            f"Using init strategy: {model_init_strategy} with parameters: {parsed_config.llm_factory.parameters}"
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
        include_passing=include_passing,
    )

    if report:
        create_llm_html_log(
            input_result_path=output,
            llm_output_path=report,
            llm_results=llm_results,
        )
    stop = time.time()
    logger.debug(f"Execution time: {stop - start}")
    return True


app = typer.Typer()

@app.command()
def main(
    output: Path = typer.Option(..., "-o", "--output", exists=True, file_okay=True, dir_okay=False, readable=True, help="Output.xml file path"),
    log_level: LogLevels = typer.Option(LogLevels.INFO, "-l", "--log-level", help="Log level verbosity", case_sensitive=True),
    config: Optional[Path] = typer.Option(None, "-c", "--config", exists=True, file_okay=True, dir_okay=False, readable=True, help="YAML Config file path"),
    report: Optional[str] = typer.Option(None, "-r", "--report", help="Write LLM Report to HTML file"),
    diff: Optional[Path] = typer.Option(None, "-d", "--diff", exists=True, file_okay=True, dir_okay=False, readable=True, help="Diff with other XML file"),
    include_passing: bool = typer.Option(False, "-i", "--include-passing", help="Include PASS test cases"),
):
    """Test Result Companion - CLI"""
    typer.echo(f"Output: {output}")
    typer.echo(f"Log Level: {log_level}")
    typer.echo(f"Config: {config}")
    typer.echo(f"Report: {report}")
    typer.echo(f"Diff: {diff}")
    typer.echo(f"Include Passing: {include_passing}")

    asyncio.run(_main(output=output,
                      log_level=log_level,
                      config=config,
                      report=report,
                      diff=diff,
                      include_passing=include_passing))

if __name__ == "__main__":
    app()
