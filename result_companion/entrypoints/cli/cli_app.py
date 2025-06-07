import subprocess
from pathlib import Path
from typing import List, Optional

import typer
from click import get_current_context

from result_companion.core.analizers.local.ollama_install import (
    auto_install_model,
    auto_install_ollama,
)
from result_companion.core.analizers.local.ollama_runner import check_ollama_installed
from result_companion.core.analizers.local.ollama_server_manager import (
    OllamaServerManager,
    resolve_server_manager,
)
from result_companion.core.utils.log_levels import LogLevels
from result_companion.core.utils.logging_config import logger
from result_companion.entrypoints.run_rc import run_rc

app = typer.Typer(context_settings={"obj": {"analyze": run_rc}})
setup_app = typer.Typer(help="Manage Ollama installation and models")
app.add_typer(setup_app, name="setup")


try:
    from importlib.metadata import version as get_version
except ImportError:
    from importlib_metadata import version as get_version

VERSION = get_version("result-companion")


def version_callback(value: bool):
    if value:
        typer.echo(f"result-companion version: {VERSION}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show the version and exit",
    ),
):
    """
    Result Companion CLI - Analyze Robot Framework results with LLM assistance.
    """
    # If no subcommand is provided, show help.
    ctx = get_current_context()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command()
def analyze(
    output: Path = typer.Option(
        ...,
        "-o",
        "--output",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Output.xml file path",
    ),
    log_level: LogLevels = typer.Option(
        LogLevels.INFO,
        "-l",
        "--log-level",
        help="Log level verbosity",
        case_sensitive=True,
    ),
    config: Optional[Path] = typer.Option(
        None,
        "-c",
        "--config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="YAML Config file path",
    ),
    report: Optional[str] = typer.Option(
        None, "-r", "--report", help="Write LLM Report to HTML file"
    ),
    include_passing: bool = typer.Option(
        False, "-i", "--include-passing", help="Include PASS test cases"
    ),
    # Add model option to specify which model to use for analysis
    model: str = typer.Option(
        "llama2", "-m", "--model", help="LLM model to use for analysis"
    ),
):
    """Analyze Robot Framework test results with LLM assistance."""
    typer.echo(f"Output: {output}")
    typer.echo(f"Log Level: {log_level}")
    typer.echo(f"Config: {config}")
    typer.echo(f"Report: {report}")
    typer.echo(f"Include Passing: {include_passing}")
    typer.echo(f"Model: {model}")

    # Get the run function from context
    ctx = get_current_context()
    run = ctx.obj.get("analyze")
    if run:
        run(output, log_level, config, report, include_passing)


# Setup commands using the original functions
@setup_app.command("ollama")
def setup_ollama(
    force: bool = typer.Option(
        False,
        "--force",
        help="Force reinstallation even if already installed",
    ),
):
    """Install Ollama on the local system."""
    try:
        typer.echo("Installing Ollama...")
        auto_install_ollama()
        typer.echo("Ollama installed successfully!")
    except Exception as e:
        typer.echo(f"Error during Ollama installation: {e}")
        raise typer.Exit(code=1)


# TODO: write unittests
def install_ollama_model(
    model_name: str,
    server_manager=OllamaServerManager,
    installation_cmd: List[str] = ["ollama", "pull"],
    ollama_list_cmd: List[str] = ["ollama", "list"],
) -> bool:
    """
    Install a specific Ollama model, ensuring the server is running.

    Args:
        model_name: Name of the model to install
        server_manager: OllamaServerManager class or instance
        installation_cmd: Command to install models
        ollama_list_cmd: Command to list installed models

    Returns:
        bool: True if installation is successful

    Raises:
        Exception: If installation fails or server cannot start
    """
    check_ollama_installed()

    with resolve_server_manager(server_manager):
        logger.info(f"Installing model '{model_name}'...")

        success = auto_install_model(
            model_name=model_name,
            installation_cmd=installation_cmd,
            ollama_list_cmd=ollama_list_cmd,
        )

        if success:
            logger.info(f"Model '{model_name}' installed successfully")
            return success
        logger.error(f"Failed to install model '{model_name}'")
        raise Exception(f"Failed to install model '{model_name}'")


# TODO: write unittests
@setup_app.command("model")
def setup_model(
    model_name: str = typer.Argument(..., help="Name of the model to install"),
):
    """Install a specific model into Ollama."""
    try:
        typer.echo(f"Installing model '{model_name}'...")

        install_ollama_model(model_name)

        typer.echo(f"Model '{model_name}' installed successfully!")
    except Exception as e:
        typer.echo(f"Error installing model '{model_name}': {e}")
        logger.error(f"Model installation failed: {e}", exc_info=True)
        raise typer.Exit(code=1)


def get_installed_models(
    server_manager=OllamaServerManager, command_runner=subprocess.run
):
    """
    Get a list of installed Ollama models.

    Args:
        server_manager: OllamaServerManager class or instance
        command_runner: Function to run commands

    Returns:
        str: Output showing installed models

    Raises:
        subprocess.SubprocessError: If the command fails
    """
    with resolve_server_manager(server_manager):
        result = command_runner(
            ["ollama", "list"], capture_output=True, text=True, check=True
        )
    return result.stdout


# TODO: write unittests
@setup_app.command("list-models")
def list_models():
    """List all installed Ollama models."""
    try:
        output = get_installed_models()
        typer.echo("Installed models:")
        typer.echo(output)
        logger.debug(f"Installed models: \n{output}")
    except subprocess.SubprocessError:
        typer.echo("Error: Failed to list models. Is Ollama installed?")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
