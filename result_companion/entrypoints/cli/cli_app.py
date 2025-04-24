from pathlib import Path
from typing import Optional

import typer
from click import get_current_context

from result_companion.core.analizers.local.ollama_install import (
    auto_install_model,
    auto_install_ollama,
)
from result_companion.core.utils.log_levels import LogLevels
from result_companion.core.utils.logging_config import logger
from result_companion.entrypoints.run_rc import run_rc

app = typer.Typer(context_settings={"obj": {"analyze": run_rc}})


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
    Result Companion CLI.
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
    _main_func: typer.Context = run_rc,
):
    """Test Result Companion - CLI"""
    typer.echo(f"Output: {output}")
    typer.echo(f"Log Level: {log_level}")
    typer.echo(f"Config: {config}")
    typer.echo(f"Report: {report}")
    typer.echo(f"Include Passing: {include_passing}")
    run = _main_func.obj.get("rc")
    if run:
        run(output, log_level, config, report, include_passing)


@app.command("install")
def installer(
    install: bool = typer.Option(
        False,
        "--install-ollama",
        help="Automatically install Ollama locally if not installed",
    ),
    install_model: Optional[str] = typer.Option(
        None,
        "--install-model",
        help="Automatically install the specified LLM model into Ollama",
    ),
):
    """
    Manage the local Ollama installation.
    """
    if install:
        try:
            typer.echo("Attempting to install Ollama...")
            auto_install_ollama()
            logger.debug(f"Auto-installation command: {auto_install_ollama}")
            typer.echo("Ollama installed successfully!")
        except Exception as e:
            typer.echo(f"Error during Ollama installation: {e}")
            raise typer.Exit(code=1)
    if install_model:
        try:
            typer.echo(f"Attempting to install model '{install_model}' into Ollama...")
            logger.debug(f"Model installation command: {install_model}")
            auto_install_model(install_model)
            typer.echo(f"Model '{install_model}' installed successfully!")
        except Exception as e:
            typer.echo(f"Error installing model '{install_model}': {e}")
            raise typer.Exit(code=1)
    if not install and not install_model:
        typer.echo("No action specified. Use --help to see available options.")


if __name__ == "__main__":
    app()
