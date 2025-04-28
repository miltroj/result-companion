from pathlib import Path
from typing import Optional

import typer
from click import get_current_context

# Keep the original functions
from result_companion.core.analizers.local.ollama_install import (
    auto_install_model,
    auto_install_ollama,
)
from result_companion.core.utils.log_levels import LogLevels
from result_companion.core.utils.logging_config import logger
from result_companion.entrypoints.run_rc import run_rc

# Create separate command groups for better organization
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


@setup_app.command("model")
def setup_model(
    model_name: str = typer.Argument(..., help="Name of the model to install"),
):
    """Install a specific model into Ollama."""
    try:
        typer.echo(f"Installing model '{model_name}'...")
        auto_install_model(model_name)
        typer.echo(f"Model '{model_name}' installed successfully!")
    except Exception as e:
        typer.echo(f"Error installing model '{model_name}': {e}")
        raise typer.Exit(code=1)


# Add a command to list installed models
@setup_app.command("list-models")
def list_models():
    """List all installed Ollama models."""
    try:
        import subprocess

        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, check=True
        )
        typer.echo("Installed models:")
        typer.echo(result.stdout)
    except subprocess.SubprocessError:
        typer.echo("Error: Failed to list models. Is Ollama installed?")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(code=1)


# Keep the original 'install' command for backward compatibility
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
    Manage the local Ollama installation (legacy command).

    Consider using 'setup ollama' and 'setup model' instead.
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
