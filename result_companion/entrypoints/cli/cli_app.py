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

app = typer.Typer()
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
    html_report: bool = typer.Option(
        True,
        "--html-report/--no-html-report",
        help="Enable or disable HTML report generation",
    ),
    text_report: Optional[str] = typer.Option(
        None, "--text-report", help="Write concise text summary to file"
    ),
    print_text_report: bool = typer.Option(
        False, "--print-text-report", help="Print concise text report to stdout"
    ),
    overall_summary: bool = typer.Option(
        True,
        "--overall-summary/--no-overall-summary",
        help="Run extra LLM pass to synthesize all per-test results (enabled by default)",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress logs/progress and CLI parameter echo output",
    ),
    include_passing: bool = typer.Option(
        False, "-i", "--include-passing", help="Include PASS test cases"
    ),
    include_tags: Optional[str] = typer.Option(
        None,
        "-I",
        "--include",
        help="Include tests by tags (comma-separated: 'smoke,critical*')",
    ),
    exclude_tags: Optional[str] = typer.Option(
        None,
        "-E",
        "--exclude",
        help="Exclude tests by tags (comma-separated: 'wip,bug*')",
    ),
    test_case_concurrency: Optional[int] = typer.Option(
        None,
        "--test-concurrency",
        help="Test cases processed in parallel (overrides config)",
    ),
    chunk_concurrency: Optional[int] = typer.Option(
        None,
        "--chunk-concurrency",
        help="Chunks per test case in parallel (overrides config)",
    ),
    dryrun: bool = typer.Option(
        False,
        "--dryrun",
        help="Skip LLM calls, generate HTML with debug metadata",
    ),
):
    """Analyze Robot Framework test results with LLM assistance."""
    if not quiet:
        typer.echo(f"Output: {output}")
        typer.echo(f"Log Level: {log_level}")
        typer.echo(f"Config: {config}")
        typer.echo(f"Report: {report}")
        typer.echo(f"HTML Report: {html_report}")
        typer.echo(f"Text Report: {text_report}")
        typer.echo(f"Print Text Report: {print_text_report}")
        typer.echo(f"Overall Summary: {overall_summary}")
        typer.echo(f"Include Passing: {include_passing}")

    # Parse CLI tag options
    include_tag_list = (
        [t.strip() for t in include_tags.split(",")] if include_tags else None
    )
    exclude_tag_list = (
        [t.strip() for t in exclude_tags.split(",")] if exclude_tags else None
    )

    # Allow test injection via context, otherwise lazy import
    ctx = get_current_context()
    run = ctx.obj.get("analyze") if ctx.obj else None
    if not run:
        from result_companion.entrypoints.run_rc import run_rc

        run = run_rc

    run(
        output=output,
        log_level=log_level,
        config=config,
        report=report,
        include_passing=include_passing,
        test_case_concurrency=test_case_concurrency,
        chunk_concurrency=chunk_concurrency,
        include_tags=include_tag_list,
        exclude_tags=exclude_tag_list,
        dryrun=dryrun,
        html_report=html_report,
        text_report=text_report,
        print_text_report=print_text_report,
        summarize_failures=overall_summary,
        quiet=quiet,
    )


@app.command()
def review(
    summary: Path = typer.Option(
        ...,
        "-s",
        "--summary",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Text summary file from 'analyze --text-report'",
    ),
    repo: str = typer.Option(
        ...,
        "--repo",
        envvar="GITHUB_REPOSITORY",
        help="GitHub repo (owner/repo). Defaults to GITHUB_REPOSITORY",
    ),
    pr: int = typer.Option(
        ...,
        "--pr",
        help="Pull request number",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "-c",
        "--config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Review config YAML (overrides defaults)",
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        help="Print review comment instead of posting to PR (Copilot still runs)",
    ),
    notify_on_pass: bool = typer.Option(
        False,
        "--notify-on-pass",
        help="Post a short all-clear comment when no test failures are found",
    ),
    log_level: LogLevels = typer.Option(
        LogLevels.INFO,
        "-l",
        "--log-level",
        help="Log level verbosity",
        case_sensitive=True,
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress logs/progress output",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Override Copilot model from config",
    ),
):
    """Post AI test failure analysis as a PR comment."""
    from result_companion.core.utils.logging_config import set_global_log_level

    resolved_log_level = "ERROR" if quiet else str(log_level)
    set_global_log_level(resolved_log_level)

    ctx = get_current_context()
    run = ctx.obj.get("review") if ctx.obj else None
    if not run:
        from result_companion.core.review.pr_reviewer import run_review

        run = run_review

    failure_summary = summary.read_text()
    try:
        result = run(
            repo_name=repo,
            pr_number=pr,
            failure_summary=failure_summary,
            config_path=config,
            preview=preview,
            notify_on_pass=notify_on_pass,
            model=model,
        )
        if result:
            typer.echo(result)
    except Exception as e:
        typer.echo(f"Review failed: {e}", err=True)
        raise typer.Exit(code=1)


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
        )

        if success:
            logger.info(f"Model '{model_name}' installed successfully")
            return success
        logger.error(f"Failed to install model '{model_name}'")
        raise Exception(f"Failed to install model '{model_name}'")


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
