from pathlib import Path
from typing import Optional

import typer

from result_companion.core.utils.log_levels import LogLevels
from result_companion.entrypoints.run_rc import run_rc

app = typer.Typer(context_settings={"obj": {"main": run_rc}})


try:
    from importlib.metadata import version as get_version
except ImportError:
    from importlib_metadata import version as get_version

VERSION = get_version("result-companion")


def version_callback(value: bool):
    if value:
        typer.echo(f"result-companion version: {VERSION}")
        raise typer.Exit()


@app.command()
def main(
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
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the version and exit",
        callback=version_callback,
        is_eager=True,
    ),
    _main_func: typer.Context = run_rc,
):
    """Test Result Companion - CLI"""
    typer.echo(f"Output: {output}")
    typer.echo(f"Log Level: {log_level}")
    typer.echo(f"Config: {config}")
    typer.echo(f"Report: {report}")
    typer.echo(f"Include Passing: {include_passing}")
    run = _main_func.obj.get("main")
    if run:
        run(output, log_level, config, report, include_passing)


if __name__ == "__main__":
    app()
