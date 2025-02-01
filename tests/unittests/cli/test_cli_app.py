from pathlib import Path

from typer import echo
from typer.testing import CliRunner

from result_companion.cli.cli_app import app

runner = CliRunner()
existing_xml_path = Path(__file__).parent / "empty.xml"


def test_cli_fail_when_outpu_not_exists():
    result = runner.invoke(app, ["-o", "not_exists.xml"], obj={})
    assert result.exit_code == 2
    assert "File 'not_exists.xml' does not exist" in result.output


def test_cli_fail_when_config_not_exists():
    result = runner.invoke(
        app, ["-o", existing_xml_path, "-c", "config_not_exists"], obj={}
    )
    assert result.exit_code == 2
    assert "File 'config_not_exists' does not exist" in result.output


def test_cli_by_default_uses_include_passing_false():
    result = runner.invoke(app, ["-o", existing_xml_path], obj={})
    assert result.exit_code == 0
    assert "Include Passing: False" in result.output


def test_cli_sets_include_passing():
    result = runner.invoke(app, ["-o", existing_xml_path, "-i"], obj={})
    assert result.exit_code == 0
    assert "Include Passing: True" in result.output


def test_cli_stets_generating_report():
    result = runner.invoke(app, ["-o", existing_xml_path, "-r", "report.html"], obj={})
    assert result.exit_code == 0
    assert "Report: report.html" in result.output


def test_cli_sets_info_as_default_log_level():
    result = runner.invoke(app, ["-o", existing_xml_path], obj={})
    assert result.exit_code == 0
    assert "Log Level: INFO" in result.output


def test_cli_calls_main_function():
    result = runner.invoke(
        app, ["-o", existing_xml_path], obj={"main": lambda *args: echo("RUNNING MAIN")}
    )
    assert result.exit_code == 0
    assert "Output: " in result.output
    assert "Log Level: " in result.output
    assert "Config: " in result.output
    assert "Report: " in result.output
    assert "Diff: " in result.output
    assert "Include Passing: " in result.output
    assert "RUNNING MAIN" in result.output
