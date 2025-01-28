import logging
from logging import getLevelName
from typing import Callable

import pytest

from result_companion.parsers.cli_parser import parse_args


@pytest.fixture
def file_exists_true() -> Callable:
    def _file_exists_true(value: str) -> str:
        return value

    return _file_exists_true


def test_required_output_path(file_exists_true) -> None:
    path = "/my/output.xml"
    args = parse_args(file_exists=file_exists_true).parse_args(["-o", path])
    assert args.output == path
    assert args.log_level == getLevelName(logging.INFO)
    assert args.config is None
    assert args.report is None


def test_log_level_default_trace(file_exists_true) -> None:
    path = "/my/output.xml"
    args = parse_args(file_exists=file_exists_true).parse_args(
        ["-o", path, "-l", "DEBUG"]
    )
    assert args.output == path
    assert args.log_level == getLevelName(logging.DEBUG)
    assert args.config is None
    assert args.report is None


def test_config_custom_path(file_exists_true) -> None:
    path = "/my/output.xml"
    config = "/my/config.yaml"
    args = parse_args(file_exists=file_exists_true).parse_args(
        ["-o", path, "-c", config]
    )
    assert args.output == path
    assert args.log_level == getLevelName(logging.INFO)
    assert args.config == config
    assert args.report is None


def test_report_path(file_exists_true) -> None:
    path = "/my/output.xml"
    report = "/my/report.html"
    args = parse_args(file_exists=file_exists_true).parse_args(
        ["-o", path, "-r", report]
    )
    assert args.output == path
    assert args.log_level == getLevelName(logging.INFO)
    assert args.config is None
    assert args.report == report


def test_diff_file_exists(file_exists_true) -> None:
    path = "/my/output.xml"
    diff = "/my/diff.xml"
    args = parse_args(file_exists=file_exists_true).parse_args(["-o", path, "-d", diff])
    assert args.output == path
    assert args.log_level == getLevelName(logging.INFO)
    assert args.config is None
    assert args.diff == diff


def test_log_level_note_existing_error() -> None:
    with pytest.raises(SystemExit):
        parse_args().parse_args(["-o", "output.xml", "-l", "Not Existing"])


def test_config_file_not_existing() -> None:
    with pytest.raises(SystemExit):
        parse_args().parse_args(["-o", "output.xml", "-c", "not_existing.yaml"])
