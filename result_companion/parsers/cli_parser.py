import argparse
import logging
from enum import Enum
from typing import Any, Callable

from result_companion.utils.utils import file_exists


class OutputLogLevel(Enum):
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_str(cls, value) -> Any:
        try:
            return cls[value.upper()]
        except KeyError as err:
            _msg = (
                f"Values available: {[e.name for e in cls]}, while provided: {value!r}"
            )
            raise ValueError(_msg) from err


LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def parse_args(
    custom_msg: str = "", file_exists: Callable = file_exists
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Test Result Companion - CLI\n{custom_msg}"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output.xml file path", type=file_exists
    )
    parser.add_argument(
        "-l",
        "--log-level",
        help=f"Log level verbocity, deafult = '{logging.INFO}' avaiable values: {LOG_LEVELS}",
        default="INFO",
        choices=LOG_LEVELS,
    )
    parser.add_argument(
        "-c",
        "--config",
        required=False,
        help="YAML Config file path",
        default=None,
        type=file_exists,
    )
    parser.add_argument(
        "-r",
        "--report",
        required=False,
        help="Write LLM Report to HTML file",
        default=None,
    )
    parser.add_argument(
        "-d",
        "--diff",
        required=False,
        help="Diff with other XML file",
        type=file_exists,
    )
    parser.add_argument(
        "-i",
        "--include-passing",
        required=False,
        help="Make sure to include PASS test cases!",
        action="store_true",
    )
    return parser
