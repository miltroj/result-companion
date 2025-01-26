import argparse
from enum import Enum
from typing import Any, Callable
from result_companion.utils.utils import file_exists


class LogLevel(Enum):
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


DEFAULT_LOG_LEVEL = LogLevel.TRACE


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
        help=f"Log level verbocity, deafult = '{DEFAULT_LOG_LEVEL}' avaiable values: {[e.name for e in LogLevel]}",
        type=LogLevel.from_str,
        default=DEFAULT_LOG_LEVEL,
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
    # TODO: remove this, since it should be in config
    parser.add_argument(
        "-lm",
        "--local-model",
        required=False,
        help="Local Ollama model name",
        type=str,
        default="llama3.2",
    )
    return parser
