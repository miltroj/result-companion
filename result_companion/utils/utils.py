import os
import argparse
from enum import Enum


class ExceptionType(Enum):
    ARGPARSE: str = "argparse"
    REGULAR: str = "regular"
    NONE: str = "none"


def file_exists(path: str, throw_exception: ExceptionType = ExceptionType.ARGPARSE) -> str | None:
    if not os.path.isfile(path):
        if throw_exception == ExceptionType.ARGPARSE:
            raise argparse.ArgumentTypeError(f"File {path} does not exist.")
        elif throw_exception == ExceptionType.NONE:
            return 
        else:
            raise FileNotFoundError(f"File {path} does not exist.")
    return path
