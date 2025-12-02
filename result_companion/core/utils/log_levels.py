from enum import Enum
from typing import Any


class LogLevels(str, Enum):
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
