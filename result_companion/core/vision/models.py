from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Screenshot:
    """One screenshot extracted from a Robot Framework output.xml.

    Attributes:
        test_name: Name from the containing Robot Framework test element.
        error_message: Failure message from the containing test, empty if pass.
        mime_type: e.g. "image/png".
        data_base64: Raw base64 payload with no data-URI prefix.
    """

    test_name: str
    error_message: str
    mime_type: str
    data_base64: str
