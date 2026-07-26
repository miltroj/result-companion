from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

from result_companion.core.vision.models import Screenshot

_IMG_SRC_PATTERN = re.compile(
    r"""<img[^>]+src=["']data:"""
    r"""(?P<mime>image/[a-z0-9+.\-]+);base64,"""
    r"""(?P<data>[A-Za-z0-9+/=\s]+)["']""",
    re.IGNORECASE,
)


def extract_screenshots(output_xml_path: Path) -> Iterator[Screenshot]:
    """Yields screenshots embedded in Robot Framework output.xml.

    Args:
        output_xml_path: Robot Framework output.xml path.

    Yields:
        Screenshots in per-test document order.
    """
    current_test: str | None = None
    current_test_message = ""
    pending: list[tuple[str, str]] = []

    for event, element in ET.iterparse(output_xml_path, events=("start", "end")):
        tag = _local_name(element.tag)

        if event == "start" and tag == "test":
            current_test = element.attrib.get("name", "<unnamed>")
            current_test_message = ""
            pending = []
            continue

        if event != "end":
            continue

        if tag == "msg" and element.attrib.get("html") == "true" and current_test:
            pending.extend(_scan_msg_for_images(element.text or ""))

        if tag == "status" and current_test:
            current_test_message = (element.text or "").strip()

        if tag == "test":
            yield from _build_screenshots(
                current_test or "", current_test_message, pending
            )
            current_test = None
            current_test_message = ""
            pending = []

        element.clear()


def _scan_msg_for_images(html_text: str) -> Iterator[tuple[str, str]]:
    """Yields image MIME type and base64 payload from an HTML message."""
    for match in _IMG_SRC_PATTERN.finditer(html_text):
        yield match.group("mime"), re.sub(r"\s+", "", match.group("data"))


def _build_screenshots(
    test_name: str,
    error_message: str,
    pending: list[tuple[str, str]],
) -> Iterator[Screenshot]:
    """Builds screenshot models after containing test status is known."""
    for mime_type, data_base64 in pending:
        yield Screenshot(
            test_name=test_name,
            error_message=error_message,
            mime_type=mime_type,
            data_base64=data_base64,
        )


def _local_name(tag: str) -> str:
    """Strips XML namespace from a tag name."""
    return tag.rsplit("}", 1)[-1]
