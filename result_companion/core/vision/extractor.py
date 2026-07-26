from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from robot.api import ExecutionResult
from robot.result.model import Message, TestCase, TestSuite

from result_companion.core.vision.models import Screenshot

_IMG_SRC_PATTERN = re.compile(
    r"""<img[^>]+src=["']data:"""
    r"""(?P<mime>image/[a-z0-9+.\-]+);base64,"""
    r"""(?P<data>[A-Za-z0-9+/=\s]+)["']""",
    re.IGNORECASE,
)


def extract_screenshots(output_xml_path: Path) -> Iterator[Screenshot]:
    """Yields screenshots embedded in Robot Framework output.xml.

    Uses Robot Framework's native result parser (`ExecutionResult`) so the
    walker stays robust across RF schema versions and matches the parsing
    strategy already used elsewhere in the project.

    Args:
        output_xml_path: Robot Framework output.xml path.

    Yields:
        Screenshots in per-test document order.
    """
    suite = ExecutionResult(str(output_xml_path)).suite
    yield from _iter_suite(suite)


def _iter_suite(suite: TestSuite) -> Iterator[Screenshot]:
    for test in suite.tests:
        yield from _iter_test(test)
    for child in suite.suites:
        yield from _iter_suite(child)


def _iter_test(test: TestCase) -> Iterator[Screenshot]:
    error_message = (test.message or "").strip()
    for msg in _walk_html_messages(test):
        for mime_type, data_base64 in _scan_msg_for_images(msg.message):
            yield Screenshot(
                test_name=test.name,
                error_message=error_message,
                mime_type=mime_type,
                data_base64=data_base64,
            )


def _walk_html_messages(node: object) -> Iterator[Message]:
    """Recursively yields ``html=True`` messages under a test/keyword/branch.

    Descends into ``setup``/``teardown`` keywords (not present in ``body`` on
    modern RF versions) and any child ``body`` items (keywords, control
    structures like IF/FOR).
    """
    if isinstance(node, Message):
        if node.html:
            yield node
        return
    for attr in ("setup", "teardown"):
        child = getattr(node, attr, None)
        if child:
            yield from _walk_html_messages(child)
    for item in getattr(node, "body", ()) or ():
        yield from _walk_html_messages(item)


def _scan_msg_for_images(html_text: str) -> Iterator[tuple[str, str]]:
    """Yields ``(mime_type, base64_payload)`` for each embedded image tag."""
    for match in _IMG_SRC_PATTERN.finditer(html_text):
        yield match.group("mime"), re.sub(r"\s+", "", match.group("data"))
