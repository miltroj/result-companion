from __future__ import annotations

from pathlib import Path

from result_companion.core.vision.extractor import extract_screenshots

FIXTURE = Path(__file__).parents[3] / "fixtures" / "vision" / "embedded_screenshot.xml"


def test_extract_screenshots_yields_one_screenshot_for_failing_test() -> None:
    shots = list(extract_screenshots(FIXTURE))
    assert len(shots) == 1


def test_extracted_screenshot_carries_failing_test_name() -> None:
    shots = list(extract_screenshots(FIXTURE))
    assert shots[0].test_name == "Failing Test With Screenshot"


def test_extracted_screenshot_carries_failure_message() -> None:
    shots = list(extract_screenshots(FIXTURE))
    assert shots[0].error_message == "Element not visible"


def test_extracted_screenshot_has_png_mime_type() -> None:
    shots = list(extract_screenshots(FIXTURE))
    assert shots[0].mime_type == "image/png"


def test_extracted_screenshot_base64_has_no_whitespace() -> None:
    shots = list(extract_screenshots(FIXTURE))
    data_base64 = shots[0].data_base64

    assert data_base64
    assert " " not in data_base64
    assert "\n" not in data_base64
    assert "\t" not in data_base64


def test_extract_screenshots_returns_empty_when_no_images(tmp_path: Path) -> None:
    empty_xml = tmp_path / "empty.xml"
    empty_xml.write_text(
        '<?xml version="1.0"?><robot><suite name="S"><test name="T">'
        '<status status="PASS"/></test><status status="PASS"/></suite></robot>'
    )

    assert list(extract_screenshots(empty_xml)) == []
