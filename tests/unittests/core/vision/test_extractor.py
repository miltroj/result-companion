from __future__ import annotations

from pathlib import Path

from result_companion.core.vision.extractor import extract_screenshots

FIXTURE = Path(__file__).parents[3] / "fixtures" / "vision" / "embedded_screenshot.xml"


def _build_rf_xml(tmp_path: Path, suite_body: str) -> Path:
    """Writes a minimal Robot Framework output.xml wrapping ``suite_body``."""
    path = tmp_path / "out.xml"
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<robot generator="Robot 7.0" generated="20260101 12:00:00.000" '
        'rpa="false" schemaversion="4">'
        f"{suite_body}"
        "</robot>"
    )
    return path


def _html_img_msg(data: str = "iVBORw0KGgo=", mime: str = "image/png") -> str:
    """Renders an ``html=true`` <msg> element containing one base64 image tag."""
    return (
        '<msg time="20260101 12:00:00.001" level="INFO" html="true">'
        f'&lt;img src="data:{mime};base64,{data}"&gt;'
        "</msg>"
    )


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


def test_extract_screenshots_yields_one_per_image_in_same_message(
    tmp_path: Path,
) -> None:
    two_images = (
        '<msg time="20260101 12:00:00.001" level="INFO" html="true">'
        '&lt;img src="data:image/png;base64,AAA="&gt;'
        '&lt;img src="data:image/png;base64,BBB="&gt;'
        "</msg>"
    )
    xml = _build_rf_xml(
        tmp_path,
        f'<suite name="S"><test name="T">'
        f'<kw name="Log">{two_images}<status status="PASS"/></kw>'
        f'<status status="FAIL">boom</status></test>'
        f'<status status="FAIL"/></suite>',
    )

    assert len(list(extract_screenshots(xml))) == 2


def test_extract_screenshots_finds_image_in_test_teardown(tmp_path: Path) -> None:
    xml = _build_rf_xml(
        tmp_path,
        f'<suite name="S"><test name="T">'
        f'<kw name="Screenshot On Failure" type="TEARDOWN">'
        f'{_html_img_msg()}<status status="PASS"/></kw>'
        f'<status status="FAIL">boom</status></test>'
        f'<status status="FAIL"/></suite>',
    )

    assert len(list(extract_screenshots(xml))) == 1


def test_extract_screenshots_finds_image_in_nested_keyword(tmp_path: Path) -> None:
    xml = _build_rf_xml(
        tmp_path,
        f'<suite name="S"><test name="T">'
        f'<kw name="Outer"><kw name="Inner">'
        f'{_html_img_msg()}<status status="PASS"/></kw>'
        f'<status status="PASS"/></kw>'
        f'<status status="FAIL">boom</status></test>'
        f'<status status="FAIL"/></suite>',
    )

    assert len(list(extract_screenshots(xml))) == 1


def test_extract_screenshots_traverses_child_suites(tmp_path: Path) -> None:
    xml = _build_rf_xml(
        tmp_path,
        f'<suite name="Parent"><suite name="Child">'
        f'<test name="T"><kw name="Log">{_html_img_msg()}'
        f'<status status="PASS"/></kw>'
        f'<status status="FAIL">boom</status></test>'
        f'<status status="FAIL"/></suite>'
        f'<status status="FAIL"/></suite>',
    )

    assert [s.test_name for s in extract_screenshots(xml)] == ["T"]


def test_extract_screenshots_ignores_message_when_html_flag_is_false(
    tmp_path: Path,
) -> None:
    non_html_msg = (
        '<msg time="20260101 12:00:00.001" level="INFO">'
        '&lt;img src="data:image/png;base64,AAA="&gt;'
        "</msg>"
    )
    xml = _build_rf_xml(
        tmp_path,
        f'<suite name="S"><test name="T">'
        f'<kw name="Log">{non_html_msg}<status status="PASS"/></kw>'
        f'<status status="PASS"/></test>'
        f'<status status="PASS"/></suite>',
    )

    assert list(extract_screenshots(xml)) == []


def test_extract_screenshots_strips_whitespace_from_base64_payload(
    tmp_path: Path,
) -> None:
    wrapped_msg = (
        '<msg time="20260101 12:00:00.001" level="INFO" html="true">'
        '&lt;img src="data:image/png;base64,AAA=&#10;   BBB="&gt;'
        "</msg>"
    )
    xml = _build_rf_xml(
        tmp_path,
        f'<suite name="S"><test name="T">'
        f'<kw name="Log">{wrapped_msg}<status status="PASS"/></kw>'
        f'<status status="FAIL">boom</status></test>'
        f'<status status="FAIL"/></suite>',
    )

    assert list(extract_screenshots(xml))[0].data_base64 == "AAA=BBB="
