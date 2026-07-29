from result_companion.core.vision.extractor import scan_html_images, strip_html_images


def test_scan_html_images_returns_embedded_data_images():
    html = '<p>x</p><img src="data:image/png;base64,abc123">'

    assert scan_html_images(html) == [("image/png", "abc123")]


def test_scan_html_images_handles_case_and_base64_whitespace():
    html = "<IMG SRC='data:image/JPEG;base64,abc 123\nxyz'>"

    assert scan_html_images(html) == [("image/jpeg", "abc123xyz")]


def test_scan_html_images_ignores_file_links():
    html = '<img src="screenshots/login.png">'

    assert scan_html_images(html) == []


def test_strip_html_images_removes_image_tag_and_keeps_text():
    html = 'before <img alt="login" src="data:image/png;base64,abc123"> after'

    assert strip_html_images(html) == "before  after"
