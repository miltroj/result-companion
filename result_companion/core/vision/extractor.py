from __future__ import annotations

import re

_EMBEDDED_IMAGE_RE = re.compile(
    r"<img\b[^>]*\bsrc\s*=\s*"
    r"(?P<quote>['\"]?)"
    r"data:(?P<mime>image/[a-z0-9.+-]+);base64,(?P<data>[^'\"\s>]+(?:\s+[^'\"\s>]+)*)"
    r"(?P=quote)"
    r"[^>]*>",
    re.IGNORECASE,
)


def scan_html_images(html_text: str) -> list[tuple[str, str]]:
    """Returns embedded data URI images as ``(mime_type, base64_payload)`` pairs."""
    images: list[tuple[str, str]] = []
    for match in _EMBEDDED_IMAGE_RE.finditer(html_text):
        payload = re.sub(r"\s+", "", match.group("data"))
        images.append((match.group("mime").lower(), payload))
    return images


def strip_html_images(html_text: str) -> str:
    """Removes embedded image tags so base64 payloads do not enter LLM text."""
    return _EMBEDDED_IMAGE_RE.sub("", html_text)
