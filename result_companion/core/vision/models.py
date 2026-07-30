from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddedImage:
    """Embedded screenshot found while rendering Robot Framework results."""

    id: str
    test_name: str
    keyword_path: tuple[str, ...]
    message_index: int
    image_index: int
    ordinal: int
    mime_type: str
    data_base64: str
    test_identity: tuple[str, ...] = ()

    def placeholder(self) -> str:
        """Returns the inline screenshot placeholder text."""
        return f"[SCREENSHOT] embedded {self.mime_type} screenshot #{self.ordinal}"
