from __future__ import annotations

import asyncio
import base64
from io import BytesIO
from typing import Any, Callable, NamedTuple, Sequence

from result_companion.core.utils.logging_config import get_progress_logger
from result_companion.core.vision.models import EmbeddedImage

logger = get_progress_logger("OCR")

OCR_INSTALL_HINT = "Install result-companion[ocr] to use OCR."


class OcrDependencies(NamedTuple):
    """Runtime OCR dependencies loaded only for OCR runs."""

    engine: Callable[[Any], Any]
    image_module: Any
    numpy_module: Any


async def run_ocr_batch(
    images: Sequence[EmbeddedImage],
    max_per_test: int,
    max_text_length: int,
    concurrency: int,
) -> dict[str, str]:
    """Runs local OCR and returns text by embedded image ID."""
    if not images or max_per_test <= 0:
        return {}

    dependencies = _load_ocr_dependencies()
    selected_images = _limit_images_per_test(images, max_per_test)
    unique_images, image_aliases = _dedupe_images_by_payload(selected_images)
    semaphore = asyncio.Semaphore(max(concurrency, 1))
    tasks = [
        _run_ocr_for_image(image, dependencies, max_text_length, semaphore)
        for image in unique_images
    ]

    pairs = await asyncio.gather(*tasks)
    texts_by_primary_id = {image_id: text for image_id, text in pairs if text}
    return {
        image.id: texts_by_primary_id[image_aliases[image.id]]
        for image in selected_images
        if image_aliases[image.id] in texts_by_primary_id
    }


def _load_ocr_dependencies() -> OcrDependencies:
    """Loads optional OCR dependencies and raises install guidance when missing."""
    try:
        import numpy as np
        from PIL import Image
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise RuntimeError(OCR_INSTALL_HINT) from exc

    return OcrDependencies(engine=RapidOCR(), image_module=Image, numpy_module=np)


def _limit_images_per_test(
    images: Sequence[EmbeddedImage], max_per_test: int
) -> list[EmbeddedImage]:
    """Keeps only first N screenshots for each rendered test identity."""
    counts: dict[tuple[str, ...], int] = {}
    selected: list[EmbeddedImage] = []
    for image in images:
        identity = image.test_identity or (image.test_name,)
        count = counts.get(identity, 0)
        if count >= max_per_test:
            continue
        counts[identity] = count + 1
        selected.append(image)
    return selected


def _dedupe_images_by_payload(
    images: Sequence[EmbeddedImage],
) -> tuple[list[EmbeddedImage], dict[str, str]]:
    """Returns unique image payloads and original-to-primary image ID aliases."""
    images_by_payload: dict[tuple[str, str], EmbeddedImage] = {}
    aliases: dict[str, str] = {}
    for image in images:
        key = (image.mime_type, image.data_base64)
        primary = images_by_payload.setdefault(key, image)
        aliases[image.id] = primary.id
    return list(images_by_payload.values()), aliases


async def _run_ocr_for_image(
    image: EmbeddedImage,
    dependencies: OcrDependencies,
    max_text_length: int,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str]:
    async with semaphore:
        try:
            text = await asyncio.to_thread(
                _read_image_text,
                image,
                dependencies,
                max_text_length,
            )
        except Exception as exc:
            logger.warning(f"OCR failed for embedded image {image.id}: {exc}")
            return image.id, ""
    return image.id, text


def _read_image_text(
    image: EmbeddedImage,
    dependencies: OcrDependencies,
    max_text_length: int,
) -> str:
    """Decodes an embedded image and returns OCR text."""
    image_array = _decode_image(image.data_base64, dependencies)
    result = dependencies.engine(image_array)
    return _truncate_text(_extract_text(result), max_text_length)


def _decode_image(data_base64: str, dependencies: OcrDependencies) -> Any:
    """Decodes base64 image data to an RGB NumPy array."""
    image_bytes = base64.b64decode(data_base64)
    with dependencies.image_module.open(BytesIO(image_bytes)) as image:
        return dependencies.numpy_module.array(image.convert("RGB"))


def _extract_text(result: Any) -> str:
    """Extracts recognized text from common RapidOCR result shapes."""
    text_values = getattr(result, "txts", None)
    if text_values is not None:
        return _join_texts(text_values)

    if isinstance(result, tuple) and result:
        return _extract_text(result[0])

    if isinstance(result, list):
        return _join_texts(_text_from_item(item) for item in result)

    return ""


def _text_from_item(item: Any) -> str:
    """Returns the first string from one OCR result item."""
    if isinstance(item, str):
        return item
    if isinstance(item, (list, tuple)):
        if len(item) > 1 and isinstance(item[1], str):
            return item[1]
        return next((value for value in item if isinstance(value, str)), "")
    return ""


def _join_texts(values: Any) -> str:
    """Joins non-empty OCR text lines."""
    return "\n".join(str(value).strip() for value in values if str(value).strip())


def _truncate_text(text: str, max_text_length: int) -> str:
    """Truncates OCR text to configured maximum length."""
    if max_text_length <= 0:
        return ""
    return text[:max_text_length]
