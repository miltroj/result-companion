from __future__ import annotations

from result_companion.core.chunking.rf_results import ContextAwareRobotResults
from result_companion.core.parsers.config import DefaultConfigModel
from result_companion.core.utils.logging_config import get_progress_logger
from result_companion.core.vision.ocr import run_ocr_batch

logger = get_progress_logger("OCR")


async def prepare_vision_results(
    results: ContextAwareRobotResults,
    config: DefaultConfigModel,
    dryrun: bool = False,
) -> ContextAwareRobotResults:
    """Applies screenshot placeholder and OCR config before chunking."""
    if config.vision.enabled or config.vision.ocr:
        results.include_embedded_images()

    if config.vision.ocr:
        logger.debug("OCR enabled for embedded screenshots.")

    if dryrun or not config.vision.ocr:
        return results

    images = results.collect_embedded_images()
    if not images:
        return results

    texts = await run_ocr_batch(
        images=images,
        max_per_test=config.vision.max_screenshots_per_test,
        max_text_length=config.vision.max_text_length,
        concurrency=config.vision.concurrency,
    )
    if texts:
        results.attach_image_texts(texts)
    return results
