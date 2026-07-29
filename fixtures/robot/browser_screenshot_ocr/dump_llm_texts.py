from __future__ import annotations

import sys
from pathlib import Path

from result_companion.core.chunking.rf_results import get_rc_robot_results
from result_companion.core.parsers.config import load_config

HARNESS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_XML = Path(".rc-browser-harness/output.xml")
DEFAULT_TEXT_OUTPUT = Path(".rc-browser-harness/llm_texts.txt")


def main() -> None:
    output_xml = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT_XML
    text_output = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TEXT_OUTPUT
    config = load_config(HARNESS_DIR / "vision_enabled.yaml")

    results = get_rc_robot_results(
        file_path=output_xml,
        include_tags=config.test_filter.include_tags,
        exclude_tags=config.test_filter.exclude_tags,
        exclude_fields=config.rendering.exclude_fields or None,
        exclude_passing=not config.test_filter.include_passing,
    )
    if config.vision.enabled:
        results.include_embedded_images()

    text_output.parent.mkdir(parents=True, exist_ok=True)
    with text_output.open("w", encoding="utf-8") as file:
        for test_name, text in results.as_texts():
            file.write(f"===== TEST: {test_name} =====\n")
            file.write(text)
            file.write("\n\n")

    print(f"Wrote {text_output}")


if __name__ == "__main__":
    main()
