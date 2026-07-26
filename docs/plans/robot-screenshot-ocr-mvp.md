# Robot Framework Screenshot OCR MVP

## Quick Read

Implement screenshot awareness inside `ContextAwareRobotResults`, not as a second XML parser. Build it in two parts: first add dependency-free image placeholders and fake OCR text attachment for tests, then add real OCR dependencies and final processing.

Default behavior stays unchanged unless image awareness or OCR is enabled. OCR remains optional, local, and outside `result_companion/core/chunking/rf_results.py`.

## Table of Contents

- [High-Level Algorithm](#high-level-algorithm)
- [Design Rules](#design-rules)
- [Core Data Model](#core-data-model)
- [Implementation Split](#implementation-split)
- [ContextAwareRobotResults Changes](#contextawarerobotresults-changes)
- [Event Walker](#event-walker)
- [Part 2 OCR Runner](#part-2-ocr-runner)
- [Tests](#tests)
- [Migration From Current WIP](#migration-from-current-wip)

## High-Level Algorithm

Robot result tree already has the location we need:

```text
Suite: Web
  Test: Login fails
    Keyword: Open Login Page
      [INFO] page opened
    Keyword: Capture Page Screenshot
      [INFO html=true] <img src="data:image/png;base64,...">
    Keyword: Click Submit
      [ERROR] Element not visible
```

Renderer should produce events in the same order:

```text
RenderLine("Suite: Web")
RenderLine("Test: Login fails")
RenderLine("Keyword: Open Login Page")
RenderLine("[INFO] page opened")
RenderLine("Keyword: Capture Page Screenshot")
EmbeddedImage(test="Login fails", keyword_path=("Capture Page Screenshot",), image_index=0)
RenderLine("Keyword: Click Submit")
RenderLine("[ERROR] Element not visible")
```

Text without OCR, but with image awareness:

```text
Keyword: Capture Page Screenshot
  [SCREENSHOT] embedded image/png screenshot #1
```

Text with OCR attached:

```text
Keyword: Capture Page Screenshot
  [SCREENSHOT] embedded image/png screenshot #1
  [SCREENSHOT_OCR] Login
  [SCREENSHOT_OCR] Password
```

Core flow:

1. Parse `output.xml` once through `ExecutionResult` in `ContextAwareRobotResults`.
2. Walk suites, tests, keywords, messages with a shared event walker.
3. When an `html=True` message contains base64 `<img>`, emit `EmbeddedImage` at that exact place.
4. Part 1 renders a small placeholder and can attach fake OCR text in tests through `attach_image_texts()`.
5. Part 2 runs real OCR, attaches OCR text, and then runs normal analysis/chunking.

## Design Rules

- Do not parse `output.xml` a second time for screenshots.
- Do not group OCR by `test_name` only; duplicate test names can exist.
- Do not import RapidOCR, Pillow, NumPy, or ONNX Runtime from `rf_results.py`.
- Do not let base64 image payloads leak into rendered text or chunks.
- Keep image detection cheap. Decode image bytes only in OCR code.
- Keep default behavior unchanged unless image awareness or OCR is enabled.

## Core Data Model

Add to `result_companion/core/vision/models.py`:

```python
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

    def placeholder(self) -> str:
        return f"[SCREENSHOT] embedded {self.mime_type} screenshot #{self.ordinal}"
```

`id` is a stable key for OCR results. Generate it from:

```text
suite path + test name + keyword path + message index + image index + sha256(data_base64)[:12]
```

Hash is not the correlation mechanism. Correlation comes from tree traversal context. Hash only lets OCR return `dict[str, str]` safely.

## Implementation Split

Part 1 keeps the system light and testable without OCR dependencies:

- Add `EmbeddedImage`, HTML image scanning, event walking, image collection, and placeholder rendering.
- Add `attach_image_texts()` so tests can inject fake OCR text by `EmbeddedImage.id`.
- Add config support for placeholder rendering with `vision.enabled`.
- Do not add RapidOCR, Pillow, NumPy, ONNX Runtime, `--ocr`, or OCR runner code.
- Do not add a CLI fake OCR flag. Fake text stays in tests or small developer helpers.

Part 2 adds real OCR only after Part 1 works on real Robot examples:

- Add Poetry dependencies for OCR.
- Add `run_ocr_batch()`.
- Add `vision.ocr`, OCR limits, and CLI `--ocr/--no-ocr`.
- Wire OCR before analysis/chunking.

## HTML Image Scanner

Keep `result_companion/core/vision/extractor.py`, but reduce its job. It must not call `ExecutionResult`.

Expose small helpers:

```python
def scan_html_images(html_text: str) -> list[tuple[str, str]]:
    """Returns (mime_type, base64_payload) for embedded data URI images."""


def strip_html_images(html_text: str) -> str:
    """Removes <img ...> tags so base64 does not enter LLM text."""
```

Reuse existing regex behavior:

- Case-insensitive `<img>` and `src` handling.
- Accept whitespace in base64 and strip it.
- Only support embedded `data:image/...;base64,...` images.
- Ignore file links from `log.html` for MVP.

## `ContextAwareRobotResults` Changes

Add state:

```python
self._include_images: bool = False
self._image_texts: dict[str, str] = {}
```

Add public methods:

```python
def include_embedded_images(self, include: bool = True) -> ContextAwareRobotResults:
    """Renders embedded screenshot placeholders inline."""


def collect_embedded_images(self) -> list[EmbeddedImage]:
    """Returns embedded images from currently selected tests."""


def attach_image_texts(self, texts: dict[str, str]) -> ContextAwareRobotResults:
    """Attaches OCR text by EmbeddedImage.id and enables image rendering."""
```

Behavior:

- `include_embedded_images()` enables placeholder rendering.
- `collect_embedded_images()` respects tag filters and `exclude_passing()`.
- `attach_image_texts()` enables image rendering and invalidates caches.
- `set_chunking()` must run after image texts are attached, or `_iter_tests()` must always render latest image texts before chunking.

## Event Walker

Use one internal walker as the source of truth. Avoid separate collect/render traversals with duplicated path logic.

Concept:

```python
RenderEvent = RenderLine | EmbeddedImage


def _iter_test_events(test, depth, fields, context) -> Iterator[RenderEvent]:
    yield RenderLine(depth, "Test: ...")
    ...
    for body_item in test.body:
        yield from _iter_body_events(body_item, depth + 1, fields, context)
```

Context tracks:

```python
@dataclass
class RenderContext:
    suite_path: tuple[str, ...]
    test_name: str
    keyword_path: tuple[str, ...]
    message_index: int
    image_ordinal: int
```

When visiting a keyword:

- Push keyword name before body traversal.
- Pop after traversal.
- Include keyword index if same keyword name repeats under the same parent.

When visiting a message:

- Increment `message_index` in traversal order.
- Render normal message text after stripping image tags.
- For each scanned image, emit an `EmbeddedImage` event at the same depth.

For control structures like IF/FOR/TRY:

- Recurse into `.body` like current `_render_body_item()` does.
- Add a path segment from `type`/`name`/index when available.

## Rendering Image Events

Convert `EmbeddedImage` to render lines inside `_iter_tests()` or a small helper:

```python
def _render_image_event(image: EmbeddedImage, image_texts: dict[str, str]) -> list[RenderLine]:
    lines = [RenderLine(1, image.placeholder())]
    text = image_texts.get(image.id, "")
    lines.extend(
        RenderLine(1, f"[SCREENSHOT_OCR] {line}")
        for line in text.splitlines()
        if line.strip()
    )
    return lines
```

Use the real event depth, not hardcoded `1`, in implementation.

If image awareness is disabled and no OCR text is attached, skip `EmbeddedImage` events entirely. This preserves old output.

## Part 2 OCR Runner

Add `result_companion/core/vision/ocr.py`:

```python
async def run_ocr_batch(
    images: Sequence[EmbeddedImage],
    max_per_test: int,
    max_text_length: int,
    concurrency: int,
) -> dict[str, str]:
    """Runs local OCR and returns OCR text by EmbeddedImage.id."""
```

Rules:

- Import RapidOCR, Pillow, and NumPy inside OCR code only.
- Decode base64 inside OCR code only.
- Cap screenshots per test before OCR.
- Truncate text per image or per test according to config.
- Return no key for empty OCR output.
- Log OCR failures and continue with placeholders.

## Config And CLI

Part 1 adds placeholder config only:

```yaml
vision:
  enabled: false
```

Part 2 extends it:

```yaml
vision:
  enabled: false
  ocr: false
  max_screenshots_per_test: 3
  max_text_length: 1500
  concurrency: 2
```

Meaning:

- `vision.enabled: true` renders inline screenshot placeholders.
- `vision.ocr: true` implies `vision.enabled: true` and runs OCR.
- CLI `--ocr` sets OCR on for that run.
- Do not add placeholder or fake-OCR CLI flags. Config is enough for placeholder-only mode.

## Run Flow

In `result_companion/entrypoints/run_rc.py`:

1. Build `ContextAwareRobotResults` with existing tag/field/pass filters.
2. If `vision.enabled` or `vision.ocr`, call `results.include_embedded_images()`.
3. Part 1 stops here. Tests may call `results.attach_image_texts(fake_texts)` directly.
4. In Part 2, if OCR enabled:
   - `images = results.collect_embedded_images()`
   - `texts = await run_ocr_batch(images, ...)`
   - `results.attach_image_texts(texts)`
5. Run analysis and chunking normally.

Dry run:

- Skip OCR.
- Placeholder rendering may still happen if `vision.enabled` is true.

## Files To Change

| Part | File | Change |
|---|---|---|
| 1 | `result_companion/core/vision/models.py` | Add `EmbeddedImage`. |
| 1 | `result_companion/core/vision/extractor.py` | Keep only HTML image scan/strip helpers. Remove `ExecutionResult` parsing. |
| 1 | `result_companion/core/chunking/rf_results.py` | Add event walker, image collection, placeholders, and fake text attachment. |
| 1 | `result_companion/core/parsers/config.py` | Add `VisionConfigModel` with `enabled`. |
| 1 | `result_companion/core/configs/default_config.yaml` | Add `vision.enabled: false`. |
| 1 | `result_companion/entrypoints/run_rc.py` | Enable placeholders when `vision.enabled` is true. |
| 2 | `pyproject.toml` | Add OCR dependencies. |
| 2 | `result_companion/core/vision/ocr.py` | Add optional OCR runner over `EmbeddedImage`. |
| 2 | `result_companion/core/parsers/config.py` | Add OCR limits. |
| 2 | `result_companion/core/configs/default_config.yaml` | Add OCR config fields. |
| 2 | `result_companion/entrypoints/run_rc.py` | Run OCR before analysis/chunking. |
| 2 | `result_companion/entrypoints/cli/cli_app.py` | Add `--ocr/--no-ocr`. |
| 2 | `README.md` | Document experimental screenshot OCR. |

## Tests

Part 1 tests:

- Collects one `EmbeddedImage` from `<msg html="true"><img ...>`.
- `EmbeddedImage.keyword_path` points to containing keyword.
- Placeholder appears directly under the screenshot keyword.
- Base64 payload does not appear in rendered text.
- Duplicate test names do not collide because image IDs differ.
- Passing tests are skipped when `exclude_passing()` is active.
- `attach_image_texts({image.id: "Login\nPassword"})` renders `[SCREENSHOT_OCR]` lines next to placeholder.
- Missing OCR text keeps placeholder only.
- Empty OCR text keeps placeholder only.

Part 2 tests:

- Caps images per test.
- Groups output by `EmbeddedImage.id`.
- Truncates long OCR text.
- Missing RapidOCR extras raise clear install hint.

Entrypoint tests:

- `--no-ocr` does not import RapidOCR.
- `--ocr` collects images and attaches OCR text before analysis.
- `--ocr --dryrun` skips OCR and completes.

## Migration From Current WIP

Current `extract_screenshots(output_xml_path)` is the wrong shape for this plan because it parses `ExecutionResult` separately and returns only `test_name`.

Replace it with HTML scanning helpers. Move test coverage from XML-level extraction to `ContextAwareRobotResults` collection/rendering tests.

## Do Not Do In MVP

- No second XML parse for screenshots.
- No multimodal LLM path.
- No `log.html` file screenshot support.
- No OCR cache.
- No PII redaction.
- No visual regression analysis.
- No global RapidOCR engine cache.
- No new abstraction unless it removes duplicated traversal logic.

## Validation

Part 1 focused tests:

```bash
poetry run pytest tests/unittests/core/vision/ tests/unittests/core/chunking/test_rf_results.py -v
```

Run all unit tests:

```bash
make test-unit
```

Manual check with real embedded screenshots:

```bash
result-companion analyze -o output.xml -c vision-enabled.yaml --debug-log vision-debug.log
```

Confirm `rc_log.html` contains screenshot placeholders near the keyword that captured the screenshot.

Part 2 manual OCR check:

```bash
result-companion analyze -o output.xml --ocr --debug-log ocr-debug.log
```

Confirm `rc_log.html` contains screenshot placeholders and OCR text near the keyword that captured the screenshot.
