# Robot Framework Screenshot OCR MVP

## Quick Read

Implement screenshot awareness inside `ContextAwareRobotResults`, not as a second XML parser. Build it in small PRs: first add dependency-free image placeholders and fake OCR text attachment for tests, then add real OCR dependencies and final processing, then align the public `analyze()` API with CLI behavior, then update docs and examples.

Default behavior stays unchanged unless image awareness or OCR is enabled, except embedded data URI `<img>` tags are always stripped from rendered text so base64 never reaches LLM context. OCR remains optional, local, and outside `result_companion/core/chunking/rf_results.py`.

## Table of Contents

- [High-Level Algorithm](#high-level-algorithm)
- [Design Rules](#design-rules)
- [Core Data Model](#core-data-model)
- [Implementation Split](#implementation-split)
- [ContextAwareRobotResults Changes](#contextawarerobotresults-changes)
- [Pre-Refactor Regression Lock](#pre-refactor-regression-lock)
- [Event Walker](#event-walker)
- [Source Hash And Full-Suite Rendering](#source-hash-and-full-suite-rendering)
- [Part 2 OCR Runner](#part-2-ocr-runner)
- [Part 3 Public API Wiring](#part-3-public-api-wiring)
- [Tests](#tests)
- [Historical Note: No Existing Vision Code](#historical-note-no-existing-vision-code)

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
3. Strip embedded data URI `<img>` tags from rendered message text in all modes.
4. When an `html=True` message contains base64 `<img>`, emit `EmbeddedImage` at that exact place if image awareness is enabled.
5. Part 1 renders a small placeholder and can attach fake OCR text in tests through `attach_image_texts()`.
6. Part 2 runs real OCR, attaches OCR text, and then runs normal analysis/chunking.

## Design Rules

- Do not parse `output.xml` a second time for screenshots.
- Do not group OCR by `test_name` only; duplicate test names can exist.
- Do not import RapidOCR, Pillow, NumPy, or ONNX Runtime from `rf_results.py`.
- Do not let base64 image payloads leak into rendered text or chunks.
- Keep image detection cheap. Decode image bytes only in OCR code.
- Always strip embedded data URI `<img>` tags from rendered message text. This is an intentional safety fix, not an OCR feature.
- Keep other default behavior unchanged unless image awareness or OCR is enabled.

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

Pre-work PR:

- Change `source_hash` to stream raw SHA-256 for file-path inputs and keep sanitized rendered fallback for non-file inputs.
- Keep `source_hash` as source identity, not analyzed result identity.
- Add a code TODO near `source_hash` for future `analysis_hash`:

```python
# TODO: Add analysis_hash for analyzed-result-set identity. source_hash only
# tracks raw output.xml identity; analysis_hash should combine source_hash,
# selected tests, tag/pass filters, field exclusions, and vision/OCR config so
# reports from the same source but different analysis scope do not collide.
```

- Update focused `source_hash` tests only. This is independent of screenshot/OCR work.

Part 1 keeps the system light and testable without OCR dependencies:

- Add `EmbeddedImage`, HTML image scanning, event walking, image collection, and placeholder rendering.
- Add `attach_image_texts()` so tests can inject fake OCR text by `EmbeddedImage.id`.
- Add config support for placeholder rendering with `vision.placeholder`.
- Add `vision` merging to `ConfigLoader.load_config()` so user config can enable placeholders.
- Do not add RapidOCR, Pillow, NumPy, ONNX Runtime, `--ocr`, or OCR runner code.
- Do not add a CLI fake OCR flag. Fake text stays in tests or small developer helpers.

Part 2 adds real OCR only after Part 1 works on real Robot examples:

- Add OCR dependencies as optional Poetry extras, not normal install dependencies.
- Add `run_ocr_batch()`.
- Add `vision.ocr`, OCR limits, and CLI `--ocr/--no-ocr`.
- Wire OCR before analysis/chunking.

Part 3 keeps public API behavior aligned with CLI behavior:

- Make `result_companion.api.analyze()` honor `config.vision` when it builds results from a path.
- Keep pre-configured `ContextAwareRobotResults` caller-managed. Do not mutate it beyond existing chunking behavior.
- Reuse the same vision preparation helper as CLI if Part 2 already introduced one.
- Do not add new CLI flags, OCR dependencies, or rendering behavior in this PR.
- Keep the PR small: `api.py` plus focused API tests only, unless shared helper extraction is already needed.

Part 4 documents the OCR path and keeps examples runnable:

- Update user docs with the embedded-base64 requirement: Robot screenshots must use `EMBED` for portable OCR.
- Document that sibling screenshot files and `log.html` parsing are out of MVP scope.
- Add or update a minimal Browser example that uses `run_on_failure=Capture Embedded Screenshot` and `Take Screenshot    EMBED`.
- Show the CLI/config path for placeholders and OCR without duplicating provider setup.
- Keep examples small enough to run locally and use generated artifacts only under ignored harness output directories.

## HTML Image Scanner

Add `result_companion/core/vision/extractor.py` for HTML image scanning only. It must not call `ExecutionResult`.

Expose small helpers:

```python
def scan_html_images(html_text: str) -> list[tuple[str, str]]:
    """Returns (mime_type, base64_payload) for embedded data URI images."""


def strip_html_images(html_text: str) -> str:
    """Removes <img ...> tags so base64 does not enter LLM text."""
```

Use focused regex behavior:

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

## Pre-Refactor Regression Lock

Before replacing render helpers with the event walker, lock high-level behavior through `ContextAwareRobotResults` tests. Use public-ish APIs first, private helper tests second.

Test through:

- `as_texts()` for rendered per-test text.
- `render_chunks()` for chunk text and status.
- `test_names` and `total_test_count` for filter behavior.
- `source_hash` for source identity only.
- `__str__()` for full-suite rendering and base64 stripping fallback.

Cover existing invariants before refactor:

- Suite setup failure collapses skipped tests into one analysis unit.
- Nested suite setup failure keeps unique suite names and one collapsed unit.
- Suite teardown appears after test body, including ancestor teardowns.
- Test setup and teardown render once, not duplicated from body items.
- `exclude_passing()` skips both `PASS` and `SKIP`.
- RF native include/exclude tag filters still apply before rendering.
- Field exclusion still removes requested fields.
- Control structures recurse into child body items.
- `render_chunks()` uses current rendered lines and preserves model status.

Only start walker refactor after those tests pass on current code. After refactor, run the same tests unchanged; any changed expectation must be intentional and documented in the PR.

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
- If stripped message text is empty, skip the normal message line but still emit image events when image rendering is enabled.
- Add one short code comment near the strip call: embedded data URI images are stripped even when placeholders are disabled so base64 never reaches LLM context.
- For each scanned image, emit an `EmbeddedImage` event at the same depth.

For control structures like IF/FOR/TRY:

- Recurse into `.body` like current `_render_body_item()` does.
- Add a path segment from `type`/`name`/index when available.

Preserve existing suite context behavior:

- Failed suite setup still collapses skipped tests into one analysis unit.
- Suite teardown and ancestor teardown context still appear where current tests expect them.
- Keep existing setup/teardown unit tests untouched; the event walker must satisfy them.

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

Message text still goes through `strip_html_images()` in all modes so base64 never enters LLM text.

## Source Hash And Full-Suite Rendering

Keep `source_hash` as source identity. It does not include image placeholders or OCR text.

Future `analysis_hash` should become analyzed result set identity. It should include `source_hash`, selected tests, tag/pass filters, field exclusions, and vision/OCR config. Do not implement it in the OCR MVP; keep the TODO so the distinction is visible in code.

Pragmatic rule:

- If input source is a `Path` or path-like `str`, stream the raw source file through SHA-256.
- If input source is XML bytes/string, `ExecutionResult`, or `TestSuite`, fall back to sanitized full-suite rendering unless the caller supplies source bytes or a source path.
- Store `_source_path` only for real file-path inputs. Do not keep raw XML bytes in memory just for hashing.

Reason:

- Raw file hashing preserves source identity without leaking base64 into rendered text.
- Streaming SHA-256 is cheaper than Robot XML parsing and avoids loading the whole file at once.
- `source_hash` is used to make analyzed result sets unique.
- Image placeholders and OCR text are derived rendering context, not source identity.
- `__str__()` remains the fallback hash source for non-file inputs.

Still strip embedded image tags from `__str__()` output so base64 does not leak if full-suite text is rendered directly.

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
  placeholder: true
```

Part 2 extends it:

```yaml
vision:
  placeholder: true
  ocr: false
  max_screenshots_per_test: 3
  max_text_length: 1500
  concurrency: 2
```

Meaning:

- Embedded data URI `<img>` tags are always stripped, even when `vision.placeholder` is false.
- `vision.placeholder: false` keeps default text output unchanged except for that base64 stripping safety fix.
- `vision.placeholder: true` renders inline screenshot placeholders.
- `vision.ocr: true` implies placeholder rendering and runs OCR.
- CLI `--ocr` sets OCR on for that run.
- Do not add placeholder or fake-OCR CLI flags. Config is enough for placeholder-only mode.
- `ConfigLoader.load_config()` must merge `vision` from user YAML, like `rendering` and `test_filter`.

## Run Flow

In `result_companion/entrypoints/run_rc.py`:

1. Build `ContextAwareRobotResults` with existing tag/field/pass filters.
2. Rendered messages always strip embedded data URI `<img>` tags.
3. If `vision.placeholder` or `vision.ocr`, call `results.include_embedded_images()`.
4. Part 1 stops here. Tests may call `results.attach_image_texts(fake_texts)` directly.
5. In Part 2, if OCR enabled:
   - `images = results.collect_embedded_images()`
   - `texts = await run_ocr_batch(images, ...)`
   - `results.attach_image_texts(texts)`
6. Run analysis and chunking normally.

Dry run:

- Skip OCR.
- Placeholder rendering may still happen if `vision.placeholder` is true.

## Files To Change

| Part | File | Change |
|---|---|---|
| 1 | `result_companion/core/vision/models.py` | Add `EmbeddedImage`. |
| 1 | `result_companion/core/vision/extractor.py` | Add HTML image scan/strip helpers only. No `ExecutionResult` parsing. |
| 1 | `result_companion/core/chunking/rf_results.py` | Add event walker, image collection, placeholders, and fake text attachment. |
| 1 | `result_companion/core/parsers/config.py` | Add `VisionConfigModel` with `placeholder`; merge `vision` in `ConfigLoader.load_config()`. |
| 1 | `result_companion/core/configs/default_config.yaml` | Add `vision.placeholder: true`. |
| 1 | `result_companion/entrypoints/run_rc.py` | Enable placeholders when `vision.placeholder` is true. |
| 2 | `pyproject.toml` | Add OCR dependencies as optional extras. |
| 2 | `result_companion/core/vision/ocr.py` | Add optional OCR runner over `EmbeddedImage`. |
| 2 | `result_companion/core/parsers/config.py` | Add OCR limits. |
| 2 | `result_companion/core/configs/default_config.yaml` | Add OCR config fields. |
| 2 | `result_companion/entrypoints/run_rc.py` | Run OCR before analysis/chunking. |
| 2 | `result_companion/entrypoints/cli/cli_app.py` | Add `--ocr/--no-ocr`. |
| 3 | `result_companion/api.py` | Honor `config.vision` in public `analyze()` path mode. |
| 3 | `tests/unittests/test_api.py` | Cover public API vision preparation without changing caller-managed result objects. |
| 4 | `README.md` | Document experimental Robot screenshot OCR and `EMBED` requirement. |
| 4 | `examples/EXAMPLES.md` | Add concise OCR usage flow and config example. |
| 4 | `fixtures/robot/browser_screenshot_ocr/README.md` | Explain Browser embedded screenshot harness and proof commands. |
| 4 | `fixtures/robot/browser_screenshot_ocr/test_wrong_url_screenshot.robot` | Keep the Browser example self-contained with embedded screenshots. |

## Tests

Part 1 tests:

- Collects one `EmbeddedImage` from `<msg html="true"><img ...>`.
- `EmbeddedImage.keyword_path` points to containing keyword.
- Placeholder appears directly under the screenshot keyword.
- Base64 payload does not appear in rendered text.
- Base64 payload does not appear even when `vision.placeholder` is false.
- Screenshot-only HTML messages do not render empty normal message lines.
- User config with `vision.placeholder: true` enables placeholder rendering.
- Duplicate test names do not collide because image IDs differ.
- Passing tests are skipped when `exclude_passing()` is active.
- Existing suite setup failure and teardown context tests pass unchanged.
- `attach_image_texts({image.id: "Login\nPassword"})` renders `[SCREENSHOT_OCR]` lines next to placeholder.
- Missing OCR text keeps placeholder only.
- Empty OCR text keeps placeholder only.
- `source_hash` stays stable when only attached OCR text changes.
- `source_hash` uses raw file SHA-256 for `Path` input and rendered fallback for non-file input.
- `__str__()` output does not contain embedded base64 image payloads.

Part 2 tests:

- Caps images per test.
- Groups output by `EmbeddedImage.id`.
- Truncates long OCR text.
- Missing RapidOCR extras raise clear install hint.

Entrypoint tests:

- `--no-ocr` does not import RapidOCR.
- `--ocr` collects images and attaches OCR text before analysis.
- `--ocr --dryrun` skips OCR and completes.

Part 3 API tests:

- `analyze(output="output.xml", config=config_with_vision_placeholder)` enables embedded image placeholders before chunking.
- `analyze(output="output.xml", config=config_with_ocr_enabled)` runs OCR preparation when OCR exists and dry run is false.
- `analyze(output="output.xml", config=config_with_ocr_enabled, dryrun=True)` skips OCR.
- `analyze(output=preconfigured_results, config=config_with_vision_placeholder)` leaves caller-managed results unchanged except existing chunking setup.

Part 4 docs/examples checks:

- README shows `vision.placeholder`, `vision.ocr`, and `--ocr` usage in one short flow.
- Browser example uses `EMBED` for both failure hook and teardown screenshots.
- Docs state that file-path screenshots require sidecar files and are not supported by MVP OCR.
- Harness proof commands generate `output.xml`, `log.html`, and `llm_texts.txt` with `[SCREENSHOT]` lines.

## Historical Note: No Existing Vision Code

At plan start, this repo had no committed `core/vision` package yet. Part 1 was treated as new code.

Do not add XML-level screenshot extraction. Put coverage on `ContextAwareRobotResults` collection/rendering tests.

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

Part 4 docs/examples smoke check:

```bash
poetry run robot --outputdir .rc-browser-harness fixtures/robot/browser_screenshot_ocr
poetry run python fixtures/robot/browser_screenshot_ocr/dump_llm_texts.py
```

Expected `.rc-browser-harness/llm_texts.txt` contains `[SCREENSHOT] embedded image/png screenshot`.

Manual check with real embedded screenshots:

```bash
python - <<'PY'
from pathlib import Path

from result_companion.core.chunking.rf_results import ContextAwareRobotResults

results = ContextAwareRobotResults(Path("output.xml")).include_embedded_images()
for name, text in results.as_texts():
    if "[SCREENSHOT]" in text:
        print(name)
        print(text)
PY
```

Confirm `as_texts()` contains screenshot placeholders near the keyword that captured the screenshot, and no base64 payload.

Part 2 manual OCR check:

```bash
result-companion analyze -o output.xml --ocr --debug-log ocr-debug.log
```

Confirm `ocr-debug.log` or an `as_texts()` check contains screenshot placeholders and OCR text near the keyword that captured the screenshot.
