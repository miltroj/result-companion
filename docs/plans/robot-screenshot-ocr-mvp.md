# Robot Framework Screenshot OCR — MVP Plan

Status: MVP proof-of-concept. Ships as **opt-in experimental** feature. Off by default. Requires `pip install result-companion[vision]`.

Supersedes: [`robot-vision-plan.md`](robot-vision-plan.md). Vision-via-multimodal-LLM approach abandoned in favor of OCR-based text extraction. Rationale: single provider for main analysis, no Copilot adapter changes, zero data egress from screenshots, works with any existing user LLM setup.

## What This Ships

Extract embedded base64 screenshots from Robot Framework `output.xml`. Run OCR locally (RapidOCR + onnxruntime). Append extracted text to per-test rendered content so the existing text-analysis LLM (whatever the user has configured) can reason about screenshot content alongside logs. No new LLM calls, no new provider config, no vision-model policy management.

## Non-Goals

- No multimodal LLM path. If added later, it is a separate PR.
- No inline positioning of OCR text at the exact `<msg>` location. OCR text is appended per-test as a marked section. Locality-in-order is a future refinement.
- No PII redaction of OCR output. Test authors are responsible for screenshot content.
- No visual regression detection (colors, layout, icons). OCR is text-only.
- No caching of OCR results across runs.
- No support for non-Robot plugins. RC currently only has Robot; if a second plugin lands, revisit.
- No fallback to `log.html` screenshots. Only `output.xml` embedded screenshots.
- No CLI progress bar for OCR. Standard `logger.info` messages only.

## Prerequisites

- Repo uses Poetry (`pyproject.toml`). Extras go under `[tool.poetry.extras]`.
- Python 3.11+, matching existing `pyproject.toml` constraint.
- `RapidOCR` v3.9.0+ (models bundled in wheel — no first-run download needed).
- `onnxruntime` 1.17+.

Install size added when user opts in: ~310 MB (rapidocr ~29 MB + onnxruntime ~180 MB + opencv-python + numpy + Pillow + Shapely + pyclipper). Users who don't install `[vision]` pay zero.

## Architecture

Data flow when `--ocr` is enabled:

1. `_main()` calls existing `get_rc_robot_results(output)` to parse RF results.
2. If OCR enabled: `extract_screenshots(output)` iterparses `output.xml`, yielding one `Screenshot` per embedded `<img src="data:image/...;base64,...">` inside `<msg html="true">`.
3. `run_ocr_batch(screenshots)` decodes base64, runs RapidOCR per screenshot, concurrent-bounded.
4. Results grouped by `test_name` → `dict[str, str]` (concatenated OCR text per test).
5. `results.append_ocr_context(ocr_map)` attaches per-test OCR text to `ContextAwareRobotResults` before analysis.
6. Existing analysis pipeline runs unchanged. Chunker sees enriched per-test text. LLM analyzes.
7. Existing rendering (HTML, text, JSON) unchanged. OCR text flows through as part of test content.

Key insight: **OCR text is treated as extra log lines appended to each test**. Each line is prefixed `[SCREENSHOT_OCR]` to match RC's existing `[LEVEL] message` message-render convention (`[INFO]`, `[ERROR]`, ...). Signals origin to both the LLM and human reviewers. Zero changes downstream of `ContextAwareRobotResults._iter_tests`.

## Files Touched

| File | Change | Milestone |
|---|---|---|
| `pyproject.toml` | Add `[tool.poetry.extras] vision = [...]` group | M1 |
| `result_companion/core/vision/__init__.py` | New, empty | M1 |
| `result_companion/core/vision/models.py` | New — `Screenshot` dataclass | M1 |
| `result_companion/core/vision/extractor.py` | New — `extract_screenshots()` | M1 |
| `tests/unittests/core/vision/__init__.py` | New, empty | M1 |
| `tests/unittests/core/vision/test_extractor.py` | New — 5 tests | M1 |
| `tests/fixtures/vision/embedded_screenshot.xml` | New — RF output fixture with one FAIL test containing one embedded PNG | M1 |
| `result_companion/core/vision/ocr.py` | New — `run_ocr_batch()` | M2 |
| `result_companion/core/parsers/config.py` | Add `VisionConfigModel`, wire into `DefaultConfigModel`, extend merge dict | M2 |
| `result_companion/core/configs/default_config.yaml` | Append `vision:` section (disabled) | M2 |
| `result_companion/core/chunking/rf_results.py` | Add `append_ocr_context()` method, wire into `_iter_tests` | M2 |
| `result_companion/entrypoints/run_rc.py` | Add `ocr` param, thread through, call extractor + OCR when enabled | M2 |
| `result_companion/entrypoints/cli/cli_app.py` | Add `--ocr/--no-ocr` flag on `analyze` command | M2 |
| `tests/unittests/core/vision/test_ocr.py` | New — OCR wrapper tests (RapidOCR mocked) | M2 |
| `tests/unittests/core/vision/test_config.py` | New — config validation test | M2 |
| `tests/unittests/core/chunking/test_rf_results.py` | Add `test_append_ocr_context_appends_lines` | M2 |
| `tests/unittests/entrypoints/test_run_rc_ocr.py` | New — integration test with mocked OCR | M2 |
| `README.md` | Add "Screenshot OCR (experimental)" section | M2 |

Total: 2 PRs, ~9 commits.

---

# Milestone M1 — Screenshot Extraction

## Purpose

Isolate XML parsing. No OCR, no LLM. Extract base64 screenshots from `output.xml` and return them with their `test_name` + failure `error_message`. Testable with a small fixture.

## Commit M1.1 — Extras group in `pyproject.toml`

**File**: `pyproject.toml`

Add under `[tool.poetry.dependencies]` (declare optional deps):

```toml
rapidocr = { version = ">=3.9.0", optional = true }
onnxruntime = { version = ">=1.17,<2.0", optional = true }
```

Add new section (after `[tool.poetry.dependencies]`):

```toml
[tool.poetry.extras]
vision = ["rapidocr", "onnxruntime"]
```

**Note for implementer**: RapidOCR pulls `opencv-python` transitively (~90 MB). Do NOT try to swap for `opencv-python-headless` in this PR — it fights with RapidOCR's dep constraints. Ship as-is; optimize later if needed.

**Verification**:

```bash
poetry lock --no-update && poetry install --extras vision
python -c "from rapidocr import RapidOCR; print(RapidOCR)"
```

## Commit M1.2 — `Screenshot` dataclass

**File**: `result_companion/core/vision/__init__.py`

Empty file.

**File**: `result_companion/core/vision/models.py`

Complete file contents:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Screenshot:
    """One screenshot extracted from a Robot Framework output.xml.

    Attributes:
        test_name: Full test name including suite path.
        error_message: Failure message from the containing test, empty if pass.
        mime_type: e.g. "image/png".
        data_base64: Raw base64 payload (no data-URI prefix, whitespace stripped).
    """

    test_name: str
    error_message: str
    mime_type: str
    data_base64: str
```

**Do NOT add**: `breadcrumb`, `data_uri` property, `__post_init__`, or any other field. MVP wants the minimum surface. Extras are speculative and get in the way.

## Commit M1.3 — Extractor implementation

**File**: `result_companion/core/vision/extractor.py`

Complete file contents:

```python
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

from result_companion.core.vision.models import Screenshot

_IMG_SRC_PATTERN = re.compile(
    r"""<img[^>]+src=["']data:(?P<mime>image/[a-z0-9+.\-]+);base64,(?P<data>[A-Za-z0-9+/=\s]+)["']""",
    re.IGNORECASE,
)


def extract_screenshots(output_xml_path: Path) -> Iterator[Screenshot]:
    """Yields one Screenshot per embedded base64 image found in output.xml.

    Buffers per-test screenshots and flushes at </test>, so that error_message
    (which appears in <status> AFTER all <kw> children in RF output.xml) is
    populated correctly on every screenshot for that test.

    Args:
        output_xml_path: Path to a Robot Framework output.xml file.

    Yields:
        Screenshot instances in per-test document order.
    """
    context = ET.iterparse(output_xml_path, events=("start", "end"))
    current_test: str | None = None
    current_test_message: str = ""
    pending: list[tuple[str, str]] = []

    for event, element in context:
        tag = _local_name(element.tag)

        if event == "start" and tag == "test":
            current_test = element.attrib.get("name", "<unnamed>")
            current_test_message = ""
            pending = []
            continue

        if event != "end":
            continue

        if tag == "msg" and element.attrib.get("html") == "true" and current_test is not None:
            for mime, data in _scan_msg_for_images(element.text or ""):
                pending.append((mime, data))

        elif tag == "status" and current_test is not None:
            current_test_message = (element.text or "").strip()

        elif tag == "test":
            for mime, data in pending:
                yield Screenshot(
                    test_name=current_test or "",
                    error_message=current_test_message,
                    mime_type=mime,
                    data_base64=data,
                )
            current_test = None
            current_test_message = ""
            pending = []

        element.clear()


def _scan_msg_for_images(html_text: str) -> Iterator[tuple[str, str]]:
    """Yields (mime, base64_data) for every embedded <img> in the HTML message."""
    for match in _IMG_SRC_PATTERN.finditer(html_text):
        mime = match.group("mime")
        data = re.sub(r"\s+", "", match.group("data"))
        yield mime, data


def _local_name(tag: str) -> str:
    """Strips XML namespace from a tag name."""
    return tag.rsplit("}", 1)[-1]
```

**Gotchas the implementer must not "fix"**:

- `element.clear()` runs on ALL end events (including `<test>`). Do not restrict it. RF output.xml can be hundreds of MB; clearing frees memory as we go.
- Do NOT try to yield inside the `<msg>` end branch. `<status>` appears AFTER `<kw>` in RF output.xml, so `current_test_message` is unset there. The pending-buffer approach is required.
- Do NOT decode the base64 payload here. OCR module decodes.
- The regex is case-insensitive intentionally. RF SeleniumLibrary uses lowercase `src=`, but Browser library variants may differ.

## Commit M1.4 — Fixture

**File**: `tests/fixtures/vision/embedded_screenshot.xml`

Complete file contents (a hand-crafted RF output.xml with one PASS test and one FAIL test containing one embedded 1x1 red PNG):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!-- Fixture: minimal RF output with one embedded screenshot for OCR extractor tests. -->
<robot generator="Robot 7.0" generated="20260101 12:00:00.000" rpa="false" schemaversion="4">
  <suite name="OcrFixture" source="/dev/null">
    <test name="Passing Test">
      <kw name="Log" library="BuiltIn">
        <arg>hello</arg>
        <msg time="20260101 12:00:00.001" level="INFO">hello</msg>
        <status status="PASS" starttime="20260101 12:00:00.000" endtime="20260101 12:00:00.001"/>
      </kw>
      <status status="PASS" starttime="20260101 12:00:00.000" endtime="20260101 12:00:00.001"/>
    </test>
    <test name="Failing Test With Screenshot">
      <kw name="Capture Page Screenshot" library="SeleniumLibrary">
        <msg time="20260101 12:00:00.002" level="INFO" html="true">&lt;img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="&gt;</msg>
        <status status="PASS" starttime="20260101 12:00:00.001" endtime="20260101 12:00:00.002"/>
      </kw>
      <status status="FAIL" starttime="20260101 12:00:00.000" endtime="20260101 12:00:00.002">Element not visible</status>
    </test>
    <status status="FAIL" starttime="20260101 12:00:00.000" endtime="20260101 12:00:00.002"/>
  </suite>
</robot>
```

**Note**: the `<img ...>` inside `<msg html="true">` is HTML-entity-encoded (`&lt;` for `<`). RF writes it this way. Extractor reads `element.text` which returns the decoded string. Regex operates on the decoded string. The base64 payload is a real 1x1 red PNG; RapidOCR will find no text in it (that's fine — M1 doesn't OCR).

## Commit M1.5 — Extractor tests

**File**: `tests/unittests/core/vision/__init__.py`

Empty file.

**File**: `tests/unittests/core/vision/test_extractor.py`

Complete file contents:

```python
from pathlib import Path

import pytest

from result_companion.core.vision.extractor import extract_screenshots
from result_companion.core.vision.models import Screenshot

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
    data = shots[0].data_base64
    assert data
    assert " " not in data and "\n" not in data and "\t" not in data


def test_extract_screenshots_returns_empty_when_no_images(tmp_path: Path) -> None:
    empty_xml = tmp_path / "empty.xml"
    empty_xml.write_text(
        '<?xml version="1.0"?><robot><suite name="S"><test name="T">'
        '<status status="PASS"/></test><status status="PASS"/></suite></robot>'
    )
    assert list(extract_screenshots(empty_xml)) == []
```

**Do NOT add**: memory profiling tests, breadcrumb tests, multi-screenshot-per-test tests. Out of MVP scope. If they're needed, they land in follow-up PRs.

Run tests:

```bash
poetry run pytest tests/unittests/core/vision/ -v
```

Expected: 6 passed.

## M1 Acceptance

- [ ] All 6 tests pass.
- [ ] `poetry install --extras vision` succeeds locally.
- [ ] `poetry install` (without extras) succeeds and does NOT pull rapidocr/onnxruntime.
- [ ] No changes to any file outside those listed under M1.

---

# Milestone M2 — OCR + Injection + CLI

## Purpose

Wire the extractor to RapidOCR. Merge OCR text into per-test rendered content. Expose the whole path via `--ocr` CLI flag. Update docs. Ship it.

## Commit M2.1 — `VisionConfigModel` + default YAML

**File**: `result_companion/core/parsers/config.py`

Add (place after `RenderingModel` class, before `DefaultConfigModel`):

```python
class VisionConfigModel(BaseModel):
    """Experimental OCR-based screenshot enrichment (requires [vision] extras)."""

    enabled: bool = Field(default=False, description="Enable OCR of embedded screenshots.")
    max_screenshots_per_test: int = Field(
        default=3, ge=1, description="Cap screenshots processed per test to bound cost."
    )
    max_text_length: int = Field(
        default=1500, ge=100, description="Truncate OCR text per test to bound tokens."
    )
    concurrency: int = Field(
        default=2, ge=1, description="Concurrent OCR workers (CPU-bound)."
    )
```

Modify `DefaultConfigModel` to add:

```python
    vision: VisionConfigModel = Field(default_factory=VisionConfigModel)
```

Modify `ConfigLoader.load_config` — inside the `config_data` dict literal (search for `"rendering":`, add sibling entry):

```python
                "vision": {
                    **default_config.get("vision", {}),
                    **user_config.get("vision", {}),
                },
```

**File**: `result_companion/core/configs/default_config.yaml`

Append at end of file:

```yaml
vision:
  enabled: false
  max_screenshots_per_test: 3
  max_text_length: 1500
  concurrency: 2
```

**Test file**: `tests/unittests/core/vision/test_config.py`

Complete file contents:

```python
from result_companion.core.parsers.config import load_config


def test_default_config_has_vision_disabled() -> None:
    config = load_config(None)
    assert config.vision.enabled is False
    assert config.vision.max_screenshots_per_test == 3
    assert config.vision.concurrency == 2
```

## Commit M2.2 — `run_ocr_batch()` implementation

**File**: `result_companion/core/vision/ocr.py`

Complete file contents:

```python
from __future__ import annotations

import asyncio
import base64
import io
from typing import Sequence

from result_companion.core.utils.logging_config import logger
from result_companion.core.vision.models import Screenshot

_MISSING_EXTRAS_MSG = (
    "Screenshot OCR requires the [vision] extras group. "
    "Install with: pip install 'result-companion[vision]'"
)


def _require_rapidocr():
    """Imports RapidOCR + numpy + PIL. Raises RuntimeError if extras missing."""
    try:
        import numpy as np
        from PIL import Image
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise RuntimeError(_MISSING_EXTRAS_MSG) from exc
    return RapidOCR, Image, np


def _ocr_one_sync(engine, Image, np, screenshot: Screenshot) -> str:
    """Runs OCR on one screenshot synchronously. Returns joined extracted text."""
    try:
        raw = base64.b64decode(screenshot.data_base64, validate=True)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        result = engine(np.array(image))
    except Exception as exc:
        logger.warning(f"OCR failed for test '{screenshot.test_name}': {exc}")
        return ""
    txts = getattr(result, "txts", None)
    if not txts:
        return ""
    return "\n".join(str(t) for t in txts).strip()


async def run_ocr_batch(
    screenshots: Sequence[Screenshot],
    max_per_test: int,
    max_text_length: int,
    concurrency: int,
) -> dict[str, str]:
    """Runs OCR on screenshots, returns per-test concatenated text.

    Groups by test_name. Applies per-test cap BEFORE OCR (skip work).
    Runs OCR in a thread pool because RapidOCR is CPU-bound and sync.
    Truncates concatenated text per test to max_text_length characters.

    Args:
        screenshots: All extracted screenshots.
        max_per_test: Cap screenshots analyzed per test.
        max_text_length: Truncate concatenated per-test text to this many characters.
        concurrency: Max concurrent OCR workers.

    Returns:
        Mapping of test_name to concatenated OCR text (empty tests omitted).
    """
    if not screenshots:
        return {}

    RapidOCR, Image, np = _require_rapidocr()
    engine = RapidOCR()

    capped = _cap_per_test(screenshots, max_per_test)
    logger.info(f"OCR: processing {len(capped)} screenshot(s)")

    sem = asyncio.Semaphore(concurrency)
    loop = asyncio.get_running_loop()

    async def _run_one(shot: Screenshot) -> tuple[str, str]:
        async with sem:
            text = await loop.run_in_executor(
                None, _ocr_one_sync, engine, Image, np, shot
            )
            return shot.test_name, text

    pairs = await asyncio.gather(*[_run_one(s) for s in capped])
    return _group_and_truncate(pairs, max_text_length)


def _cap_per_test(
    screenshots: Sequence[Screenshot], max_per_test: int
) -> list[Screenshot]:
    """Trims screenshots to max_per_test per test_name (preserves order)."""
    counts: dict[str, int] = {}
    selected: list[Screenshot] = []
    for shot in screenshots:
        if counts.get(shot.test_name, 0) >= max_per_test:
            continue
        counts[shot.test_name] = counts.get(shot.test_name, 0) + 1
        selected.append(shot)
    return selected


def _group_and_truncate(
    pairs: list[tuple[str, str]], max_text_length: int
) -> dict[str, str]:
    """Groups (test_name, text) pairs by test_name, joined and truncated."""
    grouped: dict[str, list[str]] = {}
    for name, text in pairs:
        if text:
            grouped.setdefault(name, []).append(text)
    return {
        name: ("\n---\n".join(parts))[:max_text_length]
        for name, parts in grouped.items()
    }
```

**Gotchas the implementer must not "fix"**:

- `RapidOCR()` is instantiated once per `run_ocr_batch` call. Do not cache globally in MVP — testing becomes harder. If perf ever matters, optimize later.
- OCR runs in the default thread executor. Do not switch to a process pool for MVP.
- Empty OCR results (no text found) result in an omitted key, not `""`. Downstream `append_ocr_context` checks membership.
- Truncation is applied AFTER concatenation. Screenshots earlier in the list keep priority.

**Test file**: `tests/unittests/core/vision/test_ocr.py`

Complete file contents:

```python
from unittest.mock import MagicMock, patch

import pytest

from result_companion.core.vision.models import Screenshot
from result_companion.core.vision.ocr import _cap_per_test, run_ocr_batch


def _shot(test_name: str, data: str = "AAAA") -> Screenshot:
    return Screenshot(
        test_name=test_name,
        error_message="",
        mime_type="image/png",
        data_base64=data,
    )


def test_cap_per_test_trims_extras_and_preserves_order() -> None:
    shots = [_shot("A"), _shot("A"), _shot("A"), _shot("B")]
    capped = _cap_per_test(shots, max_per_test=2)
    assert [s.test_name for s in capped] == ["A", "A", "B"]


def test_cap_per_test_with_zero_screenshots_returns_empty() -> None:
    assert _cap_per_test([], max_per_test=3) == []


@pytest.mark.asyncio
async def test_run_ocr_batch_returns_empty_when_no_screenshots() -> None:
    result = await run_ocr_batch(
        [], max_per_test=1, max_text_length=100, concurrency=1
    )
    assert result == {}


@pytest.mark.asyncio
async def test_run_ocr_batch_groups_text_by_test_name() -> None:
    fake_result = MagicMock(txts=["Login failed"])
    fake_engine = MagicMock(return_value=fake_result)
    fake_rapid = MagicMock(return_value=fake_engine)

    with patch(
        "result_companion.core.vision.ocr._require_rapidocr",
        return_value=(fake_rapid, MagicMock(), MagicMock()),
    ), patch(
        "result_companion.core.vision.ocr._ocr_one_sync",
        return_value="Login failed",
    ):
        result = await run_ocr_batch(
            [_shot("Test A"), _shot("Test A"), _shot("Test B")],
            max_per_test=3,
            max_text_length=100,
            concurrency=1,
        )

    assert set(result.keys()) == {"Test A", "Test B"}
    assert "Login failed" in result["Test A"]


@pytest.mark.asyncio
async def test_run_ocr_batch_truncates_to_max_text_length() -> None:
    fake_rapid = MagicMock(return_value=MagicMock())

    with patch(
        "result_companion.core.vision.ocr._require_rapidocr",
        return_value=(fake_rapid, MagicMock(), MagicMock()),
    ), patch(
        "result_companion.core.vision.ocr._ocr_one_sync",
        return_value="X" * 1000,
    ):
        result = await run_ocr_batch(
            [_shot("Test A")],
            max_per_test=1,
            max_text_length=50,
            concurrency=1,
        )

    assert len(result["Test A"]) == 50


@pytest.mark.asyncio
async def test_run_ocr_batch_omits_tests_with_empty_ocr_result() -> None:
    fake_rapid = MagicMock(return_value=MagicMock())

    with patch(
        "result_companion.core.vision.ocr._require_rapidocr",
        return_value=(fake_rapid, MagicMock(), MagicMock()),
    ), patch(
        "result_companion.core.vision.ocr._ocr_one_sync",
        return_value="",
    ):
        result = await run_ocr_batch(
            [_shot("Test A")],
            max_per_test=1,
            max_text_length=100,
            concurrency=1,
        )

    assert result == {}


def test_require_rapidocr_raises_with_install_hint_when_missing(monkeypatch) -> None:
    from result_companion.core.vision import ocr as ocr_mod

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "rapidocr":
            raise ImportError("No module named 'rapidocr'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(RuntimeError, match=r"result-companion\[vision\]"):
        ocr_mod._require_rapidocr()
```

## Commit M2.3 — Wire OCR into `ContextAwareRobotResults`

**File**: `result_companion/core/chunking/rf_results.py`

Add a new attribute to `ContextAwareRobotResults.__init__` (find the constructor and add at the end of its body, right after `self._exclude_passing: bool = False`):

```python
        self._ocr_context: dict[str, str] = {}
```

Add a new public method (place right after `exclude_passing`):

```python
    def append_ocr_context(self, ocr_map: dict[str, str]) -> ContextAwareRobotResults:
        """Attaches OCR-extracted screenshot text to specific tests.

        Each non-empty line becomes a `[SCREENSHOT_OCR] ...` render line appended
        at the end of the test's body, matching the existing `[LEVEL] message`
        convention so LLM and human readers can distinguish OCR-derived content
        from real RF log messages.

        Args:
            ocr_map: Mapping of test_name to concatenated OCR text.

        Returns:
            self (chainable).
        """
        self._ocr_context = dict(ocr_map)
        self._invalidate_cache()
        return self
```

Modify `_iter_tests` — the existing loop yields `RenderedTest(name, status, deduplicate_consecutive_lines(lines))`. Change the loop body to:

```python
    def _iter_tests(self) -> Iterator[RenderedTest]:
        """Internal iterator with passing-test filter and line deduplication applied."""
        for name, status, lines in _iter_tests_with_context(
            self._suite, [], 0, self._fields
        ):
            if self._exclude_passing and status in ("PASS", "SKIP"):
                continue
            enriched = lines + _ocr_render_lines(self._ocr_context.get(name, ""))
            yield RenderedTest(name, status, deduplicate_consecutive_lines(enriched))
```

Add a module-level helper (place after `_iter_tests_with_context`):

```python
def _ocr_render_lines(ocr_text: str) -> list[RenderLine]:
    """Emits one `[SCREENSHOT_OCR] <line>` RenderLine per non-empty OCR line.

    Matches the `[LEVEL] message` convention used by `_render_message` so the
    LLM and human reviewers recognize OCR-derived content in the same shape
    as regular log messages (`[INFO]`, `[ERROR]`, ...).
    """
    return [
        RenderLine(1, f"[SCREENSHOT_OCR] {line}")
        for line in ocr_text.splitlines()
        if line.strip()
    ]
```

**Test additions**: `tests/unittests/core/chunking/test_rf_results.py`

Add one new test (place with the other `ContextAwareRobotResults` tests):

```python
def test_append_ocr_context_marks_lines_with_screenshot_ocr_prefix() -> None:
    fixture = Path(__file__).parents[3] / "fixtures" / "vision" / "embedded_screenshot.xml"
    results = ContextAwareRobotResults(fixture).exclude_passing()
    results.append_ocr_context(
        {"Failing Test With Screenshot": "Error: Element not visible\nURL: /login"}
    )

    text = dict(results.as_texts())["Failing Test With Screenshot"]
    assert "[SCREENSHOT_OCR] Error: Element not visible" in text
    assert "[SCREENSHOT_OCR] URL: /login" in text
```

**Gotchas**:

- Use existing `Path` import at top of test file. If not present, add `from pathlib import Path`.
- The fixture `embedded_screenshot.xml` is the same one M1 created — reuse.
- `_invalidate_cache()` must be called; otherwise `test_names` cache stays stale after `append_ocr_context`. Don't skip it.

## Commit M2.4 — Wire into `_main` + CLI flag

**File**: `result_companion/entrypoints/run_rc.py`

Modify `_main()` signature — add `ocr: bool = False` as a new keyword-only argument (add after `debug_log:`):

```python
    debug_log: Optional[Path] = None,
    ocr: bool = False,
) -> bool:
```

After the existing `apply_concurrency_overrides(...)` line and before the `results = get_rc_robot_results(...)` line, no changes. After `results.set_chunking(strategy)` line, ADD:

```python
    if ocr or parsed_config.vision.enabled:
        parsed_config.vision.enabled = True
        ocr_map = await _run_ocr_step(output=output, config=parsed_config, dryrun=dryrun)
        results.append_ocr_context(ocr_map)
```

Add a new helper function in the same file (place after `_emit_reports`):

```python
async def _run_ocr_step(
    output: Path, config: DefaultConfigModel, dryrun: bool
) -> dict[str, str]:
    """Extracts screenshots and runs OCR. Empty dict on dryrun or no screenshots."""
    if dryrun:
        logger.info("OCR: skipped (dryrun)")
        return {}

    from result_companion.core.vision.extractor import extract_screenshots
    from result_companion.core.vision.ocr import run_ocr_batch

    shots = list(extract_screenshots(output))
    if not shots:
        logger.info("OCR: no embedded screenshots found")
        return {}

    return await run_ocr_batch(
        screenshots=shots,
        max_per_test=config.vision.max_screenshots_per_test,
        max_text_length=config.vision.max_text_length,
        concurrency=config.vision.concurrency,
    )
```

Modify `run_rc()` signature — add `ocr: bool = False` (add after `debug_log:`). Thread through the `asyncio.run(_main(...))` invocation:

```python
                debug_log=debug_log,
                ocr=ocr,
```

**File**: `result_companion/entrypoints/cli/cli_app.py`

Add a new option to the `analyze` command (locate the function decorated with `@app.command()` for `analyze`, add near existing options like `--dryrun`):

```python
    ocr: bool = typer.Option(
        False,
        "--ocr/--no-ocr",
        help="Enable experimental OCR of embedded screenshots (requires [vision] extras).",
    ),
```

Pass `ocr=ocr` through to the `run_rc(...)` invocation inside the `analyze` command body.

**Test file**: `tests/unittests/entrypoints/test_run_rc_ocr.py`

Complete file contents:

```python
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from result_companion.entrypoints.run_rc import _main
from result_companion.core.utils.log_levels import LogLevels


FIXTURE = Path(__file__).parents[2] / "fixtures" / "vision" / "embedded_screenshot.xml"


@pytest.mark.asyncio
async def test_ocr_flag_disabled_skips_extractor(monkeypatch, tmp_path) -> None:
    mock_extract = AsyncMock()
    with patch(
        "result_companion.entrypoints.run_rc._run_ocr_step",
        new=AsyncMock(return_value={}),
    ) as mock_step, patch(
        "result_companion.entrypoints.run_rc.run_analysis",
        new=AsyncMock(return_value=_stub_analysis_result()),
    ):
        await _main(
            output=FIXTURE,
            log_level=LogLevels.INFO,
            config=None,
            report=None,
            include_passing=False,
            html_report=False,
            ocr=False,
        )
    mock_step.assert_not_called()


@pytest.mark.asyncio
async def test_ocr_flag_enabled_calls_ocr_step(tmp_path) -> None:
    with patch(
        "result_companion.entrypoints.run_rc._run_ocr_step",
        new=AsyncMock(return_value={"Failing Test With Screenshot": "OCR text"}),
    ) as mock_step, patch(
        "result_companion.entrypoints.run_rc.run_analysis",
        new=AsyncMock(return_value=_stub_analysis_result()),
    ):
        await _main(
            output=FIXTURE,
            log_level=LogLevels.INFO,
            config=None,
            report=None,
            include_passing=False,
            html_report=False,
            ocr=True,
        )
    mock_step.assert_called_once()


@pytest.mark.asyncio
async def test_ocr_dryrun_returns_empty_map() -> None:
    from result_companion.core.parsers.config import load_config
    from result_companion.entrypoints.run_rc import _run_ocr_step

    cfg = load_config(None)
    result = await _run_ocr_step(output=FIXTURE, config=cfg, dryrun=True)
    assert result == {}


def _stub_analysis_result():
    from result_companion.core.results.analysis_result import AnalysisResult

    return AnalysisResult(llm_results={}, summary=None, test_names=[])
```

**Gotchas**:

- Do not import `run_ocr_batch` at module top level in `run_rc.py`. Import it inside `_run_ocr_step` so users without `[vision]` extras don't crash on `--no-ocr` runs.
- Same for `extract_screenshots` — lazy import inside `_run_ocr_step`.
- The `if ocr or parsed_config.vision.enabled:` line accepts either the CLI flag or config-driven enable. Do not add a third override path.

## Commit M2.5 — README update

**File**: `README.md`

Add a new section (place after existing usage examples, before the "Configuration" section — or wherever the doc structure lets it flow naturally):

```markdown
## Screenshot OCR (experimental)

If your Robot Framework tests capture screenshots via SeleniumLibrary's
`Capture Page Screenshot filename=EMBED` or Browser library equivalents, RC can
extract the embedded PNGs and run local OCR (RapidOCR + ONNX Runtime). The
extracted text is appended to each test's analysis context as
`[SCREENSHOT_OCR] ...` lines (matching RC's existing `[INFO]` / `[ERROR]`
message convention), so the LLM and human reviewers can distinguish OCR
content from real RF log lines.

**Status**: experimental. Text-only OCR — no visual regression, no layout
analysis. English screenshots handled best. Extracted text may be noisy;
the LLM handles that.

### Install

```bash
pip install 'result-companion[vision]'
```

The `[vision]` extra adds ~310 MB (RapidOCR + ONNX Runtime + OpenCV +
supporting libs). Base install stays unchanged for users who don't opt in.

### Enable

Via CLI flag (per-run):

```bash
result-companion analyze -o output.xml --ocr
```

Via config (persistent):

```yaml
vision:
  enabled: true
  max_screenshots_per_test: 3
  max_text_length: 1500
  concurrency: 2
```

### Data flow

Screenshots stay on your machine. RapidOCR runs locally against bundled ONNX
models. Only the extracted text is included in the analysis prompt sent to
your configured LLM. Suitable for enterprise environments where image data
must not leave premises.

### Limitations

- Only `output.xml` embedded (base64) screenshots. File-based screenshots
  linked from `log.html` are not supported.
- OCR is text-only. Visual bugs like color mismatch or layout drift are not
  detected.
- No caching across runs. Re-analyzing the same `output.xml` re-runs OCR.
- First run downloads no additional models — RapidOCR ships them in the
  wheel from v3.9.0 onwards.
```

## M2 Acceptance

- [ ] All new unit tests pass.
- [ ] `result-companion analyze -o output.xml --ocr` on a real RF output with EMBED screenshots produces an `rc_log.html` where the failing tests' LLM analysis references screenshot content.
- [ ] `result-companion analyze -o output.xml --no-ocr` (or omitting the flag) has zero import cost from `rapidocr` — verify by uninstalling `rapidocr` and running with `--no-ocr`; must succeed.
- [ ] `result-companion analyze -o output.xml --ocr` when `[vision]` extras NOT installed → hard-fails with a clear "pip install 'result-companion[vision]'" message.
- [ ] `result-companion analyze -o output.xml --ocr --dryrun` skips OCR and completes without touching RapidOCR.

---

# Testing Summary

| Layer | Tests | Milestone |
|---|---|---|
| Extractor | 6 unit tests | M1 |
| OCR wrapper | 7 unit tests (RapidOCR mocked) | M2 |
| Config | 1 unit test | M2 |
| Injection into `ContextAwareRobotResults` | 1 unit test | M2 |
| CLI + `_main` wiring | 3 unit tests | M2 |

Total new tests: 18. No new integration or e2e tests in MVP.

Run all new tests:

```bash
poetry run pytest tests/unittests/core/vision/ tests/unittests/entrypoints/test_run_rc_ocr.py -v
```

Run existing test suite to confirm no regression:

```bash
make test-unit
```

# Manual Verification (once M1 and M2 land)

1. Install extras: `poetry install --extras vision`.
2. Grab a real RF `output.xml` with EMBED screenshots (any SeleniumLibrary suite using `Set Screenshot Directory  EMBED`).
3. Run: `result-companion analyze -o output.xml --ocr --debug-log ocr-debug.log`.
4. Open `rc_log.html`. Confirm the failing test's AI Analysis section references content that could only have come from screenshot text.
5. Grep `ocr-debug.log` for the OCR-extracted text — it should appear in the prompt block sent to the LLM.
6. Rerun with `--no-ocr`. Same output.xml. Compare AI Analysis sections — the OCR-derived content should be absent.

# Rollback

- Each PR is independently revertable.
- Reverting M2 leaves M1's extractor as dead code — no runtime effect, no user-facing change.
- Reverting M1 removes the extras group. Existing YAMLs with `vision:` block would be silently ignored (Pydantic default `extra="ignore"` behavior in `DefaultConfigModel`). No migration needed.

# Deferred (Do NOT do in MVP)

- Optional LLM-based description path (`extractor: llm` config option).
- OCR result caching (hash by image bytes + skip on re-run).
- Inline positioning of OCR text at the exact `<msg>` location.
- Multi-language OCR models (RapidOCR supports it; default is fine for MVP).
- Per-image OCR timing metrics.
- `--ocr-report` separate output file.
- Prompt injection hardening in the main analysis prompt.
- PII redaction of OCR output.
- Non-Robot plugin support.

These are follow-up work. Do not add stubs, TODOs, or "future" hooks for them in the MVP code — YAGNI. Add them in dedicated PRs when someone actually asks.
