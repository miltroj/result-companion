# Robot Framework Vision — Implementation Plan

Companion to [`docs/plugin-scope.md`](plugin-scope.md). This is the executable plan for the **multimodal vision pipeline**: extract embedded screenshots from Robot Framework `output.xml`, feed them to a multimodal LLM alongside the failure breadcrumb, render the vision output in `rc_log.html` next to the text analysis.

Sister plan: [`docs/plugin-architecture-plan.md`](plugin-architecture-plan.md) — Protocol cleanup and the bundled JUnit escape hatch.

## Purpose

Ship the feature [`docs/plugin-scope.md`](plugin-scope.md) calls RC's biggest untapped differentiator:

> Given a Robot Framework `output.xml` with embedded base64 screenshots, RC extracts each screenshot, pairs it with the failing keyword's breadcrumb, feeds the pair to a multimodal LLM, and renders the vision output in `rc_log.html` next to the text analysis.

Ships as **opt-in** — off by default. Users enable via config or `--vision` flag.

## Success Criteria (from `plugin-scope.md`)

- [ ] Robot image-analysis prototype: extract screenshots from `output.xml`, call multimodal LLM, render vision output in `rc_log.html` next to text analysis. Opt-in.

## Explicit Non-Goals

- No fallback to `log.html` screenshot extraction — only `output.xml` embedded screenshots (see Assumption below).
- No Playwright / trace-viewer integration.
- No non-Robot vision support (JUnit and others strip artifacts anyway).
- No community plugin marketing push.

## Prerequisites

- **Recommended, not strictly blocking**: Protocol cleanup — see [`docs/plugin-architecture-plan.md`](plugin-architecture-plan.md) M1. The vision code lives in a new `core/vision/` package and does not depend on the `ParsedResults` Protocol. However, M2's wiring in `run_rc.py` will conflict-adjacent with plugin-architecture M1's `C1.4` edits — if both branches run in parallel, merge M1 first.
- Existing `_smart_acompletion` LLM router — LiteLLM handles both Ollama vision (`ollama_chat/llava:7b`) and OpenAI vision (`openai/gpt-4o`) via the same multimodal message shape. No new adapter framework needed.

## Assumption

Screenshots are embedded as base64 inside `<msg html="true">` under a `<kw>` (keyword). This is how `SeleniumLibrary`'s `Capture Page Screenshot` with `filename=EMBED` (or `Set Screenshot Directory  EMBED`) produces output. If users use file-based screenshots, this pipeline is a no-op (extractor returns nothing) and the feature quietly does nothing. Explicit non-goal: fall back to `log.html`.

## Milestones Overview

| ID | Deliverable | Ships alone? | Est. commits | Risk |
|----|-------------|--------------|--------------|------|
| M1 | Screenshot Extraction — parse `output.xml` for screenshots, no LLM | Yes (dead code until M2) | 3 | Low |
| M2 | Vision LLM Wiring — config + prompt + analyzer + `run_rc` glue | Yes (produces data but not rendered) | 4 | Medium |
| M3 | Vision Rendering + CLI — HTML/JSON output, `--vision` flag | Yes (user-visible) | 4 | Low |

**Total**: 3 PRs, ~11 commits. Each PR is independently reviewable and shippable.

**Ordering**: M1 → M2 → M3. Sequential.

---

# Milestone M1 — Screenshot Extraction

## Rationale

Isolate the XML-parsing complexity. No LLM concerns yet. Extractor is pure I/O + parsing; testable with a small fixture.

## Files Touched

| File | Change type |
|------|-------------|
| `result_companion/core/vision/__init__.py` | New (empty) |
| `result_companion/core/vision/extractor.py` | New |
| `result_companion/core/vision/models.py` | New — `Screenshot` dataclass |
| `tests/core/vision/test_extractor.py` | New |
| `tests/fixtures/vision/output_with_embedded_screenshots.xml` | New — small RF output with one failure and one embedded PNG |

## Commits

### C1.1 — `Screenshot` dataclass

**File**: `result_companion/core/vision/models.py`

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Screenshot:
    """One screenshot extracted from a Robot Framework output.xml.

    Attributes:
        test_name: Full test name including suite path.
        breadcrumb: Ordered list of parent keyword names leading to this screenshot.
        error_message: Failure message from the containing test, or empty if the test passed.
        mime_type: e.g. "image/png".
        data_base64: Raw base64 payload (no data-URI prefix).
    """
    test_name: str
    breadcrumb: tuple[str, ...]
    error_message: str
    mime_type: str
    data_base64: str

    @property
    def data_uri(self) -> str:
        """Returns full data URI for LLM API consumption."""
        return f"data:{self.mime_type};base64,{self.data_base64}"
```

**Test**: `test_screenshot_data_uri_composes_correctly` — construct with `mime_type="image/png"`, `data_base64="AAA"`, assert `data_uri == "data:image/png;base64,AAA"`.

### C1.2 — Extractor implementation

**File**: `result_companion/core/vision/extractor.py`

```python
from __future__ import annotations

import base64
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

from result_companion.core.vision.models import Screenshot


_IMG_SRC_PATTERN = re.compile(
    r"""<img[^>]+src=["']data:(?P<mime>image/[a-z0-9+.-]+);base64,(?P<data>[A-Za-z0-9+/=\s]+)["']""",
    re.IGNORECASE,
)


def extract_screenshots(output_xml_path: Path) -> Iterator[Screenshot]:
    """Yields one Screenshot per embedded base64 image found in output.xml.

    Walks the RF output tree, tracks the enclosing test name and the keyword
    breadcrumb, and scans <msg html="true"> text nodes for embedded <img>
    tags with data-URI sources. Skips screenshots inside PASS tests unless
    they are inside a failed keyword.

    Args:
        output_xml_path: Path to a Robot Framework output.xml file.

    Yields:
        Screenshot instances in document order.
    """
    context = ET.iterparse(output_xml_path, events=("start", "end"))
    current_test: str | None = None
    current_test_status: str = ""
    current_test_message: str = ""
    keyword_stack: list[str] = []

    for event, element in context:
        tag = _local_name(element.tag)

        if event == "start":
            if tag == "test":
                current_test = element.attrib.get("name", "<unnamed>")
                current_test_status = ""
                current_test_message = ""
            elif tag == "kw":
                keyword_stack.append(element.attrib.get("name", "<unnamed>"))
            elif tag == "msg" and element.attrib.get("html") == "true":
                pass  # text is only reliable on end

        else:  # end
            if tag == "msg" and element.attrib.get("html") == "true":
                yield from _scan_msg_for_images(
                    element.text or "",
                    test_name=current_test or "",
                    breadcrumb=tuple(keyword_stack),
                    error_message=current_test_message,
                )
            elif tag == "kw":
                if keyword_stack:
                    keyword_stack.pop()
            elif tag == "status" and current_test is not None:
                current_test_status = element.attrib.get("status", "")
                current_test_message = (element.text or "").strip()
            elif tag == "test":
                current_test = None
                current_test_status = ""
                current_test_message = ""

            element.clear()  # free memory for large files


def _scan_msg_for_images(
    html_text: str,
    test_name: str,
    breadcrumb: tuple[str, ...],
    error_message: str,
) -> Iterator[Screenshot]:
    """Yields Screenshots for every embedded base64 <img> in the HTML message."""
    for match in _IMG_SRC_PATTERN.finditer(html_text):
        yield Screenshot(
            test_name=test_name,
            breadcrumb=breadcrumb,
            error_message=error_message,
            mime_type=match.group("mime"),
            data_base64=re.sub(r"\s+", "", match.group("data")),
        )


def _local_name(tag: str) -> str:
    """Strips XML namespace."""
    return tag.rsplit("}", 1)[-1]
```

**Constraints**:
- Use `ET.iterparse` — never load the whole tree. `output.xml` can be hundreds of MB.
- After emitting each `Screenshot`, call `element.clear()` on the parent to free memory.
- Do **not** decode the base64 payload here — pass through as-is; LLM adapter builds the data URI.

**Test**: `tests/core/vision/test_extractor.py`:
- `test_extract_screenshots_yields_one_per_embedded_img` — fixture with one failure and one screenshot.
- `test_extract_screenshots_captures_breadcrumb_in_order` — nested keywords, assert `screenshot.breadcrumb == ("Outer KW", "Inner KW")`.
- `test_extract_screenshots_carries_failure_message` — test with a failure message, assert `screenshot.error_message == expected`.
- `test_extract_screenshots_returns_empty_when_no_images` — fixture with no `<img>` tags.
- `test_extract_screenshots_handles_multiple_images_per_test` — two screenshots in one test.

### C1.3 — Fixture with a real embedded screenshot

**File**: `tests/fixtures/vision/output_with_embedded_screenshots.xml`

- One `<suite>` with two `<test>` elements: one PASS, one FAIL.
- The FAIL test contains a nested `<kw>` with a `<msg html="true">` that embeds a 10×10 red PNG as base64. Use a real tiny PNG so extractor is exercised end-to-end.
- Include a `<status>` element on the test with the FAIL message.

Keep it under 5 KB. Comment the fixture at the top: `<!-- Fixture: minimal RF output with one embedded screenshot for extractor tests -->`.

## M1 Acceptance Criteria

- [ ] `extract_screenshots(fixture)` yields exactly one `Screenshot` from the FAIL test.
- [ ] Memory profile with a 100 MB synthetic `output.xml` stays under 50 MB peak (spot-check, not CI).
- [ ] All 5 unit tests pass.

---

# Milestone M2 — Vision LLM Wiring

## Rationale

Given a `Screenshot`, ask the multimodal LLM to describe what failed. Reuse the existing `_smart_acompletion` router — **LiteLLM handles both Ollama vision (`ollama_chat/llava:7b`) and OpenAI vision (`openai/gpt-4o`) via the same multimodal message shape**. No new adapter framework needed.

## Files Touched

| File | Change type |
|------|-------------|
| `result_companion/core/parsers/config.py` | Add `VisionConfigModel`; add optional `vision` field on `DefaultConfigModel` |
| `result_companion/core/configs/default_config.yaml` | Add `vision:` section (disabled by default) |
| `result_companion/core/vision/prompt.py` | New — builds multimodal message list |
| `result_companion/core/vision/analyzer.py` | New — `analyze_screenshot()` async function |
| `tests/core/vision/test_prompt.py` | New |
| `tests/core/vision/test_analyzer.py` | New (uses fake `acompletion`) |

## Commits

### C2.1 — `VisionConfigModel` in config

**File**: `result_companion/core/parsers/config.py`

Add:

```python
class VisionConfigModel(BaseModel):
    """Optional vision analysis configuration."""

    enabled: bool = Field(default=False, description="Enable multimodal analysis of embedded screenshots.")
    model: str = Field(
        default="ollama_chat/llava:7b",
        description="LiteLLM model identifier for a vision-capable model.",
    )
    prompt: str = Field(
        default=(
            "You are analyzing a screenshot from an automated test failure.\n"
            "Test: {test_name}\n"
            "Failure context: {breadcrumb}\n"
            "Error message: {error_message}\n\n"
            "Describe what is visible on the screen that would explain the failure. "
            "Focus on error dialogs, missing elements, unexpected states, or visual anomalies. "
            "Be concise (2-4 sentences)."
        ),
        description="Prompt template. Placeholders: {test_name}, {breadcrumb}, {error_message}.",
    )
    max_screenshots_per_test: int = Field(
        default=1, ge=1,
        description="Limit screenshots analyzed per test to bound cost.",
    )


class DefaultConfigModel(BaseModel):
    # ... existing fields ...
    vision: VisionConfigModel = Field(default_factory=VisionConfigModel)
```

Add `vision` to the merge dict in `ConfigLoader.load_config`:

```python
"vision": {
    **default_config.get("vision", {}),
    **user_config.get("vision", {}),
},
```

**File**: `result_companion/core/configs/default_config.yaml`

Append:

```yaml
vision:
  enabled: false
  model: ollama_chat/llava:7b
  prompt: |
    You are analyzing a screenshot from an automated test failure.
    Test: {test_name}
    Failure context: {breadcrumb}
    Error message: {error_message}

    Describe what is visible on the screen that would explain the failure.
    Focus on error dialogs, missing elements, unexpected states, or visual anomalies.
    Be concise (2-4 sentences).
  max_screenshots_per_test: 1
```

**Test**: `tests/core/parsers/test_config.py::test_default_config_has_vision_disabled` — load default config, assert `config.vision.enabled is False`.

### C2.2 — Multimodal prompt builder

**File**: `result_companion/core/vision/prompt.py`

```python
from __future__ import annotations

from typing import Any

from result_companion.core.vision.models import Screenshot


def build_vision_messages(
    screenshot: Screenshot,
    prompt_template: str,
) -> list[dict[str, Any]]:
    """Builds a LiteLLM-compatible multimodal message list for one screenshot.

    Args:
        screenshot: Extracted screenshot with breadcrumb + error message.
        prompt_template: Format string with {test_name}, {breadcrumb}, {error_message}.

    Returns:
        Messages list ready to pass to _smart_acompletion(messages=...).
    """
    text = prompt_template.format(
        test_name=screenshot.test_name,
        breadcrumb=" > ".join(screenshot.breadcrumb) or "(no keyword context)",
        error_message=screenshot.error_message or "(no explicit error message)",
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": screenshot.data_uri}},
            ],
        }
    ]
```

**Constraints**:
- The `content: [...]` list shape is what LiteLLM forwards to both Ollama and OpenAI. Do not use a `str` for `content`.
- Do **not** pre-decode base64. Pass the data URI as-is.

**Test**: `tests/core/vision/test_prompt.py`:
- `test_build_vision_messages_returns_single_user_message`
- `test_build_vision_messages_formats_prompt_placeholders`
- `test_build_vision_messages_uses_content_list_shape`

### C2.3 — `analyze_screenshot()` async

**File**: `result_companion/core/vision/analyzer.py`

```python
from __future__ import annotations

import asyncio
from typing import Any

from result_companion.core.analizers.llm_router import _smart_acompletion
from result_companion.core.utils.llm_debug import LLMDebugLogger
from result_companion.core.vision.models import Screenshot
from result_companion.core.vision.prompt import build_vision_messages


async def analyze_screenshot(
    screenshot: Screenshot,
    model: str,
    prompt_template: str,
    debug_logger: LLMDebugLogger = LLMDebugLogger(),
    **llm_params: Any,
) -> str:
    """Analyzes a single screenshot via a multimodal LLM.

    Args:
        screenshot: Screenshot to analyze.
        model: LiteLLM model identifier for a vision-capable model.
        prompt_template: Format string for the text portion of the prompt.
        debug_logger: Optional debug logger for prompt/response capture.
        **llm_params: Additional LLM parameters forwarded to _smart_acompletion.

    Returns:
        Vision LLM response text.
    """
    messages = build_vision_messages(screenshot, prompt_template)
    response = await _smart_acompletion(messages=messages, model=model, **llm_params)
    result = response.choices[0].message.content
    if debug_logger.enabled:
        debug_logger.write_record(
            label=f"[VISION] {screenshot.test_name}",
            prompt=str(messages),
            response=result,
        )
    return result


async def analyze_screenshots(
    screenshots: list[Screenshot],
    model: str,
    prompt_template: str,
    max_per_test: int = 1,
    debug_logger: LLMDebugLogger = LLMDebugLogger(),
    **llm_params: Any,
) -> dict[str, list[str]]:
    """Analyzes screenshots grouped by test name, respecting max_per_test.

    Args:
        screenshots: All extracted screenshots.
        model: LiteLLM model identifier.
        prompt_template: Prompt format string.
        max_per_test: Cap per test to bound cost.
        debug_logger: Optional debug logger.
        **llm_params: Additional LLM parameters.

    Returns:
        Mapping of test_name -> list of vision responses (in extraction order).
    """
    selected = _cap_per_test(screenshots, max_per_test)
    tasks = [
        analyze_screenshot(
            screenshot=shot,
            model=model,
            prompt_template=prompt_template,
            debug_logger=debug_logger,
            **llm_params,
        )
        for shot in selected
    ]
    responses = await asyncio.gather(*tasks)
    grouped: dict[str, list[str]] = {}
    for shot, response in zip(selected, responses):
        grouped.setdefault(shot.test_name, []).append(response)
    return grouped


def _cap_per_test(
    screenshots: list[Screenshot], max_per_test: int
) -> list[Screenshot]:
    """Returns screenshots trimmed to max_per_test per test_name."""
    counts: dict[str, int] = {}
    selected: list[Screenshot] = []
    for shot in screenshots:
        if counts.get(shot.test_name, 0) >= max_per_test:
            continue
        counts[shot.test_name] = counts.get(shot.test_name, 0) + 1
        selected.append(shot)
    return selected
```

**Constraints**:
- Reuse `_smart_acompletion`. Do not open a second LLM client.
- Concurrency via `asyncio.gather` is fine for now; add a semaphore later if needed.
- `_cap_per_test` is pure — testable in isolation.

**Test**: `tests/core/vision/test_analyzer.py`:
- Use `unittest.mock.patch` on `_smart_acompletion` returning a fake response.
- `test_analyze_screenshot_returns_llm_content`
- `test_analyze_screenshots_groups_by_test_name`
- `test_analyze_screenshots_respects_max_per_test`
- `test_cap_per_test_returns_original_order`

### C2.4 — Wire vision into `_main` (behind `config.vision.enabled`)

**File**: `result_companion/entrypoints/run_rc.py`

After `plugin.parse(...)` and **only when** the plugin is `RobotPlugin` and `parsed_config.vision.enabled`:

```python
vision_results: dict[str, list[str]] = {}
if isinstance(plugin, RobotPlugin) and parsed_config.vision.enabled and not dryrun:
    from result_companion.core.vision.analyzer import analyze_screenshots
    from result_companion.core.vision.extractor import extract_screenshots

    screenshots = list(extract_screenshots(output))
    if screenshots:
        logger.info(f"Extracted {len(screenshots)} embedded screenshots for vision analysis")
        vision_results = await analyze_screenshots(
            screenshots=screenshots,
            model=parsed_config.vision.model,
            prompt_template=parsed_config.vision.prompt,
            max_per_test=parsed_config.vision.max_screenshots_per_test,
            debug_logger=parsed_config.debug_logger,
        )
```

Pass `vision_results` to `_emit_reports(...)` as a new argument (rendering happens in M3, but wiring goes in now so the data reaches the caller).

**Constraints**:
- Guard with `isinstance(plugin, RobotPlugin)` — vision is Robot-only.
- `dryrun=True` must skip the vision LLM call. Extractor may still run for logging counts.
- Import inside the block to avoid loading vision code when disabled.

**Test**: `tests/entrypoints/test_run_rc_vision.py`:
- `test_vision_disabled_by_default_no_extractor_call` — patch `extract_screenshots`, run with default config, assert not called.
- `test_vision_enabled_calls_extractor_and_analyzer` — enable vision in config, patch both, assert both called with expected args.
- `test_vision_dryrun_skips_llm_call` — enable vision + dryrun, assert `analyze_screenshots` not called.

## M2 Acceptance Criteria

- [ ] `config.vision.enabled = false` → zero import/execution cost from vision modules.
- [ ] `config.vision.enabled = true` + fixture with 1 embedded screenshot → `vision_results` has 1 entry.
- [ ] All new unit tests green.

---

# Milestone M3 — Vision Rendering + CLI

## Rationale

Data flows to `_emit_reports`. Now surface it in HTML, JSON, and text reports. Add CLI toggle.

## Files Touched

| File | Change type |
|------|-------------|
| `result_companion/entrypoints/cli/cli_app.py` | Add `--vision / --no-vision` flag on `analyze` |
| `result_companion/entrypoints/run_rc.py` | Wire CLI override; pass `vision_results` to `_emit_reports` |
| `result_companion/api.py` | Same |
| `result_companion/core/html/html_creator.py` | Accept + render `vision_results` |
| `result_companion/core/results/text_report.py` | Accept `vision_results` for JSON output |
| `examples/PLUGIN_ARCHITECTURE.md` | Document `render_html_report`'s new optional `vision_results` param |

## Commits

### C3.1 — CLI `--vision` / `--no-vision` override

**File**: `result_companion/entrypoints/cli/cli_app.py`

Add on `analyze`:

```python
vision: Optional[bool] = typer.Option(
    None,
    "--vision/--no-vision",
    help="Enable multimodal analysis of embedded screenshots (Robot only). "
         "Overrides config.vision.enabled.",
),
```

Pass through `run(..., vision=vision)`.

**File**: `result_companion/entrypoints/run_rc.py`

Add `vision: Optional[bool] = None` to `run_rc()` and `_main()`. After `load_config`:

```python
if vision is not None:
    parsed_config.vision.enabled = vision  # CLI wins
```

Note: this mutates the frozen-ish config. If `DefaultConfigModel` is truly frozen (Pydantic v2 defaults to mutable), leave as-is; otherwise use `parsed_config.vision = parsed_config.vision.model_copy(update={"enabled": vision})`.

**Test**: `tests/entrypoints/test_cli_app.py::test_analyze_vision_flag_overrides_config`.

### C3.2 — `render_html_report` accepts `vision_results`

**File**: `result_companion/core/plugins/robot.py`

Extend signature (**additive**, keyword-only, defaulting to `None`):

```python
def render_html_report(
    self,
    input_path: Path,
    output_path: Path,
    llm_results: dict[str, str],
    model_info: dict[str, str] | None = None,
    overall_summary: str | None = None,
    vision_results: dict[str, list[str]] | None = None,
) -> None:
    create_llm_html_log(
        input_result_path=input_path,
        llm_output_path=output_path,
        llm_results=llm_results,
        model_info=model_info,
        overall_summary=overall_summary,
        vision_results=vision_results,
    )
```

**File**: `result_companion/core/html/html_creator.py`

Add `vision_results` parameter; thread it into the template. Rendering choice: per test, insert an additional `<section class="vision-analysis">` after the text-analysis block, containing `join("\n\n---\n\n", responses)` markdown-rendered.

**Constraints**:
- Backward compat — old callers omitting `vision_results` continue to work.
- If `vision_results is None` or the specific test has no entries → skip the section entirely (no empty box).

**Test**: `tests/core/html/test_html_creator.py`:
- `test_render_html_includes_vision_section_when_provided`
- `test_render_html_omits_vision_section_when_absent`

### C3.3 — JSON report includes vision output

**File**: `result_companion/core/results/text_report.py`

Extend `render_json_report(...)` with `vision_results: dict[str, list[str]] | None = None`. Include per-test in output:

```json
{
  "test_name": "My Failing Test",
  "llm_analysis": "...",
  "vision_analysis": ["...response 1...", "...response 2..."]
}
```

If a test has no vision entries → key omitted (not `null`, not `[]`).

**Test**: `tests/core/results/test_text_report.py::test_render_json_report_includes_vision_when_present`.

### C3.4 — Update `_emit_reports` to forward `vision_results`

**File**: `result_companion/entrypoints/run_rc.py` and `result_companion/api.py`

Add `vision_results: dict[str, list[str]] | None = None` to `_emit_reports` signature. Pass through to `render_html_report(...)` and `render_json_report(...)`.

**Test**: `tests/entrypoints/test_run_rc_vision.py::test_vision_results_reach_emit_reports`.

## M3 Acceptance Criteria

- [ ] `rc analyze -o output.xml --vision` with fixture → `rc_log.html` contains a "Screenshot analysis" section under the failing test.
- [ ] `rc analyze -o output.xml --no-vision` → HTML has no vision section even if config enables vision.
- [ ] `--json-report` includes `vision_analysis` key for tests that had vision runs.
- [ ] Non-Robot plugins with `--vision` → warning logged, feature skipped, no crash.

# Full Test Plan Summary

| Layer | Tests | Milestone |
|-------|-------|-----------|
| Extractor | 5 unit tests | M1 |
| Prompt builder | 3 unit tests | M2 |
| Analyzer | 4 unit tests | M2 |
| Config | 1 unit test | M2 |
| `run_rc` wiring | 3 unit tests | M2 |
| CLI | 1 unit test | M3 |
| HTML | 2 unit tests | M3 |
| JSON | 1 unit test | M3 |
| E2E | 1 fixture-based test running the full pipeline end-to-end with a patched `_smart_acompletion` returning fake vision responses | M3 |

---

# Backward Compatibility & Rollback

- **M2** adds `config.vision.*` — defaults to disabled → existing users see no change.
- **M3** adds `--vision` flag — new capability, does not affect existing runs.
- Each PR is independently revertable. M2 can be reverted without touching M1 or M3 (M1's extractor becomes dead code again, which is fine).

## Config File Migration

**None required.** All new config keys (`vision:`) have defaults. Existing user YAMLs remain valid.

---

# Deferred / Not Doing Yet

Deliberately not included in this plan (per `plugin-scope.md` non-goals):

- Playwright native plugin — deferred until Robot vision proves its differentiation.
- `log.html` screenshot extraction fallback — add only if EMBED-mode assumption breaks real users.
- Multi-image reasoning (correlate screenshots across steps) — YAGNI; add if users ask.
- Vision-model caching / dedup — bounded already by `max_screenshots_per_test`.

---

# Reference

- [`docs/plugin-scope.md`](plugin-scope.md) — honest scope, priorities, non-goals.
- [`docs/plugin-architecture-plan.md`](plugin-architecture-plan.md) — companion plan for Protocol cleanup + bundled JUnit.
- [`AGENTS.md`](../AGENTS.md) — project coding standards (test style, docstrings, nesting, complexity, dependency injection).
- [`result_companion/core/analizers/llm_router.py`](../result_companion/core/analizers/llm_router.py) — the shared LLM entry point vision reuses.
- [`result_companion/entrypoints/run_rc.py`](../result_companion/entrypoints/run_rc.py) — CLI orchestrator, primary integration point.
- LiteLLM multimodal message spec: `content` list with `{"type": "text"}` + `{"type": "image_url"}` blocks.
