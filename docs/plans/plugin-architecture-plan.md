# Plugin Architecture — Implementation Plan

Companion to [`docs/plugin-scope.md`](plugin-scope.md). This is the executable plan for the **plugin-architecture** portion: Protocol cleanup and the bundled JUnit escape hatch. Each milestone maps to a PR, each PR to commits small enough for a focused implementer to complete without design decisions.

Sister plan: [`docs/robot-vision-plan.md`](robot-vision-plan.md) — the multimodal Robot Framework vision pipeline.

## Purpose

Ship the plugin-architecture foundation [`docs/plugin-scope.md`](plugin-scope.md) identifies as necessary for RC's second-priority feature, in the smallest coherent slices:

1. Minimal Protocol cleanup that removes Robot-specific leaks from `ParsedResults`.
2. JUnit XML escape hatch, bundled built-in.

## Success Criteria (from `plugin-scope.md`)

- [ ] `ParserPlugin` Protocol lives in core; Robot plugin conforms; **`set_chunking` / `has_chunking` no longer in Protocol**.
- [ ] JUnit adapter ships as bundled built-in; `rc analyze -o pytest.xml -f pytest-junit` works end-to-end.
- [ ] `analyze --format` errors clearly when the user asks for a capability the plugin does not support.
- [ ] Docs describe when RC beats "paste to Copilot Chat" and when it does not.

## Explicit Non-Goals (from `plugin-scope.md`, unchanged)

- No `Capabilities` dataclass. No `api_version`. No conformance kit. No renderer/reporter planes.
- No Playwright / Cucumber / TestNG native plugins.
- No community plugin marketing push.
- No changes to the multi-plugin infra beyond `ResultParserPlugin`.

## Milestones Overview

| ID | Title | Blocker for | PRs | Est. commits | Risk |
|----|-------|-------------|-----|--------------|------|
| M1 | Protocol Cleanup Foundation | M2 (and the vision plan) | 1 | 7 | Low |
| M2 | JUnit Escape Hatch (Bundled) | — | 1 | 5 | Low |

**Total**: 2 PRs, ~12 commits. Each PR is independently reviewable and shippable.

**Ordering**: M1 → M2. M1 also unblocks the vision plan (see [`docs/robot-vision-plan.md`](robot-vision-plan.md)), but the vision pipeline is otherwise independent and can proceed in parallel after M1 lands.

---

# Milestone M1 — Protocol Cleanup Foundation

## Rationale

Current `ParsedResults` Protocol requires `has_chunking: bool` and `set_chunking(strategy)`. Both are Robot-shaped: `ChunkingStrategy.apply()` takes `list[RenderLine]`. The bundled JUnit escape hatch (M2) and any future non-RF plugin need to either fake this shape or bypass the Protocol. Cleanup removes the leak so M2 can be small and the vision plan doesn't have to inherit Robot's chunking coupling in new code.

Scope: **behaviour-preserving** for Robot users. No new features. No public API changes visible to CLI users. Internal Protocol shape changes only.

## Files Touched

| File | Change type |
|------|-------------|
| `result_companion/core/plugins/base.py` | Update `ParseOptions`; slim `ParsedResults` Protocol; add `@runtime_checkable` |
| `result_companion/core/chunking/helpers.py` | **New** — `token_aware_chunk()` helper |
| `result_companion/core/chunking/chunking.py` | Keep for now; `ChunkingStrategy` becomes a thin wrapper over the new helper. Do **not** delete in this milestone. |
| `result_companion/core/chunking/rf_results.py` | Chunking is invoked from inside `render_chunks()` using the helper + `ParseOptions` |
| `result_companion/core/plugins/robot.py` | No functional change; may adjust type hints |
| `result_companion/core/plugins/registry.py` | `_is_parser_plugin` uses `isinstance()` against runtime-checkable Protocol |
| `result_companion/entrypoints/run_rc.py` | Remove external `set_chunking` call; pass `tokenizer_config` + `system_prompt` in `ParseOptions` |
| `result_companion/api.py` | Same as `run_rc.py` |
| `examples/plugins/pytest_junit/result_companion_pytest_junit/plugin.py` | Migrate to new Protocol shape (canary) |
| `examples/PLUGIN_ARCHITECTURE.md` | Docs update: Protocol section + `render_html_report` signature |

## Commits

### C1.1 — Add `tokenizer_config` and `system_prompt` to `ParseOptions`

**File**: `result_companion/core/plugins/base.py`

```python
from result_companion.core.parsers.config import TokenizerModel

@dataclass(frozen=True)
class ParseOptions:
    """Options shared by result parser plugins."""

    include_tags: list[str] | None = None
    exclude_tags: list[str] | None = None
    exclude_fields: list[str] | None = None
    exclude_passing: bool = True
    tokenizer_config: TokenizerModel | None = None
    system_prompt: str = ""
```

**Constraints**:
- New fields have defaults → additive change, no callers break.
- Do **not** import from `chunking.py` here (would create a cycle). `TokenizerModel` lives in `core/parsers/config.py`.

**Test**: `tests/core/plugins/test_base.py::test_parse_options_defaults` — construct `ParseOptions()`, assert both new fields default to `None` / `""`.

### C1.2 — New `token_aware_chunk()` helper

**File**: `result_companion/core/chunking/helpers.py` (new)

```python
from __future__ import annotations

from typing import Union

from result_companion.core.chunking.chunking import (
    RenderLine,
    chunk_rf_test_lines,
    render_lines_to_text,
)
from result_companion.core.chunking.utils import Chunking, calculate_chunk_size
from result_companion.core.parsers.config import TokenizerModel


def token_aware_chunk(
    lines: list[RenderLine],
    tokenizer_config: TokenizerModel,
    system_prompt: str,
) -> tuple[list[str], Chunking]:
    """Splits rendered lines into chunks fitting the model context minus system_prompt.

    Args:
        lines: (depth, text) pairs from a plugin's per-test rendering.
        tokenizer_config: Model tokenizer + max_content_tokens.
        system_prompt: Prompt whose token count is reserved from the budget.

    Returns:
        (chunks, chunk_stats) where chunks is a list of self-contained text
        blocks and chunk_stats carries totals for JSON/text reports.
    """
    rendered = render_lines_to_text(lines)
    chunk_info = calculate_chunk_size(rendered, system_prompt, tokenizer_config)
    return chunk_rf_test_lines(lines, chunk_info.chunk_size), chunk_info


def single_chunk(
    text: str,
    tokenizer_config: TokenizerModel,
) -> tuple[list[str], Chunking]:
    """Wraps small per-test text as a single chunk for plugins that never split.

    Use when a plugin's per-test output reliably fits one context window
    (e.g. JUnit testcase). Avoids importing the RF-shaped chunker.
    """
    stats = Chunking(
        chunk_size=len(text),
        number_of_chunks=1 if text else 0,
        raw_text_len=len(text),
        tokens_from_raw_text=0,
        tokenized_chunks=[],
    )
    return ([text] if text else []), stats
```

**Constraints**:
- Helper is a pure function. No I/O. Testable without RF fixtures.
- Do **not** delete `ChunkingStrategy` yet. It stays as a wrapper for one more commit.

**Test**: `tests/core/chunking/test_helpers.py`:
- `test_token_aware_chunk_returns_single_chunk_for_small_input`
- `test_token_aware_chunk_splits_when_text_exceeds_budget`
- `test_single_chunk_returns_empty_list_for_empty_text`
- `test_single_chunk_wraps_text_in_one_element_list`

### C1.3 — RF plugin uses helper internally

**File**: `result_companion/core/chunking/rf_results.py`

- `ContextAwareRobotResults.render_chunks()` no longer requires prior `set_chunking()` call.
- Instead it reads `self._options.tokenizer_config` and `self._options.system_prompt` (both now on `ParseOptions`).
- If both present → call `token_aware_chunk(lines, tokenizer_config, system_prompt)`.
- If either absent → raise `ValueError("RobotPlugin requires tokenizer_config and system_prompt in ParseOptions")`.

Keep `set_chunking()` and `has_chunking` on `ContextAwareRobotResults` as **compat wrappers** in this commit — they still work if called, but nothing internal depends on them.

```python
# Rough sketch (adjust to actual class shape):
def render_chunks(self) -> Iterator[TestChunkPayload]:
    tokenizer_config = self._options.tokenizer_config
    system_prompt = self._options.system_prompt
    if tokenizer_config is None:
        raise ValueError(
            "RobotPlugin needs tokenizer_config in ParseOptions. "
            "The CLI populates this automatically."
        )
    for test_name, lines, status in self._iter_tests():
        chunks, chunk_stats = token_aware_chunk(lines, tokenizer_config, system_prompt)
        yield TestChunkPayload(test_name, chunks, chunk_stats, status)
```

**Test**: Existing RF chunking tests must pass unchanged. Add:
- `test_render_chunks_raises_when_tokenizer_missing` — construct results with `ParseOptions(tokenizer_config=None)`, assert `ValueError` from `next(results.render_chunks())`.

### C1.4 — `run_rc.py` + `api.py` stop calling `set_chunking` externally

**Files**:
- `result_companion/entrypoints/run_rc.py`
- `result_companion/api.py`

Replace the current:

```python
options = ParseOptions(...)
results = plugin.parse(output, options)
if not results.has_chunking:
    strategy = ChunkingStrategy(
        tokenizer_config=parsed_config.tokenizer,
        system_prompt=parsed_config.llm_config.question_prompt,
    )
    results.set_chunking(strategy)
```

With:

```python
options = ParseOptions(
    ...,
    tokenizer_config=parsed_config.tokenizer,
    system_prompt=parsed_config.llm_config.question_prompt,
)
results = plugin.parse(output, options)
```

**Constraints**:
- Do **not** import `ChunkingStrategy` here anymore.
- Do **not** touch `_emit_reports()` in this commit.

**Test**: Existing e2e `run_rc` integration tests must pass unchanged.

### C1.5 — Remove `set_chunking` / `has_chunking` from Protocol; add `@runtime_checkable`

**File**: `result_companion/core/plugins/base.py`

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class ParsedResults(Protocol):
    test_names: list[str]
    total_test_count: int
    source_hash: str

    def render_chunks(self) -> Iterator["TestChunkPayload"]: ...


@runtime_checkable
class ResultParserPlugin(Protocol):
    name: str

    def can_parse(self, path: Path) -> bool: ...
    def parse(self, path: Path, options: ParseOptions) -> ParsedResults: ...
```

Also remove `set_chunking()` and `has_chunking` from `ContextAwareRobotResults` (no external caller left after C1.4).

**File**: `result_companion/core/plugins/registry.py`

Replace `_is_parser_plugin` attribute check with `isinstance(plugin, ResultParserPlugin)`:

```python
def _is_parser_plugin(plugin: Any) -> bool:
    return isinstance(plugin, ResultParserPlugin)
```

**Constraints**:
- After this commit, grep `set_chunking` and `has_chunking` in the repo → must yield only test files updated to remove references. If any production file still uses them, this commit is not complete.
- `ChunkingStrategy` class stays alive in `chunking.py` for the migration; deletion is out of scope for M1.

**Test**:
- `tests/core/plugins/test_registry.py::test_isinstance_check_rejects_non_plugin_object`
- Update `tests/core/plugins/test_robot.py` — remove any calls to `set_chunking` / `has_chunking`.
- Full test suite green.

### C1.6 — Migrate `examples/plugins/pytest_junit/` to new shape

**File**: `examples/plugins/pytest_junit/result_companion_pytest_junit/plugin.py`

Remove `has_chunking` property, `set_chunking()` method, and internal `self._chunking` storage. Replace `render_chunks()` body:

```python
from result_companion.core.chunking.helpers import single_chunk

def render_chunks(self) -> Iterator[TestChunkPayload]:
    for case in self._selected_cases():
        lines = _render_case(case, self._options.exclude_fields or [])
        rendered = render_lines_to_text(lines)
        chunks, chunk_stats = single_chunk(rendered, self._options.tokenizer_config)
        yield TestChunkPayload(case.name, chunks, chunk_stats, case.status)
```

**Constraints**:
- Do **not** delete this example. It still serves as the reference "how to write a plugin" doc target for now.
- If `single_chunk` returns empty list (empty text) → skip the test rather than yielding an empty payload.

**Test**: `examples/plugins/pytest_junit/tests/test_plugin.py` — update to construct `ParseOptions(tokenizer_config=...)`, assert `render_chunks()` yields expected payloads.

### C1.7 — Docs: refresh `PLUGIN_ARCHITECTURE.md`

**File**: `examples/PLUGIN_ARCHITECTURE.md`

Changes:
1. Remove all mentions of `has_chunking`, `set_chunking`, `ChunkingStrategy` from the plugin author sections.
2. Add "How chunking works" note: "The CLI populates `ParseOptions.tokenizer_config` and `ParseOptions.system_prompt`. Plugins call `token_aware_chunk(lines, tokenizer_config, system_prompt)` from `result_companion.core.chunking.helpers` if they want context-aware chunking, or `single_chunk(text, tokenizer_config)` if per-test output always fits."
3. Add full `render_html_report` signature to the "Optional Capabilities" section:

```python
def render_html_report(
    self,
    input_path: Path,
    output_path: Path,
    llm_results: dict[str, str],
    model_info: dict[str, str] | None = None,
    overall_summary: str | None = None,
) -> None:
```

**Test**: None — docs only.

## M1 Acceptance Criteria

- [ ] `grep -R "set_chunking\|has_chunking" result_companion/` → no matches.
- [ ] `grep -R "set_chunking\|has_chunking" examples/plugins/` → no matches.
- [ ] `pytest` full suite green.
- [ ] `rc analyze -o tests/fixtures/output.xml --dryrun` produces identical output to the pre-M1 branch (byte-diff the JSON report on a fixture).
- [ ] `ResultParserPlugin` and `ParsedResults` are `@runtime_checkable`.

## M1 Test Plan Summary

| Test file | New tests |
|-----------|-----------|
| `tests/core/plugins/test_base.py` | Defaults of new `ParseOptions` fields |
| `tests/core/chunking/test_helpers.py` | 4 unit tests for `token_aware_chunk` / `single_chunk` |
| `tests/core/chunking/test_rf_results.py` | 1 test for `ValueError` when `tokenizer_config` missing |
| `tests/core/plugins/test_registry.py` | 1 test for `isinstance` rejection of non-plugin |
| `examples/plugins/pytest_junit/tests/test_plugin.py` | Update to new options shape |

---

# Milestone M2 — JUnit Escape Hatch (Bundled)

## Rationale

`plugin-scope.md` frames JUnit as a **~200 LOC courtesy for mixed Robot + pytest teams**. Move the existing `examples/plugins/pytest_junit/` into `result_companion/core/plugins/`, register in built-ins, ship no separate install. Reference clean Protocol from M1 — no `set_chunking`, no `has_chunking`, no `ChunkingStrategy`.

Runs only **after M1** merges. Independent of the vision plan.

## Files Touched

| File | Change type |
|------|-------------|
| `result_companion/core/plugins/pytest_junit.py` | **New** — migrated from `examples/plugins/pytest_junit/…/plugin.py` |
| `result_companion/core/plugins/registry.py` | Register in `get_builtin_plugins()` |
| `tests/core/plugins/test_pytest_junit.py` | New |
| `tests/fixtures/junit/pytest_junit_sample.xml` | New — small realistic pytest JUnit output |
| `examples/plugins/pytest_junit/` | Deleted or converted to a `README.md` note: "This plugin is now bundled — see result_companion.core.plugins.pytest_junit" |
| `README.md` | Small note: "Supports Robot Framework and JUnit XML (pytest, etc.) via bundled built-in `--format pytest-junit`" |

## Commits

### C2.1 — Move plugin file into core

Copy `examples/plugins/pytest_junit/result_companion_pytest_junit/plugin.py` to `result_companion/core/plugins/pytest_junit.py`. Adjust imports:

```python
from result_companion.core.chunking.helpers import single_chunk
from result_companion.core.chunking.chunking import render_lines_to_text
from result_companion.core.plugins.base import ParseOptions, TestChunkPayload
```

Remove the `ChunkingStrategy` import — it's no longer used post-M1.

**Constraints**:
- File **must** already be M1-shape (no `set_chunking`, no `has_chunking`). Since M1's C1.6 migrated the example, this move is essentially `git mv` + import path fixup.
- Do **not** move fixtures or tests from the example directory — write fresh ones in `tests/core/plugins/`.

### C2.2 — Register as built-in

**File**: `result_companion/core/plugins/registry.py`

```python
def get_builtin_plugins() -> tuple[ResultParserPlugin, ...]:
    """Returns built-in parser plugins."""
    from result_companion.core.plugins.pytest_junit import PytestJUnitPlugin

    return (RobotPlugin(), PytestJUnitPlugin())
```

Import inside the function to avoid a top-level cycle (pytest_junit imports from chunking → chunking has no deps back). Static top-level import is preferable if the cycle is absent — verify with a quick `python -c "from result_companion.core.plugins.registry import get_builtin_plugins"`.

**Constraints**:
- Do **not** rename the plugin identifier. Keep `name = "pytest-junit"` (kebab-case, per existing example).
- Dedup logic already exists in `_deduplicate_plugins`. No further change.

**Test**: `tests/core/plugins/test_registry.py::test_get_builtin_plugins_includes_pytest_junit`.

### C2.3 — Fail-early on unsupported options

**Constraint from `plugin-scope.md`**: `analyze --format` must error clearly when the user requests a capability the plugin does not support.

**File**: `result_companion/core/plugins/pytest_junit.py`

Ensure the `parse()` method (or a new `supports_tag_filters = False` attribute + `registry.validate_options`) rejects `-I/-E` before parsing. The existing `validate_options` in `registry.py` already handles this via the `supports_tag_filters` attribute — confirm `PytestJUnitPlugin` does **not** declare it (so it defaults to `False`) and that the error message is clear:

```
Format 'pytest-junit' does not support --include/--exclude tag filters.
```

**Test**: `tests/core/plugins/test_pytest_junit.py::test_tag_filter_rejected_with_clear_error`.

### C2.4 — Fixtures + unit tests

**File**: `tests/fixtures/junit/pytest_junit_sample.xml`

Minimal but realistic. Two testcases: one PASS, one FAIL with `<failure message="AssertionError">` and text body. Under 2 KB.

**File**: `tests/core/plugins/test_pytest_junit.py`

Tests (aim for 6-8):
- `test_can_parse_returns_true_for_junit_xml`
- `test_can_parse_returns_false_for_robot_output`
- `test_parse_yields_one_case_per_testcase_element`
- `test_parse_marks_failed_case_as_fail`
- `test_render_chunks_yields_one_chunk_per_selected_case`
- `test_exclude_passing_filters_out_pass_cases`
- `test_tag_filter_rejected_with_clear_error`
- `test_source_hash_stable_across_runs_same_input`

Use factory functions for `ParseOptions` (per `AGENTS.md`).

### C2.5 — Delete old example dir; docs

Delete `examples/plugins/pytest_junit/` (or reduce to a README pointer). Update:

- `README.md` — one line: "Supports Robot Framework output.xml and JUnit XML (pytest, etc.). Use `--format pytest-junit` for JUnit."
- `examples/PLUGIN_ARCHITECTURE.md` — refresh the "Writing a Plugin" walkthrough to reference the bundled `pytest_junit.py` as the reference example.

## M2 Acceptance Criteria

- [ ] `rc analyze -o tests/fixtures/junit/pytest_junit_sample.xml -f pytest-junit --dryrun` produces a text report with the FAIL case included.
- [ ] `rc analyze -o tests/fixtures/junit/pytest_junit_sample.xml -f pytest-junit -I smoke` exits with an error message naming the plugin and the unsupported flag.
- [ ] Auto-detection: `rc analyze -o tests/fixtures/junit/pytest_junit_sample.xml` (no `-f`) picks `pytest-junit` because RobotPlugin's `can_parse` returns False for JUnit XML.
- [ ] All 6-8 unit tests green.
- [ ] `examples/plugins/pytest_junit/` no longer contains active plugin code.

## M2 Test Plan Summary

| Layer | Tests |
|-------|-------|
| Plugin | 6-8 unit tests |
| Registry | 1 test |
| CLI e2e (dryrun) | 1 fixture-based test |

---

# Backward Compatibility & Rollback

- **M1** ships behavior-preserving. No user-facing change.
- **M2** adds `--format pytest-junit` — new capability, does not affect existing Robot users.
- Each PR is independently revertable. M2 revert is `git revert` of the registry commit.

## Config File Migration

**None required.** No new config keys introduced by this plan. Existing user YAMLs remain valid.

---

# Deferred / Not Doing Yet

Deliberately not included in this plan (per `plugin-scope.md` non-goals):

- `Capabilities` dataclass — postpone until a third plugin proves the need.
- `api_version` on plugins — same.
- Renderer / reporter planes — YAGNI. Add only if HTML template rot becomes real.
- Conformance test kit — write when the third-party plugin ecosystem materializes.
- REVIEW.md F3, F4, F6, F7, F8, F10, F11, F12, F13 — remain as tech debt entries; schedule when they bite.

---

# Reference

- [`docs/plugin-scope.md`](plugin-scope.md) — honest scope, priorities, non-goals.
- [`docs/robot-vision-plan.md`](robot-vision-plan.md) — companion plan for the Robot Framework vision pipeline.
- [`AGENTS.md`](../AGENTS.md) — project coding standards (test style, docstrings, nesting, complexity, dependency injection).
- [`examples/PLUGIN_ARCHITECTURE.md`](../examples/PLUGIN_ARCHITECTURE.md) — plugin author guide (updated in C1.7).
- [`examples/PLUGIN_ARCHITECTURE_REVIEW.md`](../examples/PLUGIN_ARCHITECTURE_REVIEW.md) — the F1–F13 findings; this plan schedules F1/F2/F5/F9 in M1 and defers the rest.
- [`result_companion/core/plugins/base.py`](../result_companion/core/plugins/base.py) — Protocol.
- [`result_companion/core/plugins/registry.py`](../result_companion/core/plugins/registry.py) — plugin discovery.
- [`result_companion/entrypoints/run_rc.py`](../result_companion/entrypoints/run_rc.py) — CLI orchestrator, primary integration point.
