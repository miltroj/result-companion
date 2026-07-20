# Plugin Architecture — Review & Split Plan

## Quick Read

Findings and rework plan for [`PLUGIN_ARCHITECTURE.md`](PLUGIN_ARCHITECTURE.md). Current design is ~70% there but leaks Robot Framework shape into a "generic" plugin API and misfires on high-volume log use cases. This doc catalogs gaps and proposes a small-PR rollout.

## Table of Contents

- [Findings](#findings)
- [Robot vs Other Frameworks](#robot-vs-other-frameworks)
- [High-Volume Log Concerns](#high-volume-log-concerns)
- [Doc Bugs in PLUGIN_ARCHITECTURE.md](#doc-bugs-in-plugin_architecturemd)
- [Split Plan (Sequential PRs)](#split-plan-sequential-prs)
- [Reference](#reference)

## Findings

Grouped by severity. Lines cite current implementation.

| # | Area | Severity | Summary |
|---|------|----------|---------|
| F1 | `run_rc.py` `set_chunking` unguarded | High | CLI forces RF-shaped `ChunkingStrategy` on any plugin; API path guards it. Real bug for pre-chunked non-RF plugins. |
| F2 | `ChunkingStrategy` leaks `RenderLine` | High | Protocol pretends generic; `apply()` needs RF-shaped depth-line tree. Non-RF authors have no clean path. |
| F3 | `source_hash` re-renders whole suite | High | For big logs hashes a huge string just to discard it. Should hash file bytes + option toggles. |
| F4 | `total_test_count` semantics undocumented | Medium | RF version = pre-filter; `test_names` = post-filter. Non-RF plugins will diverge. |
| F5 | `render_html_report` signature not in doc | Medium | Authors must grep `robot.py` to learn the 5-param contract. |
| F6 | Cheap `can_parse` guidance missing | Medium | With N plugins installed, all can open the file. Doc must recommend extension → magic bytes → full parse. |
| F7 | Example `MyParsedResults` uses class-level mutable attrs | Medium | Cargo-cult footgun; new plugins share state across instances. |
| F8 | `Chunking(0, 0, 8, 2, 1)` example is opaque | Medium | Magic numbers, no field-by-field explanation anywhere. |
| F9 | `has_chunking` attr vs `@property` inconsistency | Low | Protocol silent; example uses attr, RF uses property. |
| F10 | No error contract for `parse()` | Low | Corrupt file: raise what? Doc silent. |
| F11 | Ancestor breadcrumb collection is O(N²) | Low | `_collect_ancestor_context_at` walks back through full line list per line. Hurts deep 50k-line runs. |
| F12 | Registry uses attribute duck-check | Low | `_is_parser_plugin` could use `@runtime_checkable` Protocol + `isinstance()`. |
| F13 | No pre-flight cost estimation | Low | Users have no way to see "this run = ~500k tokens, ~$X" before LLM burn. |

### F1 evidence

CLI unguarded:

```69:73:result_companion/entrypoints/run_rc.py
    results = plugin.parse(output, options)
    strategy = ChunkingStrategy(
        tokenizer_config=parsed_config.tokenizer,
        system_prompt=parsed_config.llm_config.question_prompt,
    )
    results.set_chunking(strategy)
```

API guarded:

```120:125:result_companion/api.py
    if not results.has_chunking:
        strategy = ChunkingStrategy(
            tokenizer_config=config.tokenizer,
            system_prompt=config.llm_config.question_prompt,
        )
        results.set_chunking(strategy)
```

### F2 evidence

Protocol claims generic:

```30:34:result_companion/core/plugins/base.py
    def set_chunking(self, strategy: ChunkingStrategy) -> "ParsedResults":
        """Attaches a chunking strategy."""

    def render_chunks(self) -> Iterator["TestChunkPayload"]:
        """Yields chunked test payloads for LLM analysis."""
```

Strategy is RF-shaped:

```342:348:result_companion/core/chunking/chunking.py
    def apply(self, lines: list[RenderLine]) -> tuple[list[str], Chunking]:
        """Chunks rendered lines, sizing based on token budget."""
        rendered = render_lines_to_text(lines)
        chunk_info = calculate_chunk_size(
            rendered, self.system_prompt, self.tokenizer_config
        )
        return chunk_rf_test_lines(lines, chunk_info.chunk_size), chunk_info
```

### F3 evidence

```178:182:result_companion/core/chunking/rf_results.py
    @cached_property
    def source_hash(self) -> str:
        """Short SHA-256 hash of the rendered suite for reproducibility tracking."""
        blob = str(self).encode()
        return hashlib.sha256(blob).hexdigest()[:12]
```

## Robot vs Other Frameworks

RC targets high-volume log analysis. Other frameworks don't share RF's shape.

| Framework | Structure | Tags | Log Volume |
|-----------|-----------|------|-----------|
| Robot Framework | Nested suites → tests → keywords → messages | Native, wildcard | High (keyword tree) |
| pytest (junit / json-report) | Flat testcase list + captured stdout/stderr | Markers | Medium |
| JUnit XML | testsuites → testsuite → testcase | Properties | Low-medium |
| TestNG | XML with configuration methods, groups | Groups | Medium |
| Playwright / Cypress | JSON + attachments (screenshots, video refs) | Tags/annotations | High (traces) |

Implications:

- **Breadcrumb chunking is RF-specific**. Flat frameworks (pytest, JUnit) don't need depth-aware ancestor context. Forcing them through `RenderLine` is friction with no payoff.
- **Chunking should be per-plugin**, not a core-owned protocol method. Core provides a helper (`token_aware_chunk(text, tokenizer, budget)`); plugins call it as they see fit.

## High-Volume Log Concerns

RC's differentiator is big logs, yet several core paths fight scale:

1. **Eager XML load** — `ExecutionResult(source)` reads whole tree into memory. Non-RF plugins will copy this. Doc should encourage lazy `parse()` with streaming access to tests.
2. **Double rendering** — `_iter_tests` builds `list[RenderLine]`, chunker re-walks lines. 2× memory per test. For big keyword bodies (stack traces, JSON payloads), spike risk.
3. **`source_hash` full re-render** — see F3. Fix: hash file bytes + relevant option toggles.
4. **O(N²) ancestor walk** — see F11. Precompute a `depth → last_line_idx` cursor as lines are appended.
5. **No pre-flight** — RC could estimate token cost before spending. High-volume users need this to avoid surprise bills.

## Doc Bugs in PLUGIN_ARCHITECTURE.md

Direct fixes to [`PLUGIN_ARCHITECTURE.md`](PLUGIN_ARCHITECTURE.md):

1. Rewrite `MyParsedResults` example — use `__init__` for instance attrs, not class-level mutable state.
2. Add a **Field Reference** section documenting `TestChunkPayload` fields and `Chunking` fields (`chunk_size`, `number_of_chunks`, `raw_text_len`, `tokens_from_raw_text`, `tokenized_chunks`).
3. Add full `render_html_report(input_path, output_path, llm_results, model_info=None, overall_summary=None)` signature.
4. Add **`can_parse` cost guidance** — recommend extension check first, magic-bytes second, full parse last.
5. Pin **`total_test_count` semantics** — pre-filter (all tests in the artifact) vs post-filter (after tags/exclude_passing). Pick one, document it. RF's current answer: pre-filter, ignores exclude_passing.
6. Add **`source_hash` contract** — hash of _what_? File bytes + relevant options is the cheap, portable choice.
7. Add **error contract for `parse()`** — plugins raise `ValueError` on unparseable input; registry surfaces error with plugin name.
8. Add **High-Volume Logs** section — laziness expectations, memory budget, generator discipline.

## Split Plan (Sequential PRs)

Feature developed on a separate branch; merged as one drop to `main` once tested. No backward-compat window needed — no external plugins exist yet.

Small PRs, each independently reviewable and shippable. Order matters only where noted.

| PR | Title | Depends on | Scope | Risk |
|----|-------|------------|-------|------|
| PR1 | Fix `run_rc.py` `set_chunking` guard | — | 1 line + 1 test | Low |
| PR2 | Cheap `source_hash` (streamed bytes + options) | — | RF plugin only; update tests | Low |
| PR3 | Doc surgery on `PLUGIN_ARCHITECTURE.md` (non-contract fixes) | — | Text-only; add Field Reference + High-Volume section | None |
| PR4 | Reference non-RF plugin (pytest-junit) against current contract | PR3 | Example plugin in `examples/`; surfaces contract holes for PR5 | Low |
| PR5 | Decouple chunking from `ParsedResults` Protocol + `@runtime_checkable` cleanup | PR3, PR4 | Remove `set_chunking`/`has_chunking` from Protocol; move token-chunking to helper; registry uses `isinstance`; docs updated in same PR; PR4 example migrated | Medium (public API shape) |
| PR6 | O(N) ancestor context in RF chunker | — | Perf-only in `chunking.py` | Low |
| PR7 | Pre-flight cost estimation (optional capability) | PR5 | Duck-typed `estimate_tokens()` — not Protocol; CLI flag `--estimate` | Low |

### PR1 — Fix CLI/API chunking mismatch

- **Change**: Wrap `set_chunking` call in `if not results.has_chunking:`.
- **Test**: `run_rc` with a fake plugin whose `parse()` returns a pre-chunked result → no double-init.
- **Why first**: 1-line real bug, unblocks non-RF plugin authors.

### PR2 — Cheap `source_hash`

- **Change**: RF plugin streams file bytes (avoid loading multi-MB logs into memory) plus a stable repr of parse options:

```python
h = hashlib.sha256()
with path.open("rb") as f:
    for chunk in iter(lambda: f.read(65536), b""):
        h.update(chunk)
h.update(options_repr.encode())
return h.hexdigest()[:12]
```

- **Test**: Hash stable across runs; hash differs when tags/exclude_passing differ; verify on a multi-MB fixture that memory stays bounded.
- **Why**: Kills a full re-render on every JSON report emit; respects the "High-Volume Logs" concern this doc raises.

### PR3 — Doc surgery (non-contract fixes only)

- **Change**: Apply the 8 bullets from [Doc Bugs](#doc-bugs-in-plugin_architecturemd) that are orthogonal to Protocol shape (mutable-attr fix, Field Reference, `render_html_report` signature, `can_parse` cost guidance, `total_test_count` semantics, `source_hash` contract, `parse()` error contract, High-Volume section). Protocol-shape sections (`set_chunking`, `has_chunking`) are rewritten in PR5, when the shape actually changes, to avoid docs contradicting themselves.
- **Test**: N/A.
- **Why**: Cheap; unblocks future plugin authors on everything except the Protocol churn.

### PR4 — Reference non-RF plugin (pytest-junit) as canary

- **Change**: Ship `examples/plugins/pytest_junit/` — a minimal pytest JUnit XML plugin implemented against the **current** contract.
- **Test**: End-to-end: parse → chunk → dry-run analyze.
- **Why**: Deliberate canary for PR5. Writing a real non-RF plugin surfaces exactly where the current Protocol (`ChunkingStrategy` taking `list[RenderLine]`, `set_chunking` on Protocol) doesn't fit. That evidence directly informs PR5's decouple.

### PR5 — Decouple chunking + runtime-checkable Protocol

- **Change**:
  - Drop `set_chunking` and `has_chunking` from `ParsedResults` Protocol.
  - Move token-aware chunking into `result_companion.core.chunking.helpers.token_aware_chunk(text, tokenizer_config, system_prompt)`.
  - RF plugin's `parse()` returns a `ContextAwareRobotResults` that invokes the helper internally from `render_chunks()`.
  - `api.py` / `run_rc.py` stop calling `set_chunking`; they pass `tokenizer_config` + `system_prompt` via `ParseOptions`.
  - Add `@runtime_checkable` to `ResultParserPlugin`; replace `_is_parser_plugin` attr check with `isinstance(plugin, ResultParserPlugin)`.
  - Update `PLUGIN_ARCHITECTURE.md` Protocol shape sections in the same PR (no stale `set_chunking`/`has_chunking` docs left behind).
  - Migrate PR4's pytest-junit example to the new contract.
- **Test**: Existing RF tests green; PR4's pytest-junit plugin returns chunks via the helper without touching `ChunkingStrategy`; entry-point loader rejects non-conforming plugin object with a warning.
- **Why**: Removes the biggest lie in the current contract; PR4 already proved a non-RF shape can drive the design.

### PR6 — O(N) ancestor context

- **Change**: In `chunk_rf_test_lines`, maintain a `depth → last_ancestor_line` dict as lines are consumed; drop `_collect_ancestor_context_at`'s backward scan.
- **Test**: Same chunk outputs on existing fixtures; add a perf test that asserts **operation count** (line traversals) grows linearly with input size — avoids CI wall-clock flake.
- **Why**: Perf; safety for very large keyword bodies.

### PR7 — Pre-flight cost estimation (optional capability)

- **Change**:
  - Add `estimate_tokens() -> TokenEstimate` as a **duck-typed capability** (like `render_html_report`) — not on the `ParsedResults` Protocol. Plugins that don't care about tokens don't implement it.
  - Shape: `@dataclass(frozen=True) class TokenEstimate: total_tokens: int; per_test: dict[str, int]`.
  - RF impl: sum per-test rendered token count via configured tokenizer.
  - CLI: `--estimate` prints total tokens + rough chunk count and exits; errors clearly when the selected plugin doesn't implement `estimate_tokens`.
- **Test**: Estimate within ±5% of observed usage on a fixture; CLI errors gracefully on plugin without the capability.
- **Why**: High-volume users demand cost awareness before LLM spend, without re-coupling every plugin to tokenizer internals.

## Reference

- [`PLUGIN_ARCHITECTURE.md`](PLUGIN_ARCHITECTURE.md) — current architecture
- [`result_companion/core/plugins/base.py`](../result_companion/core/plugins/base.py) — Protocol definitions
- [`result_companion/core/plugins/registry.py`](../result_companion/core/plugins/registry.py) — plugin discovery
- [`result_companion/core/plugins/robot.py`](../result_companion/core/plugins/robot.py) — RF plugin (reference impl)
- [`result_companion/core/chunking/rf_results.py`](../result_companion/core/chunking/rf_results.py) — `ContextAwareRobotResults`
- [`result_companion/core/chunking/chunking.py`](../result_companion/core/chunking/chunking.py) — `ChunkingStrategy`, chunker
- [`result_companion/entrypoints/run_rc.py`](../result_companion/entrypoints/run_rc.py) — CLI entry
- [`result_companion/api.py`](../result_companion/api.py) — programmatic entry
