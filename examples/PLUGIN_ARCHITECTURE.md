# Parser Plugins

## Quick Read

Result Companion uses parser plugins to load test result artifacts before chunking and LLM analysis. Install plugin packages to add formats without changing Result Companion.

Use `--format` to select a parser explicitly, or omit it to auto-detect from the input file.

## CLI Usage

Auto-detect Robot Framework output:

```bash
result-companion analyze -o output.xml
```

Select the Robot parser explicitly:

```bash
result-companion analyze -o output.xml --format robot
```

Robot supports tag filters:

```bash
result-companion analyze -o output.xml --format robot --include smoke --exclude wip
```

Formats without tag support must reject `--include` and `--exclude` instead of ignoring them.

HTML reports are optional. Implement `render_html_report` on your plugin to enable
`--html-report` and `--report`; otherwise use `--no-html-report` with
`--text-report`, `--json-report`, or `--print-text-report`.

## Plugin Registry

Built-in plugins live under [`result_companion/core/plugins`](../result_companion/core/plugins).

The registry loads built-in plugins plus installed plugins from the `result_companion.plugins` entry-point group. It resolves a plugin in two ways:

- Explicit format: `--format robot`
- Auto-detection: plugin `can_parse(path)` returns `True`

Current built-in format:

| Format | Plugin | Tag Filters |
|--------|--------|-------------|
| `robot` | `RobotPlugin` | Supported |

Use debug logs to inspect plugin discovery:

```bash
result-companion analyze -o results.mylog --format my-format --log-level DEBUG
```

## Plugin Contract

Implement the parser protocol from [`base.py`](../result_companion/core/plugins/base.py):

```python
from pathlib import Path

from result_companion.core.chunking.utils import Chunking
from result_companion.core.plugins.base import (
    ParseOptions,
    ParsedResults,
    TestChunkPayload,
)


class MyParsedResults:
    def __init__(self) -> None:
        self.test_names = ["example"]
        self.total_test_count = 1
        self.source_hash = "0" * 12
        self.has_chunking = False

    def set_chunking(self, strategy: object) -> "MyParsedResults":
        self.has_chunking = True
        return self

    def render_chunks(self):
        yield TestChunkPayload(
            test_name="example",
            chunks=["log body"],
            chunk_stats=Chunking(
                chunk_size=0,
                number_of_chunks=0,
                raw_text_len=8,
                tokens_from_raw_text=2,
                tokenized_chunks=1,
            ),
            status="FAIL",
        )


class MyPlugin:
    name = "my-format"

    def can_parse(self, path: Path) -> bool:
        return path.suffix == ".mylog"

    def parse(self, path: Path, options: ParseOptions) -> ParsedResults:
        if not path.exists():
            raise ValueError(f"File does not exist: {path}")
        return MyParsedResults()
```

Plugin `name` is matched case-insensitively.

Plugins return a `ParsedResults` object. `ContextAwareRobotResults` is the built-in Robot Framework implementation, but custom plugins can return any object that satisfies the `ParsedResults` protocol:

- `test_names`
- `total_test_count`
- `source_hash`
- `has_chunking`
- `set_chunking(strategy)`
- `render_chunks()`

### Field Reference

`ParsedResults` fields:

- `test_names`: Names of tests selected for analysis after parser filters are applied.
- `total_test_count`: Number of tests in the parsed artifact after tag filters, but before `exclude_passing`.
- `source_hash`: Stable 12-character SHA-256 prefix for the source bytes plus parse options that affect rendered output.
- `has_chunking`: `True` when `set_chunking()` has attached a chunking strategy.

`TestChunkPayload` fields:

- `test_name`: Test or analysis-unit name shown in reports.
- `chunks`: Text chunks sent to the LLM for this test.
- `chunk_stats`: `Chunking` metadata for the rendered text.
- `status`: Source test status, for example `PASS`, `FAIL`, or `SKIP`.

`Chunking` fields:

- `chunk_size`: Character budget used per chunk; `0` means no split was needed.
- `number_of_chunks`: Number of chunks created when splitting was needed.
- `raw_text_len`: Length of rendered text plus system prompt.
- `tokens_from_raw_text`: Token count for rendered text plus system prompt.
- `tokenized_chunks`: Expected chunk count from token budgeting.

`source_hash` should avoid rendering large suites when a file path is available. Prefer streaming
file bytes and hashing only parse options that change output, such as tag filters,
excluded fields, and `exclude_passing`.

`parse()` should raise `ValueError` for unparseable input, unsupported parse options, or corrupt
artifacts. Include the file path and plugin format in the message when useful; the registry will
surface the failure to the caller.

## Capabilities

Optional features are duck-typed by attribute or method presence.

Set `supports_tag_filters = True` when the artifact has real tag data:

```python
class MyTaggedPlugin:
    name = "my-tagged-format"
    supports_tag_filters = True
```

If the plugin does not set `supports_tag_filters = True`, the registry rejects:

```bash
result-companion analyze -o results.xml --format my-format --include smoke
```

To support HTML logs, implement `render_html_report` on the plugin. Plugins that do not
implement it still work with text, JSON, printed text, and programmatic results.

Signature:

```python
def render_html_report(
    self,
    input_path: Path,
    output_path: Path,
    llm_results: dict[str, str],
    model_info: dict[str, str] | None = None,
    overall_summary: str | None = None,
) -> None:
    ...
```

`can_parse(path)` runs during auto-detection and may be called on every installed plugin. Keep it
cheap: check extension first, read only enough bytes for a magic-header/root-element check second,
and do a full parse only inside `parse()`.

## High-Volume Logs

Plugins should be cheap on large artifacts:

- Keep `can_parse()` bounded; do not load full logs during auto-detection.
- Stream source bytes for `source_hash` when possible.
- Generate chunks lazily from `render_chunks()` instead of building all LLM payloads up front.
- Avoid duplicate rendered copies of large test logs.
- Raise `ValueError` early when options would force unsupported expensive behavior.

## Add an Installable Plugin

Result Companion discovers installed plugins with `importlib.metadata.entry_points()` from the
`result_companion.plugins` group. Plugin packages declare an entry point; installation writes it to
`site-packages/<package>.dist-info/entry_points.txt`.

Use a small package layout:

```text
result-companion-my-format/
  pyproject.toml
  my_package/
    __init__.py
    plugin.py
```

Create the plugin module:

```python
# my_package/plugin.py
from pathlib import Path

from result_companion.core.plugins.base import ParseOptions, ParsedResults


class MyFormatPlugin:
    name = "my-format"

    def can_parse(self, path: Path) -> bool:
        return path.suffix == ".mylog"

    def parse(self, path: Path, options: ParseOptions) -> ParsedResults:
        return parse_my_format(path, options)
```

Register the plugin with Poetry:

```toml
[tool.poetry.plugins."result_companion.plugins"]
my-format = "my_package.plugin:MyFormatPlugin"
```

Or with PEP 621 project metadata:

```toml
[project.entry-points."result_companion.plugins"]
my-format = "my_package.plugin:MyFormatPlugin"
```

Install it in the same environment as Result Companion:

```bash
pip install result-companion-my-format
result-companion analyze -o results.mylog --format my-format
```

Verify discovery:

```bash
python -c "from importlib.metadata import entry_points; print(list(entry_points(group='result_companion.plugins')))"
```

Common install issues:

- Result Companion and the plugin must run in the same Python environment.
- Reinstall the plugin after changing entry points in `pyproject.toml`.
- Duplicate plugin names are de-duplicated; the first discovered plugin wins.

Add tests for:

- explicit format resolution
- auto-detection
- parse option forwarding
- capability validation for unsupported tag filters

## Programmatic Usage

For tests or one-off scripts, pass plugins directly:

```python
from result_companion import analyze
from result_companion.core.parsers.config import load_config
from my_package.plugin import MyFormatPlugin


config = load_config("my_config.yaml")
result = analyze(
    "results.mylog",
    config=config,
    result_format="my-format",
    parser_plugins=[MyFormatPlugin()],
)
```

When `parser_plugins` is omitted, `analyze()` uses built-in and installed plugins.
