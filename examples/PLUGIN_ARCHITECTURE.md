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

from result_companion.core.plugins.base import AnalysisResults, ParseOptions


class MyPlugin:
    name = "my-format"
    capabilities = frozenset()

    def can_parse(self, path: Path) -> bool:
        return path.suffix == ".mylog"

    def parse(self, path: Path, options: ParseOptions) -> AnalysisResults:
        raise NotImplementedError
```

Plugins return an `AnalysisResults` object. `ContextAwareRobotResults` is the built-in Robot Framework implementation, but custom plugins can return any object that satisfies the `AnalysisResults` protocol:

- `test_names`
- `total_test_count`
- `source_hash`
- `has_chunking`
- `set_chunking(strategy)`
- `render_chunks()`

## Capabilities

Declare supported optional behavior in `capabilities`.

Use `TAG_FILTERS` when the artifact has real tag data:

```python
from result_companion.core.plugins.base import TAG_FILTERS


class MyTaggedPlugin:
    name = "my-tagged-format"
    capabilities = frozenset({TAG_FILTERS})
```

If a plugin does not declare `TAG_FILTERS`, the registry rejects:

```bash
result-companion analyze -o results.xml --format my-format --include smoke
```

## Add an Installable Plugin

Create a plugin module:

```python
# my_package/plugin.py
from pathlib import Path

from result_companion.core.plugins.base import AnalysisResults, ParseOptions


class MyFormatPlugin:
    name = "my-format"
    capabilities = frozenset()

    def can_parse(self, path: Path) -> bool:
        return path.suffix == ".mylog"

    def parse(self, path: Path, options: ParseOptions) -> AnalysisResults:
        return parse_my_format(path, options)
```

Register the plugin in your package `pyproject.toml`:

```toml
[tool.poetry.plugins."result_companion.plugins"]
my-format = "my_package.plugin:MyFormatPlugin"
```

Install it in the same environment as Result Companion:

```bash
pip install result-companion-my-format
result-companion analyze -o results.mylog --format my-format
```

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
    plugins=[MyFormatPlugin()],
)
```

When `plugins` is omitted, `analyze()` uses built-in and installed plugins.
