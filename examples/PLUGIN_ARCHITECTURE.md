# Parser Plugins

## Quick Read

Result Companion uses parser plugins to load test result artifacts before chunking and LLM analysis. This first plugin step supports built-in, project-local plugins only.

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

## Built-In Registry

Built-in plugins live under [`result_companion/core/plugins`](../result_companion/core/plugins).

The registry resolves a plugin in two ways:

- Explicit format: `--format robot`
- Auto-detection: plugin `can_parse(path)` returns `True`

Current built-in format:

| Format | Plugin | Tag Filters |
|--------|--------|-------------|
| `robot` | `RobotPlugin` | Supported |

External Python package entry points are not enabled yet. Add custom parsers inside this repository and register them in the built-in registry.

## Plugin Contract

Implement the parser protocol from [`base.py`](../result_companion/core/plugins/base.py):

```python
from pathlib import Path

from result_companion.core.chunking.rf_results import ContextAwareRobotResults
from result_companion.core.plugins.base import ParseOptions


class MyPlugin:
    name = "my-format"
    capabilities = frozenset()

    def can_parse(self, path: Path) -> bool:
        return path.suffix == ".mylog"

    def parse(self, path: Path, options: ParseOptions) -> ContextAwareRobotResults:
        raise NotImplementedError
```

For this first version, plugins return `ContextAwareRobotResults` or an object with the same methods used by the analysis pipeline:

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

## Add a Project-Local Plugin

Create a plugin module:

```python
# result_companion/core/plugins/my_format.py
from pathlib import Path

from result_companion.core.plugins.base import ParseOptions


class MyFormatPlugin:
    name = "my-format"
    capabilities = frozenset()

    def can_parse(self, path: Path) -> bool:
        return path.suffix == ".mylog"

    def parse(self, path: Path, options: ParseOptions):
        return parse_my_format(path, options)
```

Register it in [`registry.py`](../result_companion/core/plugins/registry.py):

```python
from result_companion.core.plugins.my_format import MyFormatPlugin
from result_companion.core.plugins.robot import RobotPlugin


def get_builtin_plugins():
    return (RobotPlugin(), MyFormatPlugin())
```

Add tests for:

- explicit format resolution
- auto-detection
- parse option forwarding
- capability validation for unsupported tag filters

## Programmatic Usage

Use custom built-in-style plugins directly with the Python API:

```python
from result_companion import analyze
from result_companion.core.parsers.config import load_config
from result_companion.core.plugins.my_format import MyFormatPlugin


config = load_config("my_config.yaml")
result = analyze(
    "results.mylog",
    config=config,
    result_format="my-format",
    plugins=[MyFormatPlugin()],
)
```

When `plugins` is omitted, `analyze()` uses the built-in registry.
