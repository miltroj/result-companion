from __future__ import annotations

from pathlib import Path
from typing import Sequence

from result_companion.core.chunking.rf_results import ContextAwareRobotResults
from result_companion.core.plugins.base import (
    TAG_FILTERS,
    ParseOptions,
    ResultParserPlugin,
)
from result_companion.core.plugins.robot import RobotPlugin


def get_builtin_plugins() -> tuple[ResultParserPlugin, ...]:
    """Returns built-in parser plugins."""
    return (RobotPlugin(),)


def get_plugin(
    format_name: str | None,
    path: Path,
    plugins: Sequence[ResultParserPlugin] | None = None,
) -> ResultParserPlugin:
    """Resolves a parser plugin by explicit format or auto-detection."""
    available_plugins = tuple(plugins or get_builtin_plugins())
    if format_name:
        return _get_plugin_by_name(format_name, available_plugins)
    return _detect_plugin(path, available_plugins)


def load_results(
    path: Path,
    format_name: str | None,
    options: ParseOptions,
    plugins: Sequence[ResultParserPlugin] | None = None,
) -> ContextAwareRobotResults:
    """Loads parsed results with the selected parser plugin."""
    plugin = get_plugin(format_name, path, plugins)
    validate_options(plugin, options)
    return plugin.parse(path, options)


def validate_options(plugin: ResultParserPlugin, options: ParseOptions) -> None:
    """Validates parse options against plugin capabilities."""
    uses_tags = bool(options.include_tags or options.exclude_tags)
    if uses_tags and TAG_FILTERS not in plugin.capabilities:
        raise ValueError(
            f"Format '{plugin.name}' does not support --include/--exclude tag filters."
        )


def _get_plugin_by_name(
    format_name: str,
    plugins: Sequence[ResultParserPlugin],
) -> ResultParserPlugin:
    requested = format_name.lower()
    for plugin in plugins:
        if plugin.name == requested:
            return plugin
    raise ValueError(
        f"Unknown result format: {format_name}. "
        f"Available formats: {_format_plugin_names(plugins)}"
    )


def _detect_plugin(
    path: Path,
    plugins: Sequence[ResultParserPlugin],
) -> ResultParserPlugin:
    matches = [plugin for plugin in plugins if plugin.can_parse(path)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"No result parser plugin can parse: {path}. "
            f"Available formats: {_format_plugin_names(plugins)}"
        )
    raise ValueError(
        f"Multiple result parser plugins can parse: {path}. "
        f"Use --format with one of: {_format_plugin_names(matches)}"
    )


def _format_plugin_names(plugins: Sequence[ResultParserPlugin]) -> str:
    return ", ".join(plugin.name for plugin in plugins)
