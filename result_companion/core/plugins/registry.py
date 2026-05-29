from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

from result_companion.core.plugins.base import (
    TAG_FILTERS,
    AnalysisResults,
    ParseOptions,
    ResultParserPlugin,
)
from result_companion.core.plugins.robot import RobotPlugin
from result_companion.core.utils.logging_config import logger

PLUGIN_ENTRY_POINT_GROUP = "result_companion.plugins"


def get_builtin_plugins() -> tuple[ResultParserPlugin, ...]:
    """Returns built-in parser plugins."""
    return (RobotPlugin(),)


def get_available_plugins() -> tuple[ResultParserPlugin, ...]:
    """Returns built-in and installed parser plugins."""
    return _deduplicate_plugins((*get_builtin_plugins(), *_load_installed_plugins()))


def get_plugin(
    format_name: str | None,
    path: Path,
    plugins: Sequence[ResultParserPlugin] | None = None,
) -> ResultParserPlugin:
    """Resolves a parser plugin by explicit format or auto-detection."""
    available_plugins = (
        tuple(plugins) if plugins is not None else get_available_plugins()
    )
    logger.debug(
        "Available parser plugins: "
        f"{', '.join(plugin.name for plugin in available_plugins) or '<none>'}"
    )
    if format_name:
        logger.debug(f"Resolving parser plugin by format: {format_name}")
        return _get_plugin_by_name(format_name, available_plugins)
    logger.debug(f"Auto-detecting parser plugin for: {path}")
    return _detect_plugin(path, available_plugins)


def load_results(
    path: Path,
    format_name: str | None,
    options: ParseOptions,
    plugins: Sequence[ResultParserPlugin] | None = None,
) -> AnalysisResults:
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
            logger.debug(f"Selected parser plugin: {plugin.name}")
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
    logger.debug(
        "Auto-detected parser plugin matches: "
        f"{', '.join(plugin.name for plugin in matches) or '<none>'}"
    )
    if len(matches) == 1:
        logger.debug(f"Selected parser plugin: {matches[0].name}")
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


def _load_installed_plugins() -> tuple[ResultParserPlugin, ...]:
    logger.debug(f"Scanning parser plugin entry points: {PLUGIN_ENTRY_POINT_GROUP}")
    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        selected = entry_points.select(group=PLUGIN_ENTRY_POINT_GROUP)
    else:
        selected = entry_points.get(PLUGIN_ENTRY_POINT_GROUP, ())

    plugins = tuple(_load_entry_point_plugin(entry_point) for entry_point in selected)
    logger.debug(
        "Loaded installed parser plugins: "
        f"{', '.join(plugin.name for plugin in plugins) or '<none>'}"
    )
    return plugins


def _load_entry_point_plugin(entry_point: metadata.EntryPoint) -> ResultParserPlugin:
    try:
        plugin = entry_point.load()
    except Exception as exc:
        logger.debug(
            f"Failed to load parser plugin entry point '{entry_point.name}'",
            exc_info=True,
        )
        raise ValueError(
            f"Failed to load parser plugin entry point '{entry_point.name}'."
        ) from exc

    if isinstance(plugin, type):
        plugin = plugin()
    elif not _is_parser_plugin(plugin) and callable(plugin):
        plugin = plugin()

    if _is_parser_plugin(plugin):
        return plugin

    raise ValueError(
        f"Parser plugin entry point '{entry_point.name}' must expose a ResultParserPlugin."
    )


def _is_parser_plugin(plugin: Any) -> bool:
    return all(
        hasattr(plugin, attribute)
        for attribute in ("name", "capabilities", "can_parse", "parse")
    )


def _deduplicate_plugins(
    plugins: Sequence[ResultParserPlugin],
) -> tuple[ResultParserPlugin, ...]:
    unique: dict[str, ResultParserPlugin] = {}
    for plugin in plugins:
        if plugin.name in unique:
            logger.debug(f"Ignoring duplicate parser plugin: {plugin.name}")
            continue
        unique[plugin.name] = plugin
    return tuple(unique.values())
