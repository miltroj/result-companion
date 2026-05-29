import logging
from pathlib import Path

import pytest

from result_companion.core.plugins import registry
from result_companion.core.plugins.base import ParseOptions
from result_companion.core.plugins.registry import (
    PLUGIN_ENTRY_POINT_GROUP,
    get_available_plugins,
    get_plugin,
    load_results,
    validate_options,
)


class FakePlugin:
    """Simple parser plugin fake for registry tests."""

    def __init__(
        self,
        name: str = "fake",
        can_parse_result: bool = False,
        capabilities: frozenset[str] = frozenset(),
    ) -> None:
        self.name = name
        self.capabilities = capabilities
        self.can_parse_result = can_parse_result
        self.parse_calls: list[tuple[Path, ParseOptions]] = []

    def can_parse(self, path: Path) -> bool:
        """Returns configured parser match result."""
        return self.can_parse_result

    def parse(self, path: Path, options: ParseOptions):
        """Records parse call and returns a fake result object."""
        self.parse_calls.append((path, options))
        return "RESULT"


class FakeEntryPoint:
    """Simple entry point fake for discovery tests."""

    def __init__(self, name: str, plugin, error: Exception | None = None) -> None:
        self.name = name
        self.plugin = plugin
        self.error = error

    def load(self):
        """Returns configured plugin or raises configured error."""
        if self.error:
            raise self.error
        return self.plugin


class FakeEntryPoints(list):
    """Entry point collection fake with importlib.metadata-compatible select."""

    def select(self, group: str):
        """Returns entry points for the requested group."""
        if group == PLUGIN_ENTRY_POINT_GROUP:
            return self
        return []


def test_get_plugin_resolves_explicit_robot():
    plugin = get_plugin("robot", Path("output.xml"))

    assert plugin.name == "robot"


def test_get_plugin_resolves_explicit_builtin_without_discovery(monkeypatch):
    monkeypatch.setattr(
        registry.metadata,
        "entry_points",
        lambda: pytest.fail("discovery should not run for built-in formats"),
    )

    plugin = get_plugin("robot", Path("output.xml"))

    assert plugin.name == "robot"


def test_get_plugin_rejects_unknown_format():
    with pytest.raises(ValueError, match="Unknown result format: junit"):
        get_plugin("junit", Path("junit.xml"))


def test_validate_options_rejects_tag_filters_without_tag_capability():
    plugin = FakePlugin(name="junit")
    options = ParseOptions(include_tags=["smoke"])

    with pytest.raises(
        ValueError,
        match="Format 'junit' does not support --include/--exclude tag filters.",
    ):
        validate_options(plugin, options)


def test_load_results_uses_detected_plugin():
    plugin = FakePlugin(can_parse_result=True)
    options = ParseOptions()

    result = load_results(Path("results.xml"), None, options, plugins=[plugin])

    assert result == "RESULT"
    assert plugin.parse_calls == [(Path("results.xml"), options)]


def test_get_plugin_resolves_discovered_plugin(monkeypatch):
    plugin = FakePlugin(name="junit")
    entry_points = FakeEntryPoints([FakeEntryPoint("junit", plugin)])
    monkeypatch.setattr(registry.metadata, "entry_points", lambda: entry_points)

    result = get_plugin("junit", Path("junit.xml"))

    assert result is plugin


def test_load_results_uses_discovered_plugin_for_auto_detect(monkeypatch, tmp_path):
    plugin = FakePlugin(name="junit", can_parse_result=True)
    entry_points = FakeEntryPoints([FakeEntryPoint("junit", plugin)])
    monkeypatch.setattr(registry.metadata, "entry_points", lambda: entry_points)
    options = ParseOptions()
    path = tmp_path / "junit.xml"
    path.write_text("not robot xml")

    result = load_results(path, None, options)

    assert result == "RESULT"
    assert plugin.parse_calls == [(path, options)]


def test_get_plugin_uses_explicit_plugins_over_discovery(monkeypatch):
    plugin = FakePlugin(name="local")
    monkeypatch.setattr(
        registry.metadata,
        "entry_points",
        lambda: pytest.fail("discovery should not run"),
    )

    result = get_plugin("local", Path("local.xml"), plugins=[plugin])

    assert result is plugin


def test_get_available_plugins_deduplicates_plugin_names(monkeypatch):
    duplicate = FakePlugin(name="robot")
    entry_points = FakeEntryPoints([FakeEntryPoint("robot", duplicate)])
    monkeypatch.setattr(registry.metadata, "entry_points", lambda: entry_points)

    plugins = get_available_plugins()

    assert [plugin.name for plugin in plugins] == ["robot"]


def test_get_plugin_skips_broken_entry_point(monkeypatch, caplog):
    entry_points = FakeEntryPoints(
        [FakeEntryPoint("broken", None, error=RuntimeError("boom"))]
    )
    monkeypatch.setattr(registry.metadata, "entry_points", lambda: entry_points)

    with caplog.at_level(logging.WARNING, logger="RC"):
        with pytest.raises(ValueError, match="Unknown result format: broken"):
            get_plugin("broken", Path("broken.xml"))

    assert "Skipping parser plugin entry point 'broken'" in caplog.text


def test_get_plugin_broken_entry_point_does_not_block_valid_plugin(monkeypatch, caplog):
    plugin = FakePlugin(name="junit")
    entry_points = FakeEntryPoints(
        [
            FakeEntryPoint("broken", None, error=RuntimeError("boom")),
            FakeEntryPoint("junit", plugin),
        ]
    )
    monkeypatch.setattr(registry.metadata, "entry_points", lambda: entry_points)

    with caplog.at_level(logging.WARNING, logger="RC"):
        result = get_plugin("junit", Path("junit.xml"))

    assert result is plugin
    assert "Skipping parser plugin entry point 'broken'" in caplog.text


def test_get_plugin_logs_discovery_and_selection(monkeypatch, caplog):
    plugin = FakePlugin(name="junit")
    entry_points = FakeEntryPoints([FakeEntryPoint("junit", plugin)])
    monkeypatch.setattr(registry.metadata, "entry_points", lambda: entry_points)

    with caplog.at_level(logging.DEBUG, logger="RC"):
        result = get_plugin("junit", Path("junit.xml"))

    assert result is plugin
    assert (
        "Scanning parser plugin entry points: result_companion.plugins" in caplog.text
    )
    assert "Loaded installed parser plugins: junit" in caplog.text
    assert "Selected parser plugin: junit" in caplog.text
