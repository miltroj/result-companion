from pathlib import Path

import pytest

from result_companion.core.plugins.base import ParseOptions
from result_companion.core.plugins.registry import get_plugin, load_results, validate_options


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


def test_get_plugin_resolves_explicit_robot():
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
