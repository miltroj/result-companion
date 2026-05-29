from __future__ import annotations

from pathlib import Path

from result_companion.core.plugins.base import ParseOptions
from result_companion.core.plugins.registry import load_results


def test_load_results_discovers_external_plugin_package(monkeypatch, tmp_path):
    """Installed package metadata exposes parser plugins to the registry."""
    _write_external_plugin_package(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    result_path = tmp_path / "results.extlog"
    result_path.write_text("external result")
    options = ParseOptions(exclude_passing=False)

    result = load_results(result_path, "external-log", options)

    assert result == "parsed:results.extlog:False"


def _write_external_plugin_package(root: Path) -> None:
    """Writes a minimal installed package shape with entry point metadata."""
    package = root / "rc_external_plugin"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "plugin.py").write_text(
        """\
from pathlib import Path

from result_companion.core.plugins.base import ParseOptions


class ExternalLogPlugin:
    name = "external-log"

    def can_parse(self, path: Path) -> bool:
        return path.suffix == ".extlog"

    def parse(self, path: Path, options: ParseOptions) -> str:
        return f"parsed:{path.name}:{options.exclude_passing}"
"""
    )

    dist_info = root / "result_companion_external_plugin-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        """\
Metadata-Version: 2.1
Name: result-companion-external-plugin
Version: 1.0
"""
    )
    (dist_info / "entry_points.txt").write_text(
        """\
[result_companion.plugins]
external-log = rc_external_plugin.plugin:ExternalLogPlugin
"""
    )
