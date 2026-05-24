from pathlib import Path
from unittest.mock import MagicMock, patch

from result_companion.core.plugins.base import ParseOptions
from result_companion.core.plugins.robot import RobotPlugin


def test_robot_plugin_can_parse_robot_xml(tmp_path):
    output = tmp_path / "output.xml"
    output.write_text("<robot><suite /></robot>")

    assert RobotPlugin().can_parse(output)


def test_robot_plugin_can_parse_rejects_non_robot_xml(tmp_path):
    output = tmp_path / "junit.xml"
    output.write_text("<testsuite />")

    assert not RobotPlugin().can_parse(output)


def test_robot_plugin_parse_forwards_options():
    plugin = RobotPlugin()
    options = ParseOptions(
        include_tags=["smoke"],
        exclude_tags=["wip"],
        exclude_fields=["timestamp"],
        exclude_passing=False,
    )
    fake_results = MagicMock()

    with patch(
        "result_companion.core.plugins.robot.get_rc_robot_results",
        return_value=fake_results,
    ) as mocked_get_results:
        result = plugin.parse(Path("output.xml"), options)

    assert result is fake_results
    mocked_get_results.assert_called_once_with(
        file_path=Path("output.xml"),
        include_tags=["smoke"],
        exclude_tags=["wip"],
        exclude_fields=["timestamp"],
        exclude_passing=False,
    )
