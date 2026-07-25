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


def test_robot_plugin_can_parse_returns_false_for_missing_file(tmp_path):
    assert not RobotPlugin().can_parse(tmp_path / "missing.xml")


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


def test_robot_plugin_render_html_report_forwards_args():
    plugin = RobotPlugin()
    llm_results = {"test_fail": "Root cause"}
    model_info = {"model": "openai/gpt-4"}

    with patch(
        "result_companion.core.plugins.robot.create_llm_html_log"
    ) as mocked_html:
        plugin.render_html_report(
            input_path=Path("output.xml"),
            output_path=Path("report.html"),
            llm_results=llm_results,
            model_info=model_info,
            overall_summary="Summary",
        )

    mocked_html.assert_called_once_with(
        input_result_path=Path("output.xml"),
        llm_output_path=Path("report.html"),
        llm_results=llm_results,
        model_info=model_info,
        overall_summary="Summary",
    )
