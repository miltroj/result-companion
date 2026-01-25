from result_companion.core.parsers.result_parser import (
    get_robot_results_from_file_as_dict,
)
from result_companion.core.utils.log_levels import LogLevels

TAGGED_TESTS_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<robot generator="Robot 7.1" rpa="false" schemaversion="5">
<suite id="s1" name="Tests">
  <test id="t1" name="Smoke Test"><tag>smoke</tag><status status="FAIL"/></test>
  <test id="t2" name="WIP Test"><tag>wip</tag><status status="FAIL"/></test>
  <test id="t3" name="Regression"><tag>regression</tag><status status="FAIL"/></test>
  <test id="t4" name="Smoke WIP"><tag>smoke</tag><tag>wip</tag><status status="FAIL"/></test>
  <status status="FAIL"/>
</suite>
</robot>
"""


def test_get_robot_results_filters_by_tags_using_rf_native(tmp_path):
    """Integration test: RF's native filtering via result.configure()."""
    xml_file = tmp_path / "output.xml"
    xml_file.write_text(TAGGED_TESTS_XML)

    result = get_robot_results_from_file_as_dict(
        file_path=xml_file,
        log_level=LogLevels.DEBUG,
        include_tags=["smoke"],
        exclude_tags=["wip"],
    )

    # Only "Smoke Test" matches: has smoke tag, no wip tag
    assert len(result) == 1
    assert result[0]["name"] == "Smoke Test"
