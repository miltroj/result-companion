from pathlib import Path
from unittest.mock import MagicMock, patch

from result_companion.core.html.html_creator import _inject_llm_ui, create_llm_html_log


@patch(
    "result_companion.core.html.html_creator.Path.read_text",
    return_value="<html></body></html>",
)
@patch("result_companion.core.html.html_creator.Path.write_text")
@patch("result_companion.core.html.html_creator.ResultWriter")
@patch("result_companion.core.html.html_creator.ExecutionResult")
def test_create_llm_html_log(
    mock_execution_result, mock_result_writer, mock_write, mock_read
) -> None:
    """Test HTML log creation with LLM data injection."""
    input_path = "output.xml"
    output_path = "log.html"
    llm_results = {"Test 1": "Analysis 1", "Test 2": "Analysis 2"}

    # Setup mocks
    mock_results = MagicMock()
    mock_execution_result.return_value = mock_results
    mock_writer = MagicMock()
    mock_result_writer.return_value = mock_writer

    # Execute
    create_llm_html_log(input_path, output_path, llm_results)

    # Verify
    mock_execution_result.assert_called_once_with(input_path)
    assert mock_results.visit.call_count == 2  # UniqueNameVisitor and LLMDataInjector
    mock_writer.write_results.assert_called_once_with(report=None, log=output_path)

    # Verify UI injection
    mock_read.assert_called_once()
    mock_write.assert_called_once()
    injected_html = mock_write.call_args[0][0]
    assert "AI Analysis" in injected_html
    assert "</body>" in injected_html


def test_inject_llm_ui_uses_utf8_encoding_and_injects_script_before_body(
    tmp_path: Path,
) -> None:
    """Script is injected before </body> and file round-trips correctly under UTF-8."""
    html_file = tmp_path / "log.html"
    html_file.write_text("<html><body>résumé</body></html>", encoding="utf-8")

    _inject_llm_ui(html_file)

    result = html_file.read_text(encoding="utf-8")
    assert "AI Analysis" in result
    assert result.index("AI Analysis") < result.index("</body>")
    assert "résumé" in result
