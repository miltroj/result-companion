from unittest.mock import MagicMock, patch

from result_companion.core.html.html_creator import create_llm_html_log


@patch("result_companion.core.html.html_creator.ResultWriter")
@patch("result_companion.core.html.html_creator.ExecutionResult")
def test_create_llm_html_log(mock_execution_result, mock_result_writer) -> None:
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
