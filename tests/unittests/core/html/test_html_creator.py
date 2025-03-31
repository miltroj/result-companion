from unittest.mock import mock_open, patch

from result_companion.core.html.html_creator import create_llm_html_log


@patch(
    "result_companion.core.html.html_creator.open",
    new_callable=mock_open,
    read_data="data",
)
@patch(
    "result_companion.core.html.html_creator.create_base_llm_result_log", autospec=True
)
def test_create_llm_html_log(mock_create_base_llm_log, mock_open) -> None:
    """
    Test the creation of an HTML log for LLM results.
    """
    mock_create_base_llm_log.return_value = None
    # Create a sample LLM result
    llm_results = {
        "test_case_1": {
            "input": "SELECT * FROM users WHERE age > 30;",
            "output": "<div>something here</div>",
            "error": None,
        },
        "test_case_2": {
            "input": "SELECT * FROM orders WHERE amount > 1000;",
            "output": "<div>another output</div>",
            "error": None,
        },
    }

    # Call the function to create the HTML log
    create_llm_html_log("input_result_path", "llm_output_path", llm_results)

    mock_open.assert_called_once_with("llm_output_path", "a+", encoding="utf-8")
    mock_open().write.assert_called_once_with(
        '\n<script type="text/javascript">\nwindow.output = window.output || {};\nwindow.output.llm_msgs = {"test_case_1": {"input": "SELECT * FROM users WHERE age > 30;", "output": "<div>something here</div>", "error": null}, "test_case_2": {"input": "SELECT * FROM orders WHERE amount > 1000;", "output": "<div>another output</div>", "error": null}};\n</script>\n'
    )
