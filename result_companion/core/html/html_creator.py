from pathlib import Path

import markdown

from result_companion.core.html.result_writer import (
    create_base_llm_result_log,
)


def _escape_multilines_in_html(html: str) -> str:
    return html.replace("\n", "<br>").replace('"', '\\"')


def append_llm_js_mappings(test_llm_results: dict, path: str) -> None:
    final_response = "\n".join(
        [
            f'"{test_name}": "{markdown.markdown(_escape_multilines_in_html(response))}",'
            for test_name, response in test_llm_results.items()
        ]
    )

    template = (
        '\n<script type="text/javascript">'
        + 'window.output["llm_msgs"] = '
        + "{"
        + final_response
        + "}"
        + "</script>"
    )
    with open(path, "a+") as f:
        f.write(template)


def create_llm_html_log(
    input_result_path: "Path|str", llm_output_path: "Path|str", llm_results: dict
) -> None:
    create_base_llm_result_log(input_result_path, llm_output_path)
    append_llm_js_mappings(path=llm_output_path, test_llm_results=llm_results)


if __name__ == "__main__":
    # TODO: remove this test code

    multiline_another = """**General Idea Behind Test Case**
This test case is designed to execute a SQL query on a database and validate the results.

**Flow**

* The test connects to the database using a provided connection string.
* It logs a message indicating that the query is being executed.
* The test executes the SQL query.
* If the query fails, it logs an error message and raises an exception.
* Finally, the test checks if the result of the query matches an expected result.

**Failure Root Cause**
The root cause of the failure is that the database connection string is invalid, causing a "Connection Timeout" error. This prevents the test from successfully executing the SQL query and comparing its results to the expected result.

**Potential Fixes**

* Verify that the provided connection string is correct and properly formatted.
* Use a valid and existing database connection string for testing purposes.
* Consider using environment variables or configuration files to store sensitive information like database credentials, making it easier to manage and rotate them.
"""

    multiline_html_response = """<div> something here </div>
                                    <div> deeper something here </div>"""  # .replace("\n", " \\ \n")
    create_llm_html_log(
        input_result_path=r"D:\Repozytoria\result-companion\output.xml",
        llm_results={
            "Execute Another Sql Query And Validate Results": "response1",
            "Test Pydantic Validation": multiline_html_response,
            "Execute Sql Query And Validate Results": multiline_another,
        },
        llm_output_path="test_llm_full_log.html",
    )
