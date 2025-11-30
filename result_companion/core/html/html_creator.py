"""HTML report generation with embedded LLM results."""

from pathlib import Path
from typing import Dict

from robot.api import ExecutionResult
from robot.reporting.resultwriter import ResultWriter

from result_companion.core.html.llm_injector import LLMDataInjector
from result_companion.core.results.visitors import UniqueNameResultVisitor


def create_llm_html_log(
    input_result_path: Path | str,
    llm_output_path: Path | str,
    llm_results: Dict[str, str],
) -> None:
    """Create HTML log with LLM data embedded in JS model.

    Args:
        input_result_path: Path to Robot Framework output.xml.
        llm_output_path: Path for generated HTML report.
        llm_results: Mapping of test names to LLM analysis.
    """
    # Load results
    results = ExecutionResult(str(input_result_path))

    # Apply visitors
    results.visit(UniqueNameResultVisitor())
    results.visit(LLMDataInjector(llm_results))

    # Generate HTML with standard writer
    writer = ResultWriter(results)
    writer.write_results(report=None, log=str(llm_output_path))


if __name__ == "__main__":
    # TODO: remove this test code
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    input_result_path = REPO_ROOT / ".." / "examples" / "run_test" / "output.xml"
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

```python
import os
os.environ["DB_CONNECTION_STRING"] = "valid_connection_string"
```
"""

    multiline_html_response = """<div> something here </div>
                                    <div> deeper something here </div>"""  # .replace("\n", " \\ \n")
    create_llm_html_log(
        input_result_path=input_result_path,
        llm_results={
            "Test Neasted Test Case": multiline_html_response,
            "Ollama Local Model Run Should Succede": multiline_another,
        },
        llm_output_path="test_llm_full_log.html",
    )
