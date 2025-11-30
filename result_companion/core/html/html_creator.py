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
    model_info: Dict[str, str] = None,
) -> None:
    """Create HTML log with LLM data embedded in JS model.

    Args:
        input_result_path: Path to Robot Framework output.xml.
        llm_output_path: Path for generated HTML report.
        llm_results: Mapping of test names to LLM analysis.
        model_info: Optional model information.
    """
    # Load results
    results = ExecutionResult(str(input_result_path))

    # Apply visitors
    results.visit(UniqueNameResultVisitor())
    results.visit(LLMDataInjector(llm_results, model_info))

    # Generate HTML with standard writer
    writer = ResultWriter(results)
    writer.write_results(report=None, log=str(llm_output_path))

    # Inject UI enhancements
    _inject_llm_ui(Path(llm_output_path))


def _inject_llm_ui(html_path: Path) -> None:
    """Add JavaScript to display LLM results per test."""
    script = """
<style>
.llm-section { margin: 10px 0; border: 1px solid var(--secondary-color); border-radius: 4px; }
.llm-header { padding: 8px; background: var(--primary-color); cursor: pointer; }
.llm-content { padding: 10px; max-height: 300px; overflow-y: auto; }
</style>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script>
$(function() {
    var llmData = null;
    var modelInfo = null;
    var processed = new Set();

    // Get LLM data from metadata
    setTimeout(function() {
        try {
            var meta = window.testdata.suite().metadata;
            for (var i in meta) {
                if (meta[i][0] === '__llm_results') {
                    // Decode HTML entities first
                    var div = document.createElement('div');
                    div.innerHTML = meta[i][1];
                    var decoded = div.textContent || div.innerText || '';
                    var data = JSON.parse(decoded);

                    // Support both formats
                    if (data.results) {
                        llmData = data.results;
                        modelInfo = data.model;
                    } else {
                        llmData = data;  // Backwards compatibility
                    }

                    console.log('LLM Data loaded:', Object.keys(llmData));
                    if (modelInfo) console.log('Model:', modelInfo.model);
                    $('tr:has(th:contains("__llm_results"))').hide();
                    break;
                }
            }
        } catch(e) {
            console.error('Failed to load LLM data:', e);
        }
    }, 1000);

    // Process test element
    function process(test) {
        var id = test.attr('id');
        if (!id || processed.has(id) || !llmData) return;

        var name = test.find('.element-header .name').first().text().trim();
        console.log('Processing test:', name, 'Has LLM?', name in llmData);

        if (llmData[name]) {
            processed.add(id);
            console.log('Adding LLM section for:', name);
            var modelText = modelInfo ? ' (' + modelInfo.model + ')' : '';
            var html = '<div class="llm-section"><div class="llm-header">🤖 AI Analysis' + modelText + '</div>' +
                '<div class="llm-content">' + marked.parse(llmData[name]) + '</div></div>';
            test.find('.children').first().append(html);

            test.find('.llm-header').click(function() {
                $(this).next().toggle();
            });

            if (!test.find('.label.fail').length) {
                test.find('.llm-content').hide();
            }
        }
    }

    // Process all tests periodically
    setInterval(function() {
        $('.test').each(function() { process($(this)); });
    }, 1000);
});
</script>
"""
    html = html_path.read_text()
    html_path.write_text(html.replace("</body>", f"{script}\n</body>"))


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
