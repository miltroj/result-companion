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
.llm-section { margin: 12px 0; border: 1px solid #d0d7de; border-radius: 6px; overflow: hidden; transition: all 0.2s; }
.llm-section:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.llm-header { padding: 10px 16px; background: linear-gradient(90deg, #0969da, #1f6feb); color: white; cursor: pointer; display: flex; justify-content: space-between; align-items: center; user-select: none; }
.llm-header:hover { background: linear-gradient(90deg, #1f6feb, #0969da); }
.llm-chevron { transition: transform 0.2s; font-size: 12px; }
.llm-chevron.collapsed { transform: rotate(-90deg); }
.llm-content { padding: 16px; background: #f6f8fa; max-height: 500px; overflow-y: auto; display: none; position: relative; }
.llm-content h2 { color: #0969da; font-size: 14px; margin-top: 12px; margin-bottom: 8px; }
.llm-model { font-size: 11px; opacity: 0.9; background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 10px; }
.llm-copy { position: absolute; top: 8px; right: 8px; background: #0969da; color: white; border: none; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 12px; }
.llm-copy:hover { background: #1f6feb; }
.llm-copy.copied { background: #28a745; }
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
                    var div = document.createElement('div');
                    div.innerHTML = meta[i][1];
                    var decoded = div.textContent || div.innerText || '';
                    var data = JSON.parse(decoded);

                    if (data.results) {
                        llmData = data.results;
                        modelInfo = data.model;
                    } else {
                        llmData = data;
                    }

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

        if (llmData[name]) {
            processed.add(id);
            var modelBadge = modelInfo ? '<span class="llm-model">' + modelInfo.model + '</span>' : '';
            var html = '<div class="llm-section">' +
                '<div class="llm-header">' +
                    '<div>🤖 AI Analysis ' + modelBadge + '</div>' +
                    '<span class="llm-chevron collapsed">▼</span>' +
                '</div>' +
                '<div class="llm-content">' +
                    '<button class="llm-copy">Copy</button>' +
                    marked.parse(llmData[name]) +
                '</div></div>';
            test.find('.children').first().append(html);

            // Toggle with animation
            test.find('.llm-header').click(function() {
                var content = $(this).next();
                var chevron = $(this).find('.llm-chevron');
                content.slideToggle(200);
                chevron.toggleClass('collapsed');
            });

            // Copy button
            test.find('.llm-copy').click(function(e) {
                e.stopPropagation();
                var btn = $(this);
                navigator.clipboard.writeText(llmData[name]);
                btn.text('✓ Copied').addClass('copied');
                setTimeout(function() {
                    btn.text('Copy').removeClass('copied');
                }, 2000);
            });
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
