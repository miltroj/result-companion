from pathlib import Path
from typing import Dict, Optional

from robot.api import ExecutionResult
from robot.reporting.resultwriter import ResultWriter

from result_companion.core.html.llm_injector import LLMDataInjector
from result_companion.core.results.visitors import UniqueNameResultVisitor


def create_llm_html_log(
    input_result_path: Path | str,
    llm_output_path: Path | str,
    llm_results: Dict[str, str],
    model_info: Dict[str, str] = None,
    overall_summary: Optional[str] = None,
) -> None:
    """Create HTML log with LLM data embedded in JS model.

    Args:
        input_result_path: Path to Robot Framework output.xml.
        llm_output_path: Path for generated HTML report.
        llm_results: Mapping of test names to LLM analysis.
        model_info: Optional model information.
        overall_summary: Optional synthesized summary of all failures.
    """
    results = ExecutionResult(str(input_result_path))

    results.visit(UniqueNameResultVisitor())
    results.visit(LLMDataInjector(llm_results, model_info, overall_summary))

    writer = ResultWriter(results)
    writer.write_results(report=None, log=str(llm_output_path))

    _inject_llm_ui(Path(llm_output_path))


def _inject_llm_ui(html_path: Path) -> None:
    """Add JavaScript to display LLM results per test."""
    script = """
<style>
.llm-section { margin: 12px 0; border: 1px solid var(--secondary-color); border-radius: 6px; overflow: hidden; }
.llm-header { padding: 10px 16px; background: var(--primary-color); color: var(--text-color); cursor: pointer; display: flex; justify-content: space-between; align-items: center; user-select: none; }
.llm-header:hover { background: var(--secondary-color); }
.llm-chevron { transition: transform 0.2s; font-size: 12px; }
.llm-chevron.collapsed { transform: rotate(-90deg); }
.llm-content { padding: 16px; background: var(--background-color); border-top: 1px solid var(--secondary-color); max-height: 500px; overflow-y: auto; display: none; position: relative; }
.llm-content h2 { color: var(--link-color); font-size: 14px; margin-top: 12px; margin-bottom: 8px; }
.llm-model { font-size: 11px; opacity: 0.7; margin-left: 8px; }
.llm-copy { position: absolute; top: 8px; right: 8px; background: var(--secondary-color); color: var(--text-color); border: 1px solid var(--primary-color); padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 12px; }
.llm-copy:hover { background: var(--primary-color); }
.llm-copy.copied { background: var(--pass-color); color: white; }
.test.fail .llm-header { border-left: 4px solid var(--fail-color); }
.test.pass .llm-header { border-left: 4px solid var(--pass-color); }
</style>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script>
$(function() {
    var llmData = null;
    var modelInfo = null;
    var processed = new Set();

    // Hide metadata row immediately
    $('th').filter(function() { return $(this).text().indexOf('__llm_results') !== -1; }).parent().hide();

    // Get LLM data
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

                if (data.overall_summary) {
                    var badge = modelInfo ? '<span class="llm-model">' + modelInfo.model + '</span>' : '';
                    var sumHtml = '<div class="llm-section">' +
                        '<div class="llm-header" style="border-left:4px solid var(--fail-color)">' +
                            '<div>Overall Summary ' + badge + '</div>' +
                            '<span class="llm-chevron collapsed">▼</span>' +
                        '</div>' +
                        '<div class="llm-content">' +
                            '<button class="llm-copy" data-raw="__overall_summary__">Copy</button>' +
                            marked.parse(data.overall_summary) +
                        '</div></div>';
                    var $sum = $(sumHtml);
                    $sum.find('.llm-header').click(function() {
                        var c = $(this).next(); var ch = $(this).find('.llm-chevron');
                        c.slideToggle(200); ch.toggleClass('collapsed');
                    });
                    $sum.find('.llm-copy').click(function(e) {
                        e.stopPropagation(); var b = $(this);
                        navigator.clipboard.writeText(data.overall_summary);
                        b.text('✓ Copied').addClass('copied');
                        setTimeout(function() { b.text('Copy').removeClass('copied'); }, 2000);
                    });
                    $('.suite').first().find('.children').first().prepend($sum);
                }
                break;
            }
        }
    } catch(e) {
        console.error('Failed to load LLM data:', e);
    }

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
            test.find('.children').first().prepend(html);

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
        $('.test, .suite').each(function() { process($(this)); });
    }, 1000);
});
</script>
"""
    html = html_path.read_text()
    html_path.write_text(html.replace("</body>", f"{script}\n</body>"))
