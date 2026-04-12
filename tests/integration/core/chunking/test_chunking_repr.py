from result_companion.core.chunking.chunking import (
    chunk_rf_test_lines,
    deduplicate_consecutive_lines,
    render_lines_to_text,
    render_rf_test_structure,
)

SAMPLE_TEST_CASE = {
    "name": "Processes Data And Writes Output",
    "status": "FAIL",
    "suite_context": [
        {"name": "My Application"},
        {"name": "Data Pipeline"},
    ],
    "body": [
        {
            "type": "KEYWORD",
            "name": "Connect To Database",
            "status": "PASS",
            "body": [
                {
                    "type": "MESSAGE",
                    "message": "DeprecationWarning: Call to deprecated create function.",
                },
                {
                    "type": "MESSAGE",
                    "message": "DeprecationWarning: Call to deprecated create function.",
                },
                {
                    "type": "MESSAGE",
                    "message": "DeprecationWarning: Call to deprecated create function.",
                },
            ],
        },
        {
            "type": "KEYWORD",
            "name": "Write Output",
            "status": "FAIL",
            "body": [
                {"type": "MESSAGE", "message": "Writing 1000 rows to output table."},
            ],
        },
    ],
}


def test_render_pipeline_produces_readable_indented_text():
    lines = render_rf_test_structure(SAMPLE_TEST_CASE)
    lines = deduplicate_consecutive_lines(lines)
    text = render_lines_to_text(lines)

    expected = (
        "Suite: My Application\n"
        "    Suite: Data Pipeline\n"
        "        Test: Processes Data And Writes Output - FAIL\n"
        "            Keyword: Connect To Database - PASS\n"
        "                DeprecationWarning: Call to deprecated create function. (repeats ×3)\n"
        "            Keyword: Write Output - FAIL\n"
        "                Writing 1000 rows to output table."
    )

    assert text == expected


def test_chunking_injects_ancestor_breadcrumbs_into_continuation_chunk():
    lines = render_rf_test_structure(SAMPLE_TEST_CASE)
    lines = deduplicate_consecutive_lines(lines)

    # chunk_size=220: keyword header fits in chunk 0, its log message spills to chunk 1,
    # second keyword spills to chunk 2 — each continuation repeats the ancestor chain.
    chunks = chunk_rf_test_lines(lines, chunk_size=220)

    assert len(chunks) == 3

    # chunk 0: keyword header only (log message overflows budget)
    assert "Connect To Database" in chunks[0]
    assert "DeprecationWarning" not in chunks[0]

    # chunk 1: log message continuation — ancestor chain re-injected for LLM context
    assert "Suite: My Application" in chunks[1]
    assert "Keyword: Connect To Database" in chunks[1]
    assert "{...}" in chunks[1]
    assert "DeprecationWarning" in chunks[1]

    # chunk 2: second keyword — suite/test ancestry repeated again
    assert "Suite: My Application" in chunks[2]
    assert "Test: Processes Data And Writes Output" in chunks[2]
    assert "{...}" in chunks[2]
    assert "Write Output" in chunks[2]
