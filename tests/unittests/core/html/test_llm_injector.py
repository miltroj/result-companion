"""Tests for LLM data injection."""

import json
from unittest.mock import MagicMock

from result_companion.core.html.llm_injector import LLMDataInjector


def test_llm_data_injector_stores_in_suite_metadata():
    """Test that LLM data is stored in suite metadata."""
    llm_results = {
        "Test 1": "Analysis for test 1",
        "Test 2": "Analysis for test 2",
    }

    injector = LLMDataInjector(llm_results)

    # Create mock result with suite
    result = MagicMock()
    result.suite.metadata = {}

    # Call end_result
    injector.end_result(result)

    # Verify data was stored as JSON in metadata
    assert "__llm_results" in result.suite.metadata
    stored_data = json.loads(result.suite.metadata["__llm_results"])
    assert stored_data == llm_results
