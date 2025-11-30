"""LLM data injection into Robot Framework results."""

import json
from typing import Dict

from robot.result.visitor import ResultVisitor


class LLMDataInjector(ResultVisitor):
    """Injects LLM results directly into test data."""

    def __init__(self, llm_results: Dict[str, str]):
        self.llm_results = llm_results

    def end_result(self, result):
        """Store LLM data as global metadata."""
        # Store all LLM results in suite metadata for JS access
        if result.suite and self.llm_results:
            # Store as JSON in a special metadata field
            result.suite.metadata["__llm_results"] = json.dumps(self.llm_results)
