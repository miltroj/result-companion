import json
from typing import Dict, Optional

from robot.result.visitor import ResultVisitor


class LLMDataInjector(ResultVisitor):
    """Injects LLM results directly into test data."""

    def __init__(
        self,
        llm_results: Dict[str, str],
        model_info: Dict[str, str] = None,
        overall_summary: Optional[str] = None,
    ):
        self.llm_results = llm_results
        self.model_info = model_info
        self.overall_summary = overall_summary

    def end_result(self, result):
        """Store LLM data as global metadata."""
        if result.suite and self.llm_results:
            data = {"results": self.llm_results}
            if self.model_info:
                data["model"] = self.model_info
            if self.overall_summary:
                data["overall_summary"] = self.overall_summary
            result.suite.metadata["__llm_results"] = json.dumps(data)
