from result_companion.core.results.analysis_result import AnalysisResult


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_defaults_are_empty(self):
        result = AnalysisResult()

        assert result.llm_results == {}
        assert result.test_names == []
        assert result.summary is None

    def test_text_report_includes_results_and_summary(self):
        result = AnalysisResult(
            llm_results={"Test A": "Timeout on login"},
            test_names=["Test A"],
            summary="Root cause: flaky network",
        )

        report = result.text_report

        assert "Test A" in report
        assert "Timeout on login" in report
        assert "Root cause: flaky network" in report

    def test_text_report_empty_results(self):
        result = AnalysisResult()

        report = result.text_report

        assert "Tests analyzed: 0" in report
