"""E2E test for review agentic flow — real Copilot, minimal prompt."""

import pytest

from result_companion.core.parsers.config import ReviewConfigModel, ReviewPromptModel
from result_companion.core.results.text_report import AnalyzeReport
from result_companion.core.review.pr_reviewer import _generate_review_comment

pytestmark = pytest.mark.e2e

FAST_PROMPT = (
    "Reply with exactly one sentence: 'No issues found for PR #{pr_number} "
    "in {repo_name}.'\nContext: {failure_summary}"
)


def _make_fast_config() -> ReviewConfigModel:
    return ReviewConfigModel(
        version=1.0,
        review=ReviewPromptModel(
            review_prompt=FAST_PROMPT,
            model="gpt-4.1",
            timeout=60,
            startup_timeout=30,
            mcp_server_url="",
        ),
    )


class TestReviewAgentE2E:
    """Smoke test: verifies the agentic review flow starts and returns a response."""

    @pytest.mark.asyncio
    async def test_generate_review_comment_returns_nonempty(self):
        summary = AnalyzeReport(
            failed_test_count=1,
            analyzed_tests=["test_login"],
            per_test_results={"test_login": "503 error"},
        ).to_text()

        result = await _generate_review_comment(
            repo_name="miltroj/result-companion",
            pr_number=1,
            failure_summary=summary,
            config=_make_fast_config(),
            quiet=True,
        )

        assert result
        assert len(result) > 5
