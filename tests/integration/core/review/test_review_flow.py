"""Integration tests for the review flow — real run_review with fake Copilot."""

from types import SimpleNamespace

import pytest

from result_companion.core.results.text_report import AnalyzeReport
from result_companion.core.review.pr_reviewer import run_review


def _make_summary(failed: int = 1) -> str:
    return AnalyzeReport(
        failed_test_count=failed,
        analyzed_tests=["test_login"] if failed else [],
        per_test_results={"test_login": "503 from backend"} if failed else {},
        overall_summary="Backend down" if failed else None,
        total_test_count=10,
        source_file="output.xml",
        source_hash="abc123def456",
        timestamp="2026-03-27T12:00:00+00:00",
    ).to_json()


class FakeGhRunner:
    """Records subprocess calls without executing anything."""

    def __init__(self, returncode: int = 0):
        self.calls: list[list[str]] = []
        self._rc = returncode

    def run(self, command, **kwargs):
        self.calls.append(command)
        return SimpleNamespace(returncode=self._rc, stdout="", stderr="")


class TestReviewFlowWithFailures:
    """End-to-end: real run_review → real config → fake Copilot → comment."""

    @staticmethod
    async def _fake_generator(**kwargs) -> str:
        return f"Review for {kwargs['repo_name']}#{kwargs['pr_number']}"

    def test_preview_returns_comment_without_posting(self):
        result = run_review(
            repo_name="owner/repo",
            pr_number=42,
            summary=_make_summary(failed=1),
            preview=True,
            comment_runner=self._fake_generator,
        )

        assert "owner/repo#42" in result

    def test_post_mode_calls_gh_and_returns_comment(self):
        gh = FakeGhRunner(returncode=0)
        posted = {}

        def capture_poster(repo_name, pr_number, comment_body, runner):
            posted["body"] = comment_body

        result = run_review(
            repo_name="owner/repo",
            pr_number=42,
            summary=_make_summary(failed=1),
            preview=False,
            comment_runner=self._fake_generator,
            gh_runner=gh,
            comment_poster=capture_poster,
        )

        assert result == posted["body"]

    def test_output_file_written(self, tmp_path):
        out = tmp_path / "review.md"
        run_review(
            repo_name="owner/repo",
            pr_number=1,
            summary=_make_summary(failed=1),
            preview=True,
            output_path=str(out),
            comment_runner=self._fake_generator,
        )

        assert out.exists()
        assert "owner/repo#1" in out.read_text()


class TestReviewFlowNoFailures:
    """End-to-end: no failures path — no Copilot call needed."""

    def test_skips_silently_without_notify_on_pass(self):
        result = run_review(
            repo_name="owner/repo",
            pr_number=1,
            summary=_make_summary(failed=0),
        )

        assert result == ""

    def test_notify_on_pass_preview_returns_all_clear(self):
        result = run_review(
            repo_name="owner/repo",
            pr_number=1,
            summary=_make_summary(failed=0),
            notify_on_pass=True,
            preview=True,
        )

        assert "All Robot Framework tests passed" in result
        assert "Total tests: 10" in result
        assert "`output.xml`" in result

    def test_notify_on_pass_posts_and_saves(self, tmp_path):
        gh = FakeGhRunner(returncode=0)
        posted = {}
        out = tmp_path / "pass.md"

        def capture_poster(repo_name, pr_number, comment_body, runner):
            posted["body"] = comment_body

        result = run_review(
            repo_name="owner/repo",
            pr_number=1,
            summary=_make_summary(failed=0),
            notify_on_pass=True,
            preview=False,
            gh_runner=gh,
            comment_poster=capture_poster,
            output_path=str(out),
        )

        assert result == posted["body"]
        assert out.read_text() == result


class TestReviewFlowErrorPaths:
    """End-to-end: error handling without mocking internals."""

    def test_invalid_json_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="Invalid summary"):
            run_review(
                repo_name="owner/repo",
                pr_number=1,
                summary="not json at all",
            )

    def test_empty_copilot_response_raises(self):
        async def empty_generator(**kwargs) -> str:
            return "   "

        with pytest.raises(RuntimeError, match="empty comment"):
            run_review(
                repo_name="owner/repo",
                pr_number=1,
                summary=_make_summary(failed=1),
                preview=True,
                comment_runner=empty_generator,
            )

    def test_unauthenticated_gh_blocks_post(self):
        async def fake_gen(**kwargs) -> str:
            return "review"

        with pytest.raises(RuntimeError, match="not authenticated"):
            run_review(
                repo_name="owner/repo",
                pr_number=1,
                summary=_make_summary(failed=1),
                preview=False,
                comment_runner=fake_gen,
                gh_runner=FakeGhRunner(returncode=1),
            )
