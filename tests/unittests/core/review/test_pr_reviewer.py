"""Tests for PR review orchestration."""

from types import SimpleNamespace

import pytest

from result_companion.core.results.text_report import AnalyzeReport
from result_companion.core.review.pr_reviewer import (
    build_review_prompt,
    ensure_gh_auth,
    run_review,
    save_review,
)


def make_review_config() -> SimpleNamespace:
    """Creates minimal review configuration for tests."""
    return SimpleNamespace(
        version=1.0,
        review=SimpleNamespace(
            review_prompt="Repo {repo_name} PR {pr_number}\n{failure_summary}",
            model="gpt-5-mini",
            timeout=300,
            startup_timeout=30,
            mcp_server_url="https://example.com/mcp",
        ),
    )


class FakeGhRunner:
    """Simple subprocess-like runner for gh checks."""

    def __init__(self, returncode: int = 0, missing: bool = False):
        self.returncode = returncode
        self.missing = missing
        self.calls: list[tuple[list[str], dict]] = []

    def run(self, command: list[str], **kwargs):
        """Records the command and returns a fake result."""
        self.calls.append((command, kwargs))
        if self.missing:
            raise FileNotFoundError("gh")
        return SimpleNamespace(returncode=self.returncode, stdout="", stderr="")


class TestBuildReviewPrompt:
    """Tests for prompt formatting."""

    def test_includes_runtime_values(self):
        prompt = build_review_prompt(
            repo_name="owner/repo",
            pr_number=42,
            failure_summary="summary text",
            prompts=make_review_config().review,
        )

        assert "owner/repo" in prompt
        assert "42" in prompt
        assert "summary text" in prompt


class TestEnsureGhAuth:
    """Tests for gh CLI preflight checks."""

    def test_raises_when_gh_is_missing(self):
        with pytest.raises(RuntimeError, match="not installed"):
            ensure_gh_auth(FakeGhRunner(missing=True))

    def test_raises_when_gh_is_not_authenticated(self):
        with pytest.raises(RuntimeError, match="gh auth login"):
            ensure_gh_auth(FakeGhRunner(returncode=1))

    def test_accepts_authenticated_gh(self):
        runner = FakeGhRunner(returncode=0)

        ensure_gh_auth(runner)

        assert runner.calls[0][0] == ["gh", "auth", "status"]


def make_empty_summary() -> str:
    """Creates JSON summary with no test failures."""
    return AnalyzeReport(
        failed_test_count=0,
        analyzed_tests=[],
        total_test_count=5,
        source_file="output.xml",
        source_hash="abc123",
    ).to_json()


def make_failure_summary() -> str:
    """Creates JSON summary with one test failure."""
    return AnalyzeReport(
        failed_test_count=1,
        analyzed_tests=["test_fail"],
        per_test_results={"test_fail": "real failure"},
    ).to_json()


class TestRunReview:
    """Tests for run_review orchestration."""

    def test_skips_when_no_failures(self, monkeypatch):
        monkeypatch.setattr(
            "result_companion.core.review.pr_reviewer.load_review_config",
            lambda *_: pytest.fail("config should not be loaded"),
        )

        result = run_review(
            repo_name="owner/repo",
            pr_number=5,
            summary=make_empty_summary(),
        )

        assert result == ""

    def test_raises_on_invalid_json(self):
        with pytest.raises(RuntimeError, match="Invalid summary"):
            run_review(
                repo_name="owner/repo",
                pr_number=5,
                summary="not json",
            )

    def test_checks_gh_before_generating_comment(self, monkeypatch):
        called = {"generated": False}

        async def fake_comment_runner(**kwargs) -> str:
            called["generated"] = True
            return "review body"

        monkeypatch.setattr(
            "result_companion.core.review.pr_reviewer.load_review_config",
            lambda *_: make_review_config(),
        )

        with pytest.raises(RuntimeError, match="gh auth login"):
            run_review(
                repo_name="owner/repo",
                pr_number=5,
                summary=make_failure_summary(),
                preview=False,
                comment_runner=fake_comment_runner,
                gh_runner=FakeGhRunner(returncode=1),
            )

        assert called["generated"] is False

    def test_skips_gh_check_during_preview(self, monkeypatch):
        async def fake_comment_runner(**kwargs) -> str:
            return "review body"

        monkeypatch.setattr(
            "result_companion.core.review.pr_reviewer.load_review_config",
            lambda *_: make_review_config(),
        )

        result = run_review(
            repo_name="owner/repo",
            pr_number=5,
            summary=make_failure_summary(),
            preview=True,
            comment_runner=fake_comment_runner,
            gh_runner=FakeGhRunner(returncode=1),
        )

        assert result == "review body"

    def test_posts_comment_when_not_preview(self, monkeypatch):
        posted = {}

        async def fake_comment_runner(**kwargs) -> str:
            return "review body"

        def fake_comment_poster(
            repo_name: str,
            pr_number: int,
            comment_body: str,
            runner,
        ) -> None:
            posted["repo_name"] = repo_name
            posted["pr_number"] = pr_number
            posted["comment_body"] = comment_body
            posted["runner"] = runner

        gh_runner = FakeGhRunner(returncode=0)
        monkeypatch.setattr(
            "result_companion.core.review.pr_reviewer.load_review_config",
            lambda *_: make_review_config(),
        )

        result = run_review(
            repo_name="owner/repo",
            pr_number=5,
            summary=make_failure_summary(),
            preview=False,
            comment_runner=fake_comment_runner,
            gh_runner=gh_runner,
            comment_poster=fake_comment_poster,
        )

        assert result == "review body"
        assert posted["repo_name"] == "owner/repo"
        assert posted["pr_number"] == 5
        assert posted["comment_body"] == "review body"
        assert posted["runner"] is gh_runner

    def test_raises_on_empty_comment(self, monkeypatch):
        async def fake_comment_runner(**kwargs) -> str:
            return "   "

        monkeypatch.setattr(
            "result_companion.core.review.pr_reviewer.load_review_config",
            lambda *_: make_review_config(),
        )

        with pytest.raises(RuntimeError, match="empty comment"):
            run_review(
                repo_name="owner/repo",
                pr_number=5,
                summary=make_failure_summary(),
                preview=True,
                comment_runner=fake_comment_runner,
            )

    def test_notify_on_pass_returns_all_passed_comment_in_preview(self):
        result = run_review(
            repo_name="owner/repo",
            pr_number=5,
            summary=make_empty_summary(),
            notify_on_pass=True,
            preview=True,
        )

        assert "All Robot Framework tests passed" in result
        assert "Total tests: 5" in result
        assert "`output.xml`" in result
        assert "`abc123`" in result

    def test_notify_on_pass_posts_comment_when_not_preview(self):
        posted = {}

        def fake_poster(repo_name, pr_number, comment_body, runner):
            posted["comment_body"] = comment_body

        result = run_review(
            repo_name="owner/repo",
            pr_number=5,
            summary=make_empty_summary(),
            notify_on_pass=True,
            preview=False,
            gh_runner=FakeGhRunner(returncode=0),
            comment_poster=fake_poster,
        )

        assert "All Robot Framework tests passed" in result
        assert posted["comment_body"] == result

    def test_notify_on_pass_saves_to_output_file(self, tmp_path):
        out = tmp_path / "pass.md"
        result = run_review(
            repo_name="owner/repo",
            pr_number=5,
            summary=make_empty_summary(),
            notify_on_pass=True,
            preview=True,
            output_path=str(out),
        )

        assert out.read_text() == result

    def test_notify_on_pass_false_skips_on_no_failures(self):
        result = run_review(
            repo_name="owner/repo",
            pr_number=5,
            summary=make_empty_summary(),
            notify_on_pass=False,
        )

        assert result == ""


class TestSaveReview:
    """Tests for save_review function."""

    def test_writes_content_to_file(self, tmp_path):
        out = tmp_path / "review.md"

        save_review(str(out), "review body")

        assert out.read_text() == "review body"

    def test_warns_on_non_markdown_extension(self, tmp_path, caplog):
        import logging

        out = tmp_path / "review.txt"

        with caplog.at_level(logging.WARNING):
            save_review(str(out), "review body")

        assert out.read_text() == "review body"
        assert "not a Markdown file" in caplog.text

    def test_no_warning_on_md_extension(self, tmp_path, caplog):
        import logging

        out = tmp_path / "review.md"

        with caplog.at_level(logging.WARNING):
            save_review(str(out), "content")

        assert "not a Markdown file" not in caplog.text
