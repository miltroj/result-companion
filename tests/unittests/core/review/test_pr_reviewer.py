"""Tests for PR review orchestration."""

from types import SimpleNamespace

import pytest

from result_companion.core.review.pr_reviewer import (
    build_review_prompt,
    ensure_gh_auth,
    run_review,
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


class TestRunReview:
    """Tests for run_review orchestration."""

    def test_skips_when_summary_has_no_failures(self, monkeypatch):
        monkeypatch.setattr(
            "result_companion.core.review.pr_reviewer.load_review_config",
            lambda *_: pytest.fail("config should not be loaded"),
        )

        result = run_review(
            repo_name="owner/repo",
            pr_number=5,
            failure_summary="Tests analyzed: 0\nnothing to do\n",
        )

        assert result == ""

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
                failure_summary="real failure",
                dry_run=False,
                comment_runner=fake_comment_runner,
                gh_runner=FakeGhRunner(returncode=1),
            )

        assert called["generated"] is False

    def test_skips_gh_check_during_dry_run(self, monkeypatch):
        async def fake_comment_runner(**kwargs) -> str:
            return "review body"

        monkeypatch.setattr(
            "result_companion.core.review.pr_reviewer.load_review_config",
            lambda *_: make_review_config(),
        )

        result = run_review(
            repo_name="owner/repo",
            pr_number=5,
            failure_summary="real failure",
            dry_run=True,
            comment_runner=fake_comment_runner,
            gh_runner=FakeGhRunner(returncode=1),
        )

        assert result == "review body"

    def test_posts_comment_when_not_dry_run(self, monkeypatch):
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
            failure_summary="real failure",
            dry_run=False,
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
                failure_summary="real failure",
                dry_run=True,
                comment_runner=fake_comment_runner,
            )
