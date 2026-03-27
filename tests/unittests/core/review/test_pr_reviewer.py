"""Tests for PR review orchestration."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from result_companion.core.results.text_report import AnalyzeReport
from result_companion.core.review.pr_reviewer import (
    Spinner,
    _all_passed_comment,
    _generate_review_comment,
    build_review_prompt,
    ensure_gh_auth,
    on_post_tool_use,
    on_pre_tool_use,
    post_comment,
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


class FakeCopilotSession:
    """Fake Copilot session that returns a canned response."""

    def __init__(self, content: str = "review comment"):
        self._content = content
        self.sent_prompts: list[str] = []

    async def send_and_wait(self, payload, timeout=None):
        self.sent_prompts.append(payload["prompt"])
        if not self._content:
            return None
        return SimpleNamespace(data=SimpleNamespace(content=self._content))


class FakeCopilotClient:
    """Fake CopilotClient with injectable session response."""

    def __init__(self, content: str = "review comment"):
        self._session = FakeCopilotSession(content)
        self.session_configs: list[dict] = []
        self.stopped = False

    async def create_session(self, config):
        self.session_configs.append(config)
        return self._session


class TestGenerateReviewComment:
    """Tests for _generate_review_comment function."""

    @pytest.fixture(autouse=True)
    def _patch_copilot_lifecycle(self, monkeypatch):
        async def noop_start(client, startup_timeout=30):
            pass

        async def noop_stop(client):
            client.stopped = True

        monkeypatch.setattr(
            "result_companion.core.review.pr_reviewer.start_copilot_client",
            noop_start,
        )
        monkeypatch.setattr(
            "result_companion.core.review.pr_reviewer.stop_copilot_client",
            noop_stop,
        )

    @pytest.mark.asyncio
    async def test_returns_response_content(self):
        client = FakeCopilotClient(content="found regression")
        result = await _generate_review_comment(
            repo_name="owner/repo",
            pr_number=1,
            failure_summary="test failed",
            config=make_review_config(),
            quiet=True,
            client_factory=lambda: client,
        )

        assert result == "found regression"

    @pytest.mark.asyncio
    async def test_returns_empty_on_no_response(self):
        client = FakeCopilotClient(content="")
        result = await _generate_review_comment(
            repo_name="owner/repo",
            pr_number=1,
            failure_summary="test failed",
            config=make_review_config(),
            quiet=True,
            client_factory=lambda: client,
        )

        assert result == ""

    @pytest.mark.asyncio
    async def test_includes_mcp_config_when_url_set(self):
        client = FakeCopilotClient()
        await _generate_review_comment(
            repo_name="owner/repo",
            pr_number=1,
            failure_summary="test failed",
            config=make_review_config(),
            quiet=True,
            client_factory=lambda: client,
        )

        session_cfg = client.session_configs[0]
        assert "mcp_servers" in session_cfg
        assert session_cfg["mcp_servers"]["github"]["url"] == "https://example.com/mcp"

    @pytest.mark.asyncio
    async def test_skips_mcp_config_when_no_url(self):
        client = FakeCopilotClient()
        config = make_review_config()
        config.review.mcp_server_url = ""

        await _generate_review_comment(
            repo_name="owner/repo",
            pr_number=1,
            failure_summary="test failed",
            config=config,
            quiet=True,
            client_factory=lambda: client,
        )

        assert "mcp_servers" not in client.session_configs[0]

    @pytest.mark.asyncio
    async def test_stops_client_even_on_error(self):
        client = FakeCopilotClient()
        client.create_session = lambda _: (_ for _ in ()).throw(RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            await _generate_review_comment(
                repo_name="owner/repo",
                pr_number=1,
                failure_summary="test failed",
                config=make_review_config(),
                quiet=True,
                client_factory=lambda: client,
            )

        assert client.stopped is True

    @pytest.mark.asyncio
    async def test_sends_formatted_prompt(self):
        client = FakeCopilotClient()
        await _generate_review_comment(
            repo_name="owner/repo",
            pr_number=42,
            failure_summary="login failed",
            config=make_review_config(),
            quiet=True,
            client_factory=lambda: client,
        )

        prompt = client._session.sent_prompts[0]
        assert "owner/repo" in prompt
        assert "42" in prompt
        assert "login failed" in prompt


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

    def test_model_override_applied_to_config(self, monkeypatch):
        captured = {}

        async def fake_comment_runner(**kwargs) -> str:
            captured["config"] = kwargs["config"]
            return "review body"

        monkeypatch.setattr(
            "result_companion.core.review.pr_reviewer.load_review_config",
            lambda *_: make_review_config(),
        )

        run_review(
            repo_name="owner/repo",
            pr_number=5,
            summary=make_failure_summary(),
            model="gpt-5",
            comment_runner=fake_comment_runner,
        )

        assert captured["config"].review.model == "gpt-5"

    def test_saves_generated_comment_to_output_file(self, tmp_path, monkeypatch):
        async def fake_comment_runner(**kwargs) -> str:
            return "generated review"

        monkeypatch.setattr(
            "result_companion.core.review.pr_reviewer.load_review_config",
            lambda *_: make_review_config(),
        )
        out = tmp_path / "review.md"

        run_review(
            repo_name="owner/repo",
            pr_number=5,
            summary=make_failure_summary(),
            output_path=str(out),
            comment_runner=fake_comment_runner,
        )

        assert out.read_text() == "generated review"


class TestAllPassedComment:
    """Tests for _all_passed_comment formatting."""

    def test_includes_only_present_metadata(self):
        full = AnalyzeReport(
            failed_test_count=0,
            analyzed_tests=[],
            total_test_count=42,
            source_file="output.xml",
            source_hash="abc123",
            timestamp="2025-01-01T00:00:00",
        )

        full_result = _all_passed_comment(full)

        assert "All Robot Framework tests passed." in full_result
        assert "- Total tests: 42" in full_result
        assert "- Source: `output.xml`" in full_result
        assert "- Data hash: `abc123`" in full_result
        assert "- Analyzed at: 2025-01-01T00:00:00" in full_result


class TestSpinner:
    """Tests for Spinner context manager."""

    def test_enabled_spinner_starts_and_stops_thread(self):
        with Spinner("testing", enabled=True) as s:
            assert s._thread is not None
            assert s._thread.is_alive()
        assert not s._thread.is_alive()

    def test_disabled_spinner_does_not_start_thread(self):
        with Spinner("testing", enabled=False) as s:
            assert s._thread is None

    def test_spinner_cleans_up_line_on_exit(self, capsys):
        with Spinner("cleanup", enabled=True):
            pass
        captured = capsys.readouterr()
        assert "\r\033[K" in captured.err


class FakePostRunner:
    """Subprocess-like runner that records calls and supports check=True."""

    def __init__(self, fail: bool = False):
        self._fail = fail
        self.calls: list[list[str]] = []

    def run(self, command: list[str], **kwargs):
        self.calls.append(command)
        if self._fail and kwargs.get("check"):
            raise subprocess.CalledProcessError(1, command)


class TestPostComment:
    """Tests for post_comment function."""

    def test_posts_with_correct_gh_args(self):
        runner = FakePostRunner()
        post_comment("owner/repo", 42, "body text", runner=runner)

        cmd = runner.calls[0]
        assert cmd[:2] == ["gh", "pr"]
        assert "42" in cmd
        assert "owner/repo" in cmd

    def test_cleans_up_temp_file_on_success(self):
        runner = FakePostRunner()
        post_comment("owner/repo", 1, "body", runner=runner)

        body_file = runner.calls[0][runner.calls[0].index("--body-file") + 1]
        assert not Path(body_file).exists()

    def test_cleans_up_temp_file_on_failure(self):
        runner = FakePostRunner(fail=True)

        with pytest.raises(subprocess.CalledProcessError):
            post_comment("owner/repo", 1, "body", runner=runner)

        body_file = runner.calls[0][runner.calls[0].index("--body-file") + 1]
        assert not Path(body_file).exists()

    def test_writes_comment_body_to_temp_file(self):
        written_content = {}

        class CapturingRunner:
            def run(self, command, **kwargs):
                idx = command.index("--body-file") + 1
                written_content["body"] = Path(command[idx]).read_text()

        post_comment("owner/repo", 1, "my review", runner=CapturingRunner())

        assert written_content["body"] == "my review"


class TestOnPreToolUse:
    """Tests for on_pre_tool_use hook."""

    @pytest.mark.asyncio
    async def test_returns_allow_decision(self):
        result = await on_pre_tool_use(
            {"toolName": "get_file", "toolArgs": {"path": "a.py"}}, None
        )

        assert result == {"permissionDecision": "allow"}

    @pytest.mark.asyncio
    async def test_handles_missing_tool_args(self):
        result = await on_pre_tool_use({"toolName": "list_prs"}, None)

        assert result == {"permissionDecision": "allow"}


class TestOnPostToolUse:
    """Tests for on_post_tool_use hook."""

    @pytest.mark.asyncio
    async def test_passes_through_tool_result(self):
        result = await on_post_tool_use(
            {"toolName": "get_file", "toolResult": "file content"}, None
        )

        assert result == {"toolResult": "file content"}

    @pytest.mark.asyncio
    async def test_handles_missing_tool_result(self):
        result = await on_post_tool_use({"toolName": "list_prs"}, None)

        assert result == {"toolResult": ""}

    @pytest.mark.asyncio
    async def test_truncates_long_result_in_log(self, caplog):
        import logging

        long_text = "x" * 500
        with caplog.at_level(logging.DEBUG):
            await on_post_tool_use({"toolName": "read", "toolResult": long_text}, None)

        log_line = [r for r in caplog.records if "tool_call_end" in r.message][0]
        assert len(log_line.message) < 500


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
