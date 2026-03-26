"""PR review: correlates test failures with code changes via Copilot agent."""

import asyncio
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from copilot import CopilotClient, PermissionHandler

from result_companion.core.copilot_client import (
    start_copilot_client,
    stop_copilot_client,
)
from result_companion.core.parsers.config import (
    ReviewConfigModel,
    ReviewPromptModel,
    load_review_config,
)
from result_companion.core.results.text_report import AnalyzeReport
from result_companion.core.utils.logging_config import logger


class Spinner:
    """Simple stderr spinner for long-running operations."""

    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, message: str = "Working", enabled: bool = True):
        self._message = message
        self._enabled = enabled
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        if self._enabled:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_):
        if not self._enabled:
            return
        self._stop.set()
        if self._thread:
            self._thread.join()
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()

    def _spin(self):
        idx = 0
        while not self._stop.is_set():
            frame = self._FRAMES[idx % len(self._FRAMES)]
            sys.stderr.write(f"\r{frame} {self._message}")
            sys.stderr.flush()
            idx += 1
            time.sleep(0.1)


def build_review_prompt(
    repo_name: str,
    pr_number: int,
    failure_summary: str,
    prompts: ReviewPromptModel,
) -> str:
    """Builds the Copilot agent prompt from config template and runtime values.

    Args:
        repo_name: GitHub repo in "owner/repo" format.
        pr_number: Pull request number.
        failure_summary: Plaintext failure output from result-companion.
        prompts: Review prompt configuration.

    Returns:
        Formatted prompt for the Copilot agent.
    """
    return prompts.review_prompt.format(
        repo_name=repo_name,
        pr_number=pr_number,
        failure_summary=failure_summary,
    )


def post_comment(
    repo_name: str,
    pr_number: int,
    comment_body: str,
    runner: Any = subprocess,
) -> None:
    """Posts a comment to a GitHub PR via gh CLI.

    Args:
        repo_name: GitHub repo in "owner/repo" format.
        pr_number: Pull request number.
        comment_body: Markdown comment text.
        runner: Subprocess module (injectable for testing).
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(comment_body)
        body_file = f.name

    try:
        logger.info(f"Posting comment to {repo_name}#{pr_number}...")
        runner.run(
            [
                "gh",
                "pr",
                "comment",
                str(pr_number),
                "--repo",
                repo_name,
                "--body-file",
                body_file,
            ],
            check=True,
        )
        logger.info("Comment posted successfully.")
    finally:
        Path(body_file).unlink(missing_ok=True)


def ensure_gh_auth(runner: Any = subprocess) -> None:
    """Validates that the GitHub CLI is installed and authenticated.

    Args:
        runner: Subprocess-like module used for command execution.

    Raises:
        RuntimeError: If `gh` is missing or not authenticated.
    """
    try:
        result = runner.run(
            ["gh", "auth", "status"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "GitHub CLI is not installed. Install `gh` before posting review comments."
        ) from exc

    if result.returncode == 0:
        return

    raise RuntimeError("GitHub CLI is not authenticated. Run: gh auth login")


async def _generate_review_comment(
    repo_name: str,
    pr_number: int,
    failure_summary: str,
    config: ReviewConfigModel,
    quiet: bool = False,
) -> str:
    """Runs Copilot agent with GitHub MCP to generate a review comment.

    Args:
        repo_name: GitHub repo in "owner/repo" format.
        pr_number: Pull request number to review.
        failure_summary: Output from result-companion analyze.
        config: Validated review configuration.
        quiet: If True, suppresses spinner output.

    Returns:
        Generated review comment text.
    """
    prompt = build_review_prompt(repo_name, pr_number, failure_summary, config.review)

    async def on_pre_tool_use(tool_input, _invocation):
        logger.debug(
            f"[tool_call_start ->] {tool_input['toolName']}"
            f"  args={tool_input.get('toolArgs', {})}"
        )
        return {"permissionDecision": "allow"}

    async def on_post_tool_use(tool_input, _invocation):
        preview = str(tool_input.get("toolResult", ""))[:300]
        logger.debug(
            f"[tool_call_end <-] {tool_input['toolName']}" f"  result={preview}"
        )
        return {"toolResult": tool_input.get("toolResult", "")}

    model = config.review.model
    timeout = config.review.timeout
    logger.info(f"Starting Copilot review agent (model={model})...")

    client = CopilotClient()
    await start_copilot_client(
        client,
        startup_timeout=config.review.startup_timeout,
    )
    try:
        session_config: dict[str, Any] = {
            "model": model,
            "on_permission_request": PermissionHandler.approve_all,
            "hooks": {
                "on_pre_tool_use": on_pre_tool_use,
                "on_post_tool_use": on_post_tool_use,
            },
        }
        if config.review.mcp_server_url:
            session_config["mcp_servers"] = {
                "github": {
                    "type": "sse",
                    "url": config.review.mcp_server_url,
                    "tools": ["*"],
                },
            }

        session = await client.create_session(session_config)
        logger.info(
            f"Copilot session created. Sending prompt" f" ({len(prompt)} chars)..."
        )
        with Spinner("Copilot agent is analyzing PR...", enabled=not quiet):
            response = await session.send_and_wait({"prompt": prompt}, timeout=timeout)
        logger.info("Review response received.")
        if response and response.data:
            return response.data.content
        return ""
    finally:
        await stop_copilot_client(client)


_ALL_PASSED_COMMENT = "✅ **result-companion:** All Robot Framework tests passed."


def run_review(
    repo_name: str,
    pr_number: int,
    summary: str,
    config_path: Path | None = None,
    preview: bool = True,
    notify_on_pass: bool = False,
    model: str | None = None,
    output_path: str | None = None,
    quiet: bool = False,
    comment_runner: Callable[..., Any] | None = None,
    gh_runner: Any = subprocess,
    comment_poster: Callable[..., None] = post_comment,
) -> str:
    """Sync entry point for PR review.

    Args:
        repo_name: GitHub repo in "owner/repo" format.
        pr_number: Pull request number to review.
        summary: Raw JSON content from 'analyze --json-report'.
        config_path: Optional user config YAML override.
        preview: If True, prints comment instead of posting to PR.
        notify_on_pass: If True, posts a short all-clear comment when no failures found.
        model: Override model from config.
        output_path: Optional path to write the review comment as a file.
        quiet: If True, suppresses spinner output.
        comment_runner: Injectable comment generator for tests.
        gh_runner: Injectable subprocess-like module for gh checks.
        comment_poster: Injectable PR comment poster for tests.

    Returns:
        Generated review comment text.

    Raises:
        RuntimeError: If summary is not valid JSON from 'analyze --json-report'.
    """
    try:
        report = AnalyzeReport.from_json(summary)
    except (ValueError, KeyError, TypeError) as e:
        raise RuntimeError(
            f"Invalid summary: expected JSON from 'analyze --json-report'. ({e})"
        ) from e

    if not report.has_failures():
        if not notify_on_pass:
            logger.info("No test failures found — skipping review.")
            return ""
        if not preview:
            ensure_gh_auth(gh_runner)
            comment_poster(repo_name, pr_number, _ALL_PASSED_COMMENT, runner=gh_runner)
        return _ALL_PASSED_COMMENT

    config = load_review_config(config_path)
    if model:
        config.review.model = model

    if not preview:
        ensure_gh_auth(gh_runner)

    generator = comment_runner or _generate_review_comment
    comment = asyncio.run(
        generator(
            repo_name=repo_name,
            pr_number=pr_number,
            failure_summary=report.to_text(),
            config=config,
            quiet=quiet,
        )
    )
    if not comment.strip():
        raise RuntimeError("Copilot review returned an empty comment.")

    if not preview and comment:
        comment_poster(repo_name, pr_number, comment, runner=gh_runner)

    if output_path and comment:
        save_review(output_path, comment)

    return comment


def save_review(output_path: str, content: str) -> None:
    """Writes review comment to file, warns if not Markdown.

    Args:
        output_path: Destination file path.
        content: Review comment text.
    """
    out = Path(output_path)
    if out.suffix not in (".md", ".markdown"):
        logger.warning(f"Output file '{out.name}' is not a Markdown file (.md)")
    out.write_text(content)
    logger.info(f"Review written to {out.resolve()}")
