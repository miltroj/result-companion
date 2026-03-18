"""PR review: correlates test failures with code changes via Copilot agent."""

import asyncio
import subprocess
import tempfile
from pathlib import Path

from copilot import CopilotClient, PermissionHandler

from result_companion.core.parsers.config import (
    ReviewConfigModel,
    ReviewPromptModel,
    load_review_config,
)
from result_companion.core.utils.logging_config import logger


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
    runner: type = subprocess,
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


async def _generate_review_comment(
    repo_name: str,
    pr_number: int,
    failure_summary: str,
    config: ReviewConfigModel,
) -> str:
    """Runs Copilot agent with GitHub MCP to generate a review comment.

    Args:
        repo_name: GitHub repo in "owner/repo" format.
        pr_number: Pull request number to review.
        failure_summary: Output from result-companion analyze.
        config: Validated review configuration.

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
    await client.start()
    try:
        session = await client.create_session(
            {
                "model": model,
                "on_permission_request": PermissionHandler.approve_all,
                "hooks": {
                    "on_pre_tool_use": on_pre_tool_use,
                    "on_post_tool_use": on_post_tool_use,
                },
                "mcp_servers": {
                    "github": {
                        "type": "sse",
                        "url": config.review.mcp_server_url,
                        "tools": ["*"],
                    },
                },
            }
        )
        logger.info(
            f"Copilot session created. Sending prompt" f" ({len(prompt)} chars)..."
        )
        response = await session.send_and_wait({"prompt": prompt}, timeout=timeout)
        logger.info("Review response received.")
        if response and response.data:
            return response.data.content
        return ""
    finally:
        await client.stop()


def run_review(
    repo_name: str,
    pr_number: int,
    failure_summary: str,
    config_path: Path | None = None,
    dry_run: bool = True,
    model: str | None = None,
) -> str:
    """Sync entry point for PR review.

    Args:
        repo_name: GitHub repo in "owner/repo" format.
        pr_number: Pull request number to review.
        failure_summary: Output from result-companion analyze.
        config_path: Optional user config YAML override.
        dry_run: If True, prints comment instead of posting to PR.
        model: Override model from config.

    Returns:
        Generated review comment text.
    """
    config = load_review_config(config_path)
    if model:
        config.review.model = model

    comment = asyncio.run(
        _generate_review_comment(
            repo_name=repo_name,
            pr_number=pr_number,
            failure_summary=failure_summary,
            config=config,
        )
    )

    if not dry_run and comment:
        post_comment(repo_name, pr_number, comment)

    return comment
