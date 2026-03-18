"""PR review: correlates test failures with code changes via Copilot agent."""

import asyncio
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
    dry_run: bool,
    prompts: ReviewPromptModel,
) -> str:
    """Builds the Copilot agent prompt from config template and runtime values.

    Args:
        repo_name: GitHub repo in "owner/repo" format.
        pr_number: Pull request number.
        failure_summary: Plaintext failure output from result-companion.
        dry_run: If True, agent prints the comment instead of posting it.
        prompts: Review prompt configuration.

    Returns:
        Formatted prompt for the Copilot agent.
    """
    action = (
        prompts.dry_run_action
        if dry_run
        else prompts.post_action_template.format(
            pr_number=pr_number, repo_name=repo_name
        )
    )
    return prompts.review_prompt.format(
        repo_name=repo_name,
        pr_number=pr_number,
        failure_summary=failure_summary,
        action=action,
    )


async def _run_review_async(
    repo_name: str,
    pr_number: int,
    failure_summary: str,
    config: ReviewConfigModel,
    dry_run: bool = True,
) -> str:
    """Runs Copilot agent to analyze PR diff against test failures.

    Args:
        repo_name: GitHub repo in "owner/repo" format.
        pr_number: Pull request number to review.
        failure_summary: Output from result-companion analyze.
        config: Validated review configuration.
        dry_run: If True, prints comment instead of posting to PR.

    Returns:
        Agent response text.
    """
    prompt = build_review_prompt(
        repo_name, pr_number, failure_summary, dry_run, config.review
    )

    async def on_pre_tool_use(tool_input, _invocation):
        logger.debug(
            f"[review ->] {tool_input['toolName']}"
            f"  args={tool_input.get('toolArgs', {})}"
        )
        return {"permissionDecision": "allow"}

    async def on_post_tool_use(tool_input, _invocation):
        preview = str(tool_input.get("toolResult", ""))[:300]
        logger.debug(f"[review <-] {tool_input['toolName']}" f"  result={preview}")

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
        Agent response text.
    """
    config = load_review_config(config_path)
    if model:
        config.review.model = model
    return asyncio.run(
        _run_review_async(
            repo_name=repo_name,
            pr_number=pr_number,
            failure_summary=failure_summary,
            config=config,
            dry_run=dry_run,
        )
    )
