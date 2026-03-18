"""PR review: correlates test failures with code changes via Copilot agent."""

from copilot import CopilotClient, PermissionHandler

from result_companion.core.utils.logging_config import logger


def build_review_prompt(
    repo_name: str, pr_number: int, failure_summary: str, dry_run: bool
) -> str:
    """Builds the Copilot agent prompt combining PR context with test failures.

    Args:
        repo_name: GitHub repo in "owner/repo" format.
        pr_number: Pull request number.
        failure_summary: Plaintext failure output from result-companion analyze.
        dry_run: If True, agent prints the comment instead of posting it.

    Returns:
        Formatted prompt for the Copilot agent.
    """
    action = (
        "Print the review comment body only — do NOT run gh pr comment."
        if dry_run
        else (
            "Run this shell command to post the comment (prefix GH_TOKEN"
            " to avoid interactive auth): GH_TOKEN=$(gh auth token)"
            f" gh pr comment {pr_number} --repo {repo_name}"
            ' --body "<write the actual review text here>"'
        )
        # TODO: another issue I see is that agent is running @review.py (28-29)  which is a command line interface for github and it might be missing so we would need to consider eather programatic way on how to swap it or make sure ite beeing installed before even we run review
    )
    return (
        f"You are a QA assistant. Robot Framework tests failed after "
        f"PR #{pr_number} in {repo_name}.\n\n"
        f"FAILURE SUMMARY (result-companion output):\n"
        f"{failure_summary}\n\n"
        f"Do these steps in order:\n"
        f"1. Execute shell command: "
        f"gh pr diff {pr_number} --repo {repo_name}\n"
        f"2. Read the diff output and identify which changed lines "
        f"caused the failures above\n"
        f"3. {action}\n\n"
        f"Format the comment as clean GitHub Markdown:\n"
        f"- Use a `## 🔍 result-companion: Test Failure Analysis` "
        f"heading\n"
        f"- List each finding with the file and line reference\n"
        f"- Use bullet points per finding: **file**, **line**, "
        f"**why it caused the failure**\n"
        f"- End with a `## 💡 Suggested Fix` section with concrete "
        f"code or steps"
    )


async def run_review(
    repo_name: str,
    pr_number: int,
    failure_summary: str,
    dry_run: bool = True,
    model: str = "gpt-5-mini",
) -> str:
    """Runs Copilot agent to analyze PR diff against test failures.

    Args:
        repo_name: GitHub repo in "owner/repo" format.
        pr_number: Pull request number to review.
        failure_summary: Output from result-companion analyze.
        dry_run: If True, prints comment instead of posting to PR.
        model: Copilot model to use.

    Returns:
        Agent response text.
    """
    prompt = build_review_prompt(repo_name, pr_number, failure_summary, dry_run)

    async def on_pre_tool_use(tool_input, _invocation):
        logger.debug(
            "[review →] %s args=%s",
            tool_input["toolName"],
            tool_input.get("toolArgs", {}),
        )
        return {"permissionDecision": "allow"}

    async def on_post_tool_use(tool_input, _invocation):
        preview = str(tool_input.get("toolResult", ""))[:300]
        logger.debug(
            "[review ←] %s result=%s",
            tool_input["toolName"],
            preview,
        )

    logger.info("Starting Copilot review agent (model=%s)...", model)
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
            "Copilot session created. Sending prompt (%d chars)...",
            len(prompt),
        )
        response = await session.send_and_wait({"prompt": prompt}, timeout=300)
        logger.info("Review response received.")
        if response and response.data:
            return response.data.content
        return ""
    finally:
        await client.stop()
