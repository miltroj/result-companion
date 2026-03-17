"""PoC: Copilot agent fetches PR diff and posts review comment based on test failure summary.

Requirements:
    - Copilot CLI installed and authenticated:  copilot -i "/login"
    - gh CLI installed and authenticated:       gh auth login
    - pip install github-copilot-sdk           (already in pyproject.toml)
    - Python 3.11+

Usage:
    python poc_pr_review.py
"""

import asyncio

from copilot import CopilotClient, PermissionHandler


def build_agent_prompt(
    repo_name: str, pr_number: int, failure_summary: str, dry_run: bool
) -> str:
    """Builds the agent task prompt combining PR context with test failure findings.

    Args:
        repo_name: GitHub repo in "owner/repo" format.
        pr_number: Pull request number.
        failure_summary: Plaintext failure output from result-companion.
        dry_run: If True, agent prints the comment instead of posting it.

    Returns:
        Formatted prompt for the Copilot agent.
    """
    action = (
        "Print the review comment body only — do NOT run gh pr comment."
        if dry_run
        else f'Run this shell command to post the comment (prefix GH_TOKEN to avoid interactive auth): GH_TOKEN=$(gh auth token) gh pr comment {pr_number} --repo {repo_name} --body "<write the actual review text here>"'
    )
    return (
        f"You are a QA assistant. Robot Framework tests failed after PR #{pr_number} in {repo_name}.\n\n"
        f"FAILURE SUMMARY (result-companion output):\n{failure_summary}\n\n"
        f"Do these steps in order:\n"
        f"1. Execute shell command: gh pr diff {pr_number} --repo {repo_name}\n"
        f"2. Read the diff output and identify which changed lines caused the failures above\n"
        f"3. {action}\n\n"
        f"Format the comment as clean GitHub Markdown:\n"
        f"- Use a `## 🔍 result-companion: Test Failure Analysis` heading\n"
        f"- List each finding with the file and line reference as a fenced code block or inline `code` quote\n"
        f"- Use bullet points per finding: **file**, **line**, **why it caused the failure**\n"
        f"- End with a `## 💡 Suggested Fix` section with concrete code or steps"
    )


async def run_review(
    repo_name: str,
    pr_number: int,
    failure_summary: str,
    dry_run: bool = True,
    model: str = "gpt-5-mini",
) -> str:
    """Runs the Copilot agent to analyze a PR diff against test failures and comment.

    Args:
        repo_name: GitHub repo in "owner/repo" format.
        pr_number: Pull request number to review.
        failure_summary: Output from result-companion (LLM findings about test failures).
        dry_run: If True, prints the comment instead of posting it to the PR.
        model: Copilot model to use.

    Returns:
        Agent response text.
    """
    prompt = build_agent_prompt(repo_name, pr_number, failure_summary, dry_run)

    async def on_pre_tool_use(input, _invocation):
        print(f"[tool →] {input['toolName']}  args={input.get('toolArgs', {})}")
        return {"permissionDecision": "allow"}

    async def on_post_tool_use(input, _invocation):
        result_preview = str(input.get("toolResult", ""))[:300]
        print(f"[tool ←] {input['toolName']}  result={result_preview}")

    print(f"[poc] Starting Copilot client (model={model})...")
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
        print(f"[poc] Session created. Sending prompt ({len(prompt)} chars)...")
        response = await session.send_and_wait({"prompt": prompt}, timeout=300)
        print("[poc] Response received.")
        return response.data.content if response and response.data else ""
    finally:
        await client.stop()
        print("[poc] Client stopped.")


if __name__ == "__main__":
    FAILURE_SUMMARY = """
    Test: POC regression test
    Failed keyword: Should Login To Github
    Error: Invalid credentials
    """

    print("[poc] Reviewing PR (dry_run=True)...")
    result = asyncio.run(
        run_review(
            repo_name="miltroj/result-companion",
            pr_number=65,
            failure_summary=FAILURE_SUMMARY,
            dry_run=False,  # flip to False to actually post the comment
        )
    )
    print(f"\n[poc] Agent response:\n{result}")
