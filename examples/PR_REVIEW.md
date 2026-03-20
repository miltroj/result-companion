# PR Review Guide

`result-companion review` turns the text summary from `analyze` into a PR review comment.
It is intentionally limited to the GitHub Copilot flow. It does not work with Ollama,
OpenAI, or other LiteLLM providers.

## What It Does

1. `result-companion analyze --text-report` creates a compact failure summary.
2. `result-companion review` sends that summary to a Copilot agent.
3. The agent reads the PR through GitHub MCP and writes a Markdown review comment.
4. Python posts that comment with `gh pr comment`.

The read/write split is intentional:

- Copilot reads PR context through GitHub MCP.
- Python writes the final comment through `gh`.

## Prerequisites

You need both CLIs installed and authenticated:

macOS:

```bash
brew install gh
brew install copilot-cli
```

Linux — see [gh installation](https://github.com/cli/cli/blob/trunk/docs/install_linux.md) and [copilot-cli releases](https://github.com/github/gh-copilot/releases).

```bash
gh auth login
copilot -i "/login"
```

## Local Usage

```bash
# 1. Analyze Robot Framework results
result-companion analyze -o output.xml --text-report rc_summary.txt

# 2. Print the generated review comment without posting it
result-companion review \
  -s rc_summary.txt \
  --repo owner/repo \
  --pr 65 \
  --preview

# 3. Post the review comment to the PR
result-companion review \
  -s rc_summary.txt \
  --repo owner/repo \
  --pr 65
```

`--preview` still calls Copilot. It only skips the `gh pr comment` step.

If the summary contains no analyzed failures, review is skipped.

## GitHub Actions

`--repo` defaults to `GITHUB_REPOSITORY`, so in Actions you usually only need `--pr`:

```yaml
- run: |
    result-companion analyze -o output.xml --text-report rc_summary.txt
    result-companion review -s rc_summary.txt --pr ${{ github.event.pull_request.number }}
```

## Configuration

Default review config lives in `result_companion/core/configs/default_review_config.yaml`.
Override it with `--config custom.yaml`:

```yaml
review:
  model: "gpt-5-mini"
  timeout: 300
  startup_timeout: 30
  mcp_server_url: "https://api.enterprise.githubcopilot.com/mcp/readonly"
  review_prompt: |
    You are a QA assistant. Robot Framework tests failed after
    PR #{pr_number} in {repo_name}.
    ...
```

Available runtime placeholders:

- `{repo_name}`
- `{pr_number}`
- `{failure_summary}`

## Common Failures

If `gh` is missing:

```text
GitHub CLI is not installed. Install `gh` before posting review comments.
```

If `gh` is installed but unauthenticated:

```bash
gh auth login
```

If Copilot startup or auth fails:

```bash
copilot -i "/login"
```

If the generated comment is empty, `result-companion review` fails instead of silently
posting nothing.

## Limitations

- Review is Copilot-only.
- `gh pr comment` cannot upload `rc_log.html` as a PR attachment.
- If you want reviewers to download `rc_log.html`, publish it separately as a CI artifact
  or provide your own hosted link in workflow output or PR discussion.
