# PR Review Guide

`result-companion review` turns the JSON report from `analyze` into a PR review comment.
Currently limited to the GitHub Copilot flow (lowest-friction native agent integration). Support for other providers may be added in future versions.

## What It Does

1. `result-companion analyze --json-report` creates a structured failure report.
2. `result-companion review` parses the JSON and sends a text summary to a Copilot agent.
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
result-companion analyze -o output.xml --json-report rc_summary.json

# 2. Print the generated review comment without posting it
result-companion review \
  -s rc_summary.json \
  --repo owner/repo \
  --pr 65 \
  --preview

# 3. Post the review comment to the PR
result-companion review \
  -s rc_summary.json \
  --repo owner/repo \
  --pr 65

# 4. Post an all-clear comment when all tests pass
result-companion review \
  -s rc_summary.json \
  --repo owner/repo \
  --pr 65 \
  --notify-on-pass
```

`--preview` still calls Copilot. It only skips the `gh pr comment` step.

## Behaviour by Scenario

| Failures | `--notify-on-pass` | `--preview` | What happens |
|:-:|:-:|:-:|---|
| ✅ | any | ❌ | Copilot runs → comment posted to PR |
| ✅ | any | ✅ | Copilot runs → comment printed, nothing posted |
| ❌ | ❌ | any | Skipped silently |
| ❌ | ✅ | ❌ | All-clear comment posted (no Copilot call) |
| ❌ | ✅ | ✅ | All-clear comment printed, nothing posted |

## GitHub Actions

`--repo` defaults to `GITHUB_REPOSITORY`, so in Actions you usually only need `--pr`:

```yaml
- run: |
    result-companion analyze -o output.xml --json-report rc_summary.json
    result-companion review -s rc_summary.json --pr ${{ github.event.pull_request.number }}
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

If the summary file is not valid JSON (e.g. a plain text file), `review` exits with:

```text
Review failed: Invalid summary: expected JSON from 'analyze --json-report'.
```

If the generated comment is empty, `result-companion review` fails instead of silently
posting nothing.

## JSON Report Structure

`--json-report` produces a structured file consumed by `review`:

```json
{
  "failed_test_count": 1,
  "analyzed_tests": ["Login With Valid Credentials"],
  "per_test_results": {"Login With Valid Credentials": "Root cause: 503 from backend..."},
  "overall_summary": "Backend service unavailable during login flow.",
  "model": "openai/gpt-4",
  "source_file": "output.xml",
  "total_test_count": 12,
  "source_hash": "a1b2c3d4e5f6"
}
```

| Field | Description |
|-------|-------------|
| `failed_test_count` | Number of failed tests analyzed by LLM |
| `analyzed_tests` | List of analyzed test names |
| `per_test_results` | LLM analysis per test (name → text) |
| `overall_summary` | Cross-test failure synthesis (optional) |
| `model` | LLM model used for analysis |
| `source_file` | Path to input `output.xml` |
| `total_test_count` | Total tests before pass/fail filtering |
| `source_hash` | SHA-256 prefix of raw test data (traceability) |

## Limitations

- Review is Copilot-only.
- `gh pr comment` cannot upload `rc_log.html` as a PR attachment.
- If you want reviewers to download `rc_log.html`, publish it separately as a CI artifact
  or provide your own hosted link in workflow output or PR discussion.
