# AGENTS.md

## Quick Read

Result Companion turns Robot Framework `output.xml` into filtered, token-aware, LLM-native failure analysis. Agents should treat `rc_summary.txt` and `rc_summary.json` as stable context for debugging, code fixes, CI triage, and PR review.

Use [README.md](README.md) for user setup, [examples/EXAMPLES.md](examples/EXAMPLES.md) for usage patterns, and this file for repo navigation and change rules.

## Agent Workflows

Generate compact chat or CI context:

```bash
result-companion analyze -o output.xml --text-report rc_summary.txt --no-html-report
```

Use the report as agent input:

```text
Read rc_summary.txt and rank failures by confidence and impact. Propose fix order.
```

Generate PR review context:

```bash
result-companion analyze -o output.xml --json-report rc_summary.json
result-companion review -s rc_summary.json --repo owner/repo --pr 65 --preview -o review.md
```

Use the public Python API for scripts:

```python
from result_companion import analyze
from result_companion.core.parsers.config import load_config

config = load_config("my_config.yaml")
result = analyze("output.xml", config=config, include_tags=["smoke"])
print(result.text_report)
```

Use low-level rendering when an agent needs raw LLM-ready test text before provider calls:

```python
from pathlib import Path

from result_companion.core.chunking.rf_results import get_rc_robot_results

results = get_rc_robot_results(
    Path("output.xml"),
    include_tags=["smoke"],
    exclude_fields=["elapsed_time", "doc", "lineno", "owner"],
)
for test_name, text in results.as_texts():
    print(test_name, text)
```

## Why This Is Agent-Native

- `rendering.exclude_fields` removes low-signal Robot Framework fields before the LLM sees them.
- Consecutive line dedup collapses noisy repeated logs into repeat-count annotations.
- Context-aware chunks repeat suite, test, and keyword breadcrumbs so each chunk can stand alone.
- Tag filters target particular tests through `--include`, `--exclude`, `include_tags`, and `exclude_tags`.
- Text and JSON reports are portable context for downstream coding agents and PR reviewers.

## Code Map

- Public API: [result_companion/api.py](result_companion/api.py), re-exported from [result_companion/__init__.py](result_companion/__init__.py).
- CLI: [result_companion/entrypoints/cli/cli_app.py](result_companion/entrypoints/cli/cli_app.py), script name from [pyproject.toml](pyproject.toml).
- Analyze orchestration: [result_companion/entrypoints/run_rc.py](result_companion/entrypoints/run_rc.py).
- Robot Framework parsing, tag filtering, field filtering, per-test text: [result_companion/core/chunking/rf_results.py](result_companion/core/chunking/rf_results.py).
- Token-aware chunking, breadcrumbs, repeated-line dedup: [result_companion/core/chunking/chunking.py](result_companion/core/chunking/chunking.py).
- Config models and env var expansion: [result_companion/core/parsers/config.py](result_companion/core/parsers/config.py).
- Default prompts, tokenizer budget, field excludes: [result_companion/core/configs/default_config.yaml](result_companion/core/configs/default_config.yaml).
- Text and JSON reports: [result_companion/core/results/text_report.py](result_companion/core/results/text_report.py).
- PR review flow: [result_companion/core/review/pr_reviewer.py](result_companion/core/review/pr_reviewer.py) and [examples/PR_REVIEW.md](examples/PR_REVIEW.md).

## Config Guidance

- Prefer YAML config plus `DefaultConfigModel` validation over ad hoc core flags.
- User config overlays [result_companion/core/configs/default_config.yaml](result_companion/core/configs/default_config.yaml).
- Secrets may use `${ENV_VAR}` expansion. Never commit API keys or generated debug logs containing prompts and responses.
- Main analysis response contract lives in `llm_config.question_prompt`.
- Large-test prompts live under `llm_config.chunking.chunk_analysis_prompt` and `llm_config.chunking.final_synthesis_prompt`.
- Provider examples live in [examples/configs](examples/configs); link them instead of duplicating setup.

## Change Rules For Agents

- Keep edits small. Do not rewrite docs, prompts, or config shape unless task requires it.
- Preserve current package names, including historical `result_companion/core/analizers`.
- Prefer public API exports from `result_companion` or `result_companion.api` before `_internal` helpers.
- Do not bypass Robot Framework native tag filtering or Pydantic config validation.
- Add tests for code changes. Match existing test style: one scenario per test, simple fakes over complex mocks.
- Public functions need type hints and Google-style docstrings.
- Keep nesting shallow and functions focused, following [CONTRIBUTING.md](CONTRIBUTING.md).

## Test And Validation Commands

```bash
make install
make test-unit
make test-integration
make lint
make format
```

Use `make test-e2e` only when external services are available, such as Copilot CLI or Ollama.

For prompt and chunk debugging:

```bash
result-companion analyze -o output.xml --dryrun
result-companion analyze -o output.xml -c my_config.yaml --debug-log debug.log
```

## Reference Links

- [README.md](README.md): user quick start and value proposition.
- [examples/EXAMPLES.md](examples/EXAMPLES.md): field filtering, token-aware chunking, custom prompts, agent chat workflows, and programmatic API.
- [examples/PR_REVIEW.md](examples/PR_REVIEW.md): analyze-to-review flow for PR comments.
- [CONTRIBUTING.md](CONTRIBUTING.md): code standards and testing requirements.
- [examples/configs](examples/configs): provider and tag filtering configs.
