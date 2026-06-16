---
name: result-companion
description: Guides agents using result-companion for Robot Framework output.xml analysis, token-aware log chunking, field filtering, text or JSON reports, and PR review context. Use when analyzing Robot Framework failures, tuning result-companion configs, or preparing agentic CI/debug workflows.
disable-model-invocation: true
---

# Result Companion

Use this skill when user invokes `/result-companion` or asks for agentic Robot Framework failure analysis with this package.

## Start Here

Read [AGENTS.md](../../../AGENTS.md) first. It is the source of truth for workflows, code paths, config rules, and validation commands.

## Commands

Create stable chat or CI context:

```bash
result-companion analyze -o output.xml --text-report rc_summary.txt --no-html-report
```

Create structured PR review context:

```bash
result-companion analyze -o output.xml --json-report rc_summary.json
result-companion review -s rc_summary.json --repo owner/repo --pr 65 --preview -o review.md
```

Inspect chunking without LLM calls:

```bash
result-companion analyze -o output.xml --dryrun
```

## Tune Signal

- Filter tests with `--include`, `--exclude`, `include_tags`, and `exclude_tags`.
- Reduce tokens with `rendering.exclude_fields`.
- Keep `message` unless task only needs pass/fail structure.
- Use [examples/EXAMPLES.md](../../../examples/EXAMPLES.md) for balanced and skeleton field presets.
- Use `--debug-log debug.log` to inspect prompts and responses while tuning.
