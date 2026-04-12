# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.15] - 2026-04-11

### Changed
- **Context-aware chunking** for LLM input: nested suites → test → keywords/messages; each chunk repeats ancestor lines and uses `{...}` for continuations (plus long-line splits when needed)—replaces `str(test_case)` + overlap.

  ```text
  Suite: Outer Suite - FAIL
      Suite: Middle Suite - FAIL
          Suite: Inner Suite - FAIL
              Test: Example workflow test - FAIL (120.0s)
                  Keyword: Run job and wait - PASS (100.0s)
                      Keyword: Log - PASS (5ms)
                          {...}
  ```

- Consecutive identical rendered lines (same depth and text) collapse to one line with `(repeats ×N)` before token sizing and chunking.

### Fixed
- HTML report: LLM UI injection reads/writes the log with explicit UTF-8 so non-ASCII content is preserved on Windows (#71).

[0.0.15]: https://github.com/miltroj/result-companion/releases/tag/v0.0.15

## [0.0.14] - 2026-04-05

### Added
- User state under `~/.result-companion` (create on first use): `get_or_create_current_user_rc_state_dir()`, nested dirs via `get_or_create_rc_state_subdirectory(*parts)` with `CLI` / `APP` — exported from `result_companion` / `result_companion.core.state_dir`

[0.0.14]: https://github.com/miltroj/result-companion/releases/tag/v0.0.14

## [0.0.13] - 2026-04-04

### Added
- Programmatic API: `analyze()` and async `run_analysis()` returning `AnalysisResult` (includes `text_report`); re-exported from `result_companion`
- Programmatic `review()` — takes pre-loaded `AnalyzeReport` and `ReviewConfigModel`, optional dependency injection for tests

[0.0.13]: https://github.com/miltroj/result-companion/releases/tag/v0.0.13

## [0.0.12] - 2026-03-27

### Added
- `review` command — correlates test failures with PR code changes via Copilot agent and GitHub MCP
- `--json-report <file>` flag on `analyze` — produces structured JSON consumed by `review`
- `--preview` flag — prints generated review comment without posting to PR
- `--notify-on-pass` flag — posts an all-clear comment when no failures are found
- `--output` / `-o` flag on `review` — saves review comment to a file
- Fail-fast PR validation — `review` checks repo and PR existence via `gh pr view` before starting Copilot agent
- Review configuration via `default_review_config.yaml` with prompt template, model, timeout, and MCP server URL
- `examples/PR_REVIEW.md` — setup, usage, and GitHub Actions integration guide

### Changed
- Extracted shared Copilot client lifecycle helpers into `copilot_client.py` (reused by both `analyze` and `review`)
- Config loader gained generic `load_as()` method for loading into arbitrary Pydantic models
- Default analysis prompt now includes exact error message quoting when possible

[0.0.12]: https://github.com/miltroj/result-companion/releases/tag/v0.0.12

## [0.0.11] - 2026-03-12

### Added
- Copilot CLI startup timeout (30s) with fail-fast error on connection issues and lack of authentication

### Changed
- Overall failure summary is now **enabled by default**; disable with `--no-overall-summary`

[0.0.11]: https://github.com/miltroj/result-companion/releases/tag/v0.0.11

## [0.0.10] - 2026-03-10

### Added
- Rate limit handling in `CopilotLLM`: exponential backoff with configurable `max_retries` (default 3) and `retry_base_delay` (default 10s)
- Fail-fast on first-ever request rate limit — raises `RuntimeError` suggesting to reduce `max_content_tokens`

[0.0.10]: https://github.com/miltroj/result-companion/releases/tag/v0.0.10

## [0.0.9] - 2026-03-09

### Changed
- Dropped Python 3.10 support; minimum is now Python 3.11
- Upgraded `github-copilot-sdk` from `0.1.20` to `0.1.32`
- LLM router passes `num_retries` (default 3) to both LiteLLM and Copilot SDK for native retry handling
- Copilot session creation now sets `PermissionHandler.approve_all` automatically
- `_resolve_cli_path` falls back to SDK bundled binary when no explicit path is configured

[0.0.9]: https://github.com/miltroj/result-companion/releases/tag/v0.0.9

## [0.0.8] - 2026-03-06

### Changed
- Suite setup failure now surfaces the suite itself for analysis instead of individual tests, exposing the root cause directly
- Tests carry a `suite_context` chain (ancestor setups/teardowns) so the LLM has full execution context
- HTML report: LLM block prepended inside test children; suite elements processed alongside tests in polling interval
- Removed `__main__` dev scaffold from `html_creator.py`

### Added
- [`fixtures/robot/`](fixtures/robot/) — Robot Framework suites for generating `output.xml` used in manual and UAT testing (not part of CI)

[0.0.8]: https://github.com/miltroj/result-companion/releases/tag/v0.0.8

## [0.0.7] - 2026-03-02

### Fixed
- Copilot session pool: failed sessions are now destroyed and removed instead of being returned to the pool

[0.0.7]: https://github.com/miltroj/result-companion/releases/tag/v0.0.7

## [0.0.6] - 2026-02-27

### Added
- Plain-text report output via `--text-report <file>` and `--print-text-report` flags
- Overall LLM summary synthesis with `--overall-summary` — runs an extra LLM pass to produce a concise digest of all per-test analyses
- Overall summary section rendered in HTML reports when enabled
- `--quiet` / `-q` flag to suppress logs, progress bars, and CLI parameter echo for clean stdout redirects
- `--html-report` / `--no-html-report` toggle to skip HTML generation when only text output is needed

### Changed
- Config: added `summary_prompt_template` field to `llm_config` for controlling the synthesis prompt

[0.0.6]: https://github.com/miltroj/result-companion/releases/tag/v0.0.6

## [0.0.5] - 2026-02-17
- **BREAKING**: Migrated from LangChain to LiteLLM for LLM integrations
  - Replaced 7 LangChain packages with single `litellm` package
  - Config format changed: `model_type` replaced with `model` using LiteLLM naming (e.g., `ollama_chat/llama2`, `openai/gpt-4o`)
  - Simplified configuration: `api_key` and `api_base` now top-level in `llm_factory`
- Supports 100+ LLM providers via LiteLLM unified interface
- Reduced package dependencies significantly
- Copilot sdk LLM provider added

See [EXAMPLES.md](examples/EXAMPLES.md) for all provider configurations.

## [0.0.4] - 2026-02-08
- Brush up README and examples/ by @miltroj in #46
- Remove invoke by @miltroj in #47
- Fix problem with exec permissions for copilot binary by @miltroj in #48
- Bump version by @miltroj in #49

## [0.0.3] - 2026-02-03
- Introduce simple dry-run + stats footer by @miltroj in #43
- Add copilot support using official sdk + langchain like written adapter by @miltroj in #44

## [0.0.2] - 2026-01-25

### Fixed
- GitHub Actions permissions for automatic release creation

## [0.0.1] - 2026-01-25

### Added
- Initial PyPI release (#39)
- AI-powered Robot Framework test failure analysis (#1)
- Local Ollama support with automatic installation (#22)
- Ollama server management with graceful handling of edge cases (#19, #20)
- OpenAI API integration (#27)
- Azure OpenAI support
- Google Gemini API integration (#25)
- Anthropic Claude models support (#32)
- Custom endpoint support for any OpenAI-compatible API
- Tag-based test filtering functionality (#36)
- Environment variable support in configuration files (#25)
- HTML report generation with LLM insights (#29)
- Enhanced markdown parsing in HTML logs (#18)
- Progress bar for long-running operations (#26)
- User-configurable concurrency levels (#30)
- CLI commands: `analyze`, `setup ollama`, `setup model`
- Version command (`--version`, `-v`) (#17)
- Token redaction from logs for security (#38)
- Improved chunking logic and prompt management (#31)
- Multiple system prompts for better analysis (#37)
- Lazy imports for faster startup (#35)

### Changed
- Skip PASS tests by default during analysis (#4)
- Improved project structure and organization (#10)
- Enhanced logging framework with JSON lines support (#34)
- One global logger available across the package (#7)
- Simplified local Ollama execution (#19)
- Better CLI parsing with dependency injection (#6)
- Improved README and examples documentation (#33)
- Examples.md more prominently featured (#24)

### Fixed
- CLI invoke command issues (#23)
- Duplicated test names handling (#16)
- Logging to JSON lines in result_companion.config (#34)
- Missing Robot Framework jQuery libraries in HTML template (#2)

### Development
- GitHub Actions CI/CD workflow (#8)
- Pre-commit hooks with isort, black, and unit tests (#9)
- Code coverage reporting with codecov (#11)
- Comprehensive unit and integration tests
- Invoke tasks for running tests and local execution (#3)

### Documentation
- Comprehensive README with installation and usage instructions
- Detailed EXAMPLES.md with multiple use cases
- Contributing guidelines (CONTRIBUTING.md)
- Demo GIF showcasing functionality
- Apache 2.0 License

[0.0.2]: https://github.com/miltroj/result-companion/releases/tag/v0.0.2
[0.0.1]: https://github.com/miltroj/result-companion/releases/tag/v0.0.1
