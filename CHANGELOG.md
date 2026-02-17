# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
