# Result Companion

**Turn your Robot Framework test failures into instant, actionable insights with AI.**

![Demo](https://raw.githubusercontent.com/miltroj/result-companion/main/assets/demo.gif)

## Why Result Companion?

Every QA engineer knows the pain: A test fails. You dig through logs. You trace keywords. You hunt for that one error message buried in thousands of lines. **Hours wasted.**

Result Companion changes that. It reads your `output.xml`, understands the entire test flow, and tells you exactly what went wrong and how to fix it—in seconds, not hours.

## What It Does

```bash
# Before: Manual debugging for hours
robot tests/                     # Test fails
# Now: Where did it fail? Why? What's the root cause?

# After: Instant AI analysis
result-companion -o output.xml   # Get answers in seconds
```

Your enhanced `log.html` now includes:
- **Root Cause Analysis**: Pinpoints the exact keyword and reason for failure
- **Test Flow Summary**: Understand what happened at a glance
- **Actionable Fixes**: Specific suggestions to resolve the issue

## Quick Start

### Option 1: Local AI (Free, Private)

```bash
pip install result-companion

# Auto-setup local AI model
result-companion setup ollama
result-companion setup model deepseek-r1:1.5b

# Analyze your tests
result-companion analyze -o output.xml
```

### Option 2: Cloud AI (OpenAI, Azure, Google)

```bash
pip install result-companion

# Configure and run (see examples/)
export OPENAI_API_KEY="your-key"
result-companion analyze -o output.xml -c examples/openai_config.yaml
```

## Filter Tests by Tags

Analyze only the tests you care about:

```bash
# Analyze smoke tests only
result-companion -o output.xml --include "smoke*"

# Exclude work-in-progress tests
result-companion -o output.xml --exclude "wip,draft*"

# Analyze critical tests (including passes)
result-companion -o output.xml --include "critical*" -i
```

Or use config file:
```yaml
test_filter:
  include_tags: ["smoke", "critical*"]
  exclude_tags: ["wip", "flaky"]
  include_passing: false  # Analyze failures only
```

See [examples/tag_filtering_config.yaml](https://github.com/miltroj/result-companion/blob/main/examples/tag_filtering_config.yaml) for details.

## Real Example

**Your test fails with:**
```
Login Test Suite
└── Login With Valid Credentials [FAIL]
```

**Result Companion tells you:**
```markdown
**Flow**
- Open browser to login page ✓
- Enter username "testuser" ✓
- Enter password ✓
- Click login button ✓
- Wait for dashboard [FAILED after 10s timeout]

**Failure Root Cause**
The keyword "Wait Until Page Contains Element" failed because
element 'id=dashboard' was not found. Server returned 503 error
in network logs at timestamp 14:23:45.

**Potential Fixes**
- Check if backend service is running and healthy
- Verify dashboard element selector hasn't changed
- Increase timeout if service startup is slow
```

## Beyond Error Analysis

Customize prompts for any use case:

```yaml
# security_audit.yaml
llm_config:
  question_prompt: |
    Find security issues: hardcoded passwords,
    exposed tokens, insecure configurations...
```

```yaml
# performance_review.yaml
llm_config:
  question_prompt: |
    Identify slow operations, unnecessary waits,
    inefficient loops...
```

See [examples/EXAMPLES.md](https://github.com/miltroj/result-companion/blob/main/examples/EXAMPLES.md) for more.

## Configuration Examples

Check [`examples/`](https://github.com/miltroj/result-companion/tree/main/examples) for ready-to-use configs:
- Local Ollama setup (default)
- OpenAI, Azure, Google Cloud
- Custom endpoints (Databricks, self-hosted)
- Prompt customization for security, performance, quality reviews

## Limitations

- Text-only analysis (no screenshots/videos)
- Large test suites processed in chunks
- **Local models**: Need 4-8GB RAM + GPU/NPU for good performance (Apple Silicon, NVIDIA, AMD)

## Contributing

Contributions welcome! Fork, create a feature branch, submit a pull request.

For bugs or feature requests, open an issue on GitHub.

## Development Setup

```bash
# Install with dev dependencies
poetry install --with=dev

# Install pre-commit hooks
poetry run pre-commit install

# Run tests
poetry run inv unittests
poetry run inv test_coverage
```

## License

Apache 2.0 - See [LICENSE](https://github.com/miltroj/result-companion/blob/main/LICENSE)

## Disclaimer

Cloud AI providers may process your test data. Local models (Ollama) keep everything private on your machine.

**You are responsible for data privacy.** The creator takes no responsibility for data exposure, intellectual property leakage, or security issues. By using Result Companion, you accept all risks and ensure compliance with your organization's data policies.
