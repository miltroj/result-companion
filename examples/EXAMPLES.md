# Examples & Configuration

Quick-start configurations for different LLM providers and use cases.

## Table of Contents
- [Examples \& Configuration](#examples--configuration)
  - [Table of Contents](#table-of-contents)
  - [GitHub Copilot (Recommended for Users With Copilot)](#github-copilot-recommended-for-users-with-copilot)
    - [Setup](#setup)
    - [Configuration](#configuration)
    - [Available Models](#available-models)
    - [Troubleshooting](#troubleshooting)
  - [Quick Configs](#quick-configs)
    - [Local (Ollama) - Default](#local-ollama---default)
    - [OpenAI](#openai)
    - [Azure OpenAI](#azure-openai)
    - [Google Gemini](#google-gemini)
    - [AWS Bedrock](#aws-bedrock)
    - [Custom OpenAI-Compatible Endpoint](#custom-openai-compatible-endpoint)
    - [Anthropic with Claude Models](#anthropic-with-claude-models)
  - [Test Filtering](#test-filtering)
    - [CLI Examples](#cli-examples)
    - [Config File](#config-file)
    - [Filter Logic](#filter-logic)
  - [Dryrun Mode](#dryrun-mode)
  - [Custom Analysis](#custom-analysis)
    - [Find Performance Issues](#find-performance-issues)
    - [Security Audit](#security-audit)
    - [Test Quality Review](#test-quality-review)
  - [Advanced Settings](#advanced-settings)
    - [Concurrency Control](#concurrency-control)
    - [Token Limits by Model](#token-limits-by-model)
    - [Chunking for Large Tests](#chunking-for-large-tests)
  - [Complete Example](#complete-example)
  - [Environment Variables Reference](#environment-variables-reference)
  - [Additional Resources](#additional-resources)

## GitHub Copilot (Recommended for Users With Copilot)

If you have GitHub Copilot (Business, Enterprise, or Pro+), it's the easiest way to get started—no API keys needed.

### Setup

**1. Install Copilot CLI**

```bash
# macOS/Linux (Homebrew)
brew install copilot-cli

# Or via npm (requires Node v22+)
npm install -g @github/copilot
```

**2. Authenticate**

```bash
copilot /login
```

Follow the prompts to log in with your GitHub account. Type `/exit` when done.

**3. Verify Setup**

```bash
copilot -p "/models" --allow-all -s
```

You should see available models (e.g., `gpt-4.1`, `claude-haiku-4.5`).

### Configuration
```yaml
# examples/configs/copilot_config.yaml
llm_factory:
  model: "copilot_sdk/gpt-4.1"
  # Alternative models:
  # model: "copilot_sdk/claude-sonnet-4.5"
  # model: "copilot_sdk/gpt-5"

tokenizer:
  tokenizer: openai_tokenizer
  max_content_tokens: 50000

concurrency:
  test_case: 3
  chunk: 2
```

Run with:
```bash
result-companion analyze -o output.xml -c examples/configs/copilot_config.yaml
```

### Available Models

| Model | Best For |
|-------|----------|
| `gpt-4.1` | General analysis (default) |
| `claude-haiku-4.5` | Fast, cost-effective |
| `gpt-5-mini` | Complex reasoning |

### Troubleshooting

| Issue | Solution |
|-------|----------|
| "Connection error" | Run `copilot` to re-authenticate |
| "Failed to list models" | Check network access to `*.githubcopilot.com` |
| CLI not found | Verify installation: `which copilot` |

---

## Quick Configs

### Local (Ollama) - Default

No config needed! Just run:
```bash
result-companion analyze -o output.xml
```

The default uses Ollama with `deepseek-r1:1.5b` model.

**Other MacBook-friendly models:**
- `phi-3-mini` (2.3GB, 8GB RAM) - Fast and efficient
- `mistral:7b` (4.1GB, 16GB RAM) - Industry standard, excellent quality

**Hardware note**: GPU/NPU significantly improves inference speed. Apple Silicon Macs use unified memory (RAM serves both CPU and GPU).

### OpenAI

```yaml
# examples/configs/openai_config.yaml
llm_factory:
  model: "openai/gpt-4o"
  api_key: "${OPENAI_API_KEY}"

tokenizer:
  tokenizer: openai_tokenizer
  max_content_tokens: 390000
```

Run with:
```bash
export OPENAI_API_KEY="sk-..."
result-companion analyze -o output.xml -c examples/configs/openai_config.yaml
```

### Azure OpenAI

```yaml
# examples/configs/azure_config.yaml
llm_factory:
  model: "azure/${AZURE_DEPLOYMENT_NAME}"
  api_key: "${AZURE_API_KEY}"
  api_base: "${AZURE_API_BASE}"
  parameters:
    api_version: "2023-05-15"

tokenizer:
  tokenizer: azure_openai_tokenizer
  max_content_tokens: 16000
```

Run with:
```bash
export AZURE_DEPLOYMENT_NAME="gpt-4"
export AZURE_API_BASE="https://myresource.openai.azure.com/"
export AZURE_API_KEY="..."
result-companion analyze -o output.xml -c examples/configs/azure_config.yaml
```

### Google Gemini

```yaml
# examples/configs/gemini_config.yaml
llm_factory:
  model: "gemini/gemini-2.0-flash"
  api_key: "${GOOGLE_API_KEY}"
  parameters:
    temperature: 0

tokenizer:
  tokenizer: google_tokenizer
  max_content_tokens: 1000000
```

Run with:
```bash
export GOOGLE_API_KEY="..."
result-companion analyze -o output.xml -c examples/configs/gemini_config.yaml
```

### AWS Bedrock

```yaml
# examples/configs/bedrock_config.yaml
llm_factory:
  model: "bedrock/${AWS_BEDROCK_MODEL_ID}"
  parameters:
    aws_region_name: "${AWS_REGION}"
    aws_access_key_id: "${AWS_ACCESS_KEY_ID}"
    aws_secret_access_key: "${AWS_SECRET_ACCESS_KEY}"

tokenizer:
  tokenizer: bedrock_tokenizer
  max_content_tokens: 100000
```

Run with:
```bash
export AWS_BEDROCK_MODEL_ID="anthropic.claude-v2"
export AWS_REGION="us-west-2"
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
result-companion analyze -o output.xml -c examples/configs/bedrock_config.yaml
```

### Custom OpenAI-Compatible Endpoint

For Databricks, self-hosted, or other OpenAI-compatible APIs:

```yaml
# examples/configs/custom_endpoint_config.yaml
llm_factory:
  model: "openai/${OPENAI_MODEL}"
  api_key: "${OPENAI_API_KEY}"
  api_base: "${OPENAI_BASE_URL}"

tokenizer:
  tokenizer: openai_tokenizer
  max_content_tokens: 16000
```

Run with:
```bash
export OPENAI_MODEL="your-model"
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://your-endpoint.com/v1"
result-companion analyze -o output.xml -c examples/configs/custom_endpoint_config.yaml
```

### Anthropic with Claude Models

```yaml
# examples/configs/anthropic_config.yaml
llm_factory:
  model: "anthropic/claude-3-5-haiku-latest"
  api_key: "${ANTHROPIC_API_KEY}"
  parameters:
    temperature: 0

tokenizer:
  tokenizer: anthropic_tokenizer
  max_content_tokens: 200000  # Claude supports 200K context window
```

**Note:** Set up the `ANTHROPIC_API_KEY` environment variable. Anthropic offers multiple Claude models:
- `claude-3-5-haiku-latest`: Lightweight, cheaper, faster (recommended for most cases)
- `claude-3-5-sonnet-latest`: Balanced performance and cost
- `claude-3-opus-latest`: Best reasoning capabilities


## Test Filtering

Focus analysis on specific tests using Robot Framework tag patterns.

### CLI Examples

```bash
# Analyze only smoke tests (failures only)
result-companion analyze -o output.xml --include "smoke*"

# Analyze critical tests including passes
result-companion analyze -o output.xml --include "critical*" -i

# Exclude WIP and known bugs
result-companion analyze -o output.xml --exclude "wip,bug-*"

# Combine filters
result-companion analyze -o output.xml --include "api,smoke" --exclude "flaky"
```

### Config File

```yaml
# examples/configs/tag_filtering_config.yaml
test_filter:
  include_tags: ["smoke*", "critical"]  # Wildcards supported
  exclude_tags: ["wip", "bug-*"]        # Exclude takes precedence
  include_passing: false                # false = failures only
```

Run with:
```bash
result-companion analyze -o output.xml -c examples/configs/tag_filtering_config.yaml
```

### Filter Logic

| Filters | Result |
|---------|--------|
| No filters | All failed tests |
| `--include smoke` | Failed tests with "smoke" tag |
| `--include smoke -i` | All tests (pass+fail) with "smoke" tag |
| `--exclude wip` | All failed tests except "wip" |
| Both | Include "smoke" AND exclude "wip" |

**Note**: Exclude patterns override include patterns.

## Dryrun Mode

Validate parsing and configuration without calling LLMs:

```bash
result-companion analyze -o output.xml --dryrun
```

Generates `rc_log.html` with debug metadata per test:
- Test name and status
- Chunk count and token usage
- Raw content length

**Use cases:**
- Debug XML parsing issues
- Verify tag filtering works correctly
- Check chunking behavior before real runs

## Custom Analysis

Customize prompts via `llm_config`:

| Option | Purpose |
|--------|---------|
| `question_prompt` | Main analysis prompt for each test |
| `chunking.chunk_analysis_prompt` | Prompt for analyzing individual chunks (large tests) |
| `chunking.final_synthesis_prompt` | Prompt for combining chunk summaries |

### Find Performance Issues

```yaml
# performance_config.yaml
llm_config:
  question_prompt: |
    Identify performance bottlenecks in this test:
    - Operations taking >5 seconds
    - Unnecessary waits or sleeps
    - Inefficient loops or repeated operations

    Output format:
    **Performance Issues**
    - [Operation] took [duration] - Suggestion: [improvement]
```

### Security Audit

```yaml
# security_config.yaml
llm_config:
  question_prompt: |
    Scan this test for security concerns:
    - Hardcoded passwords, tokens, or API keys
    - Unencrypted sensitive data
    - Exposed endpoints or credentials
    - Insecure configurations

    Mark severity: CRITICAL/HIGH/MEDIUM/LOW
```

### Test Quality Review

```yaml
# quality_config.yaml
llm_config:
  question_prompt: |
    Assess test quality:
    - Missing or weak assertions
    - Poor test data or hard-coded values
    - No error handling or cleanup
    - Unclear test intent

    Suggest specific improvements.
```

## Advanced Settings

### Concurrency Control

Control how many test cases and chunks are processed in parallel:

```yaml
# Max concurrent API requests = test_case × chunk
concurrency:
  test_case: 2  # Parallel test case analyses
  chunk: 1      # Parallel chunks per large test (for chunked tests only)
```

**Maximum concurrent API requests = test_case × chunk**

For ChatCopilot, this automatically sets the session pool size.

| test_case | chunk | Max Concurrent |
|-----------|-------|----------------|
| 1 | 1 | 1 (safest) |
| 2 | 1 | 2 |
| 2 | 2 | 4 |
| 3 | 3 | 9 |

CLI override:
```bash
result-companion analyze -o output.xml --test-concurrency 2 --chunk-concurrency 1
```

**Note**: Useful for cloud APIs to avoid rate limits. Local models (Ollama) don't benefit from concurrency.

### Token Limits by Model

| Provider | Model | Input Tokens | Output Tokens | Config Setting |
|----------|-------|--------------|---------------|----------------|
| Copilot | gpt-4.1 | 64K | 16K | `max_content_tokens: 60000` |
| Copilot | claude-haiku-4.5 | 128K | 16K | `max_content_tokens: 120000` |
| Copilot | gpt-5-mini | 128K | 64K | `max_content_tokens: 120000` |
| Ollama | phi-3-mini | 4K | - | `max_content_tokens: 4000` |
| Ollama | deepseek-r1:1.5b | 8K | - | `max_content_tokens: 8000` |
| Ollama | mistral:7b | 8K | - | `max_content_tokens: 8000` |
| OpenAI | gpt-4o | 128K | 16K | `max_content_tokens: 120000` |
| Azure | gpt-4 | 8K | - | `max_content_tokens: 7000` |
| Google | gemini-2.0-flash | 1,048K | 65K | `max_content_tokens: 1000000` |

**Note**: Set input slightly below actual limit to account for prompt overhead. Gemini 2.0 Flash is ideal for large test suites.

**Local models (Ollama)**: Models run faster with GPU/NPU acceleration (Apple Silicon, NVIDIA CUDA, AMD ROCm). CPU-only is slower but works.

### Chunking for Large Tests

When tests exceed token limits, customize how chunks are analyzed:

```yaml
llm_config:
  chunking:
    chunk_analysis_prompt: |
      Extract key facts from this test chunk:
      - Actions taken and outcomes
      - Errors, exceptions, tracebacks
      - Suspicious or failing keywords
      {text}

    final_synthesis_prompt: |
      Combine all chunk summaries into final analysis.

      **Flow**: [bullet points]
      **Root Cause**: [specific cause]
      **Fixes**: [actionable steps]

      {summary}
```

## Complete Example

```bash
# Create custom config
cat > my_config.yaml << 'EOF'
llm_factory:
  model: "openai/gpt-4o"
  api_key: "${OPENAI_API_KEY}"

llm_config:
  question_prompt: |
    Find the bug, explain why it happened, and suggest fixes.
    Be specific about keyword names and line numbers when possible.
EOF

# Set credentials
export OPENAI_API_KEY="sk-..."

# Run analysis
result-companion analyze -o output.xml -c my_config.yaml

# View results
open rc_log.html
```

## Environment Variables Reference

```bash
# GitHub Copilot - No env vars needed! Uses CLI authentication.

# OpenAI
export OPENAI_API_KEY="sk-..."

# Azure OpenAI
export AZURE_DEPLOYMENT_NAME="gpt-4"
export AZURE_API_BASE="https://resource.openai.azure.com/"
export AZURE_API_KEY="..."

# Google
export GOOGLE_API_KEY="..."

# Anthropic
export ANTHROPIC_API_KEY="..."

# AWS Bedrock
export AWS_BEDROCK_MODEL_ID="anthropic.claude-v2"
export AWS_REGION="us-west-2"
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
```

## Additional Resources

LiteLLM documentation for supported providers:

- [LiteLLM Providers Overview](https://docs.litellm.ai/docs/providers)
- [Ollama](https://docs.litellm.ai/docs/providers/ollama)
- [OpenAI](https://docs.litellm.ai/docs/providers/openai)
- [Azure OpenAI](https://docs.litellm.ai/docs/providers/azure/)
- [Google Gemini](https://docs.litellm.ai/docs/providers/gemini)
- [Anthropic](https://docs.litellm.ai/docs/providers/anthropic)
- [AWS Bedrock](https://docs.litellm.ai/docs/providers/bedrock)

---

**Default Configuration**: See [`default_config.yaml`](../result_companion/core/configs/default_config.yaml) for all available options.
