# Examples & Configuration

Quick-start configurations for different LLM providers and use cases.

## Table of Contents
- [Examples \& Configuration](#examples--configuration)
  - [Table of Contents](#table-of-contents)
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

## Quick Configs

### Local (Ollama) - Default

No config needed! Just run:
```bash
result-companion -o output.xml
```

The default uses Ollama with `deepseek-r1:1.5b` model.

**Other MacBook-friendly models:**
- `phi-3-mini` (2.3GB, 8GB RAM) - Fast and efficient
- `mistral:7b` (4.1GB, 16GB RAM) - Industry standard, excellent quality

**Hardware note**: GPU/NPU significantly improves inference speed. Apple Silicon Macs use unified memory (RAM serves both CPU and GPU).

### OpenAI

```yaml
# openai_config.yaml
llm_factory:
  model_type: "ChatOpenAI"
  parameters:
    model: "gpt-5-nano"
    api_key: "${OPENAI_API_KEY}"

tokenizer:
  tokenizer: openai_tokenizer
  max_content_tokens: 390000
```

Run with:
```bash
export OPENAI_API_KEY="sk-..."
result-companion -o output.xml -c openai_config.yaml
```

### Azure OpenAI

```yaml
# azure_config.yaml
llm_factory:
  model_type: "AzureChatOpenAI"
  parameters:
    deployment_name: "${AZURE_DEPLOYMENT_NAME}"
    api_key: "${AZURE_API_KEY}"
    api_base: "${AZURE_API_BASE}"
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
result-companion -o output.xml -c azure_config.yaml
```

### Google Gemini

```yaml
# gemini_config.yaml
llm_factory:
  model_type: "ChatGoogleGenerativeAI"
  parameters:
    model: "gemini-2.5-flash"
    google_api_key: "${GOOGLE_API_KEY}"
    temperature: 0

tokenizer:
  tokenizer: google_tokenizer
  max_content_tokens: 1000000
```

Run with:
```bash
export GOOGLE_API_KEY="..."
result-companion -o output.xml -c gemini_config.yaml
```

### AWS Bedrock

```yaml
# bedrock_config.yaml
llm_factory:
  model_type: "BedrockLLM"
  parameters:
    model_id: "${AWS_BEDROCK_MODEL_ID}"
    region_name: "${AWS_REGION}"
    aws_access_key_id: "${AWS_ACCESS_KEY_ID}"
    aws_secret_access_key: "${AWS_SECRET_ACCESS_KEY}"

tokenizer:
  tokenizer: bedrock_tokenizer
  max_content_tokens: 100000
```

### Custom OpenAI-Compatible Endpoint

For Databricks, self-hosted, or other OpenAI-compatible APIs:

```yaml
# custom_config.yaml
llm_factory:
  model_type: "ChatOpenAI"
  parameters:
    model: "${OPENAI_MODEL}"
    api_key: "${OPENAI_API_KEY}"
    base_url: "${OPENAI_BASE_URL}"

tokenizer:
  tokenizer: openai_tokenizer
  max_content_tokens: 16000
```

### Anthropic with Claude Models

```yaml
# user_config.yaml
version: 1.0

llm_factory:
  model_type: "ChatAnthropic"
  parameters:
    model: "claude-haiku-4-5"
    api_key: "${ANTHROPIC_API_KEY}"
    temperature: 0

tokenizer:
  tokenizer: anthropic_tokenizer
  max_content_tokens: 200000  # Claude 4.5 supports 200K context window
```

**Note:** Set up the `ANTHROPIC_API_KEY` environment variable. Anthropic offers multiple Claude 4.5 models:
- `claude-haiku-4-5`: Lightweight, 3x cheaper, 2x faster (recommended for most cases)
- `claude-sonnet-4-5`: Balanced performance and cost
- `claude-opus-4-5`: Best reasoning capabilities

## Test Filtering

Focus analysis on specific tests using Robot Framework tag patterns.

### CLI Examples

```bash
# Analyze only smoke tests (failures only)
result-companion -o output.xml --include "smoke*"

# Analyze critical tests including passes
result-companion -o output.xml --include "critical*" -i

# Exclude WIP and known bugs
result-companion -o output.xml --exclude "wip,bug-*"

# Combine filters
result-companion -o output.xml --include "api,smoke" --exclude "flaky"
```

### Config File

```yaml
# tag_filtering_config.yaml
test_filter:
  include_tags: ["smoke*", "critical"]  # Wildcards supported
  exclude_tags: ["wip", "bug-*"]        # Exclude takes precedence
  include_passing: false                # false = failures only
```

Run with:
```bash
result-companion -o output.xml -c tag_filtering_config.yaml
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

### Find Performance Issues

Control how many API requests are made in parallel to avoid rate limits (HTTP 429 errors).

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
concurrency:
  test_case: 2  # Process 2 tests in parallel
  chunk: 1      # Sequential chunk processing
```

**Maximum concurrent requests = test_case × chunk**

| test_case | chunk | Max Concurrent |
|-----------|-------|----------------|
| 1 | 1 | 1 (safest) |
| 2 | 1 | 2 |
| 2 | 2 | 4 |
| 3 | 3 | 9 |

CLI override:
```bash
result-companion -o output.xml --test-concurrency 2 --chunk-concurrency 1
```

**Note**: Useful for cloud APIs to avoid rate limits. Local models (Ollama) don't benefit from concurrency.

### Token Limits by Model

| Provider | Model | Input Tokens | Output Tokens | Config Setting |
|----------|-------|--------------|---------------|----------------|
| Ollama | phi-3-mini | 4K | - | `max_content_tokens: 4000` |
| Ollama | deepseek-r1:1.5b | 8K | - | `max_content_tokens: 8000` |
| Ollama | mistral:7b | 8K | - | `max_content_tokens: 8000` |
| OpenAI | gpt-5-nano | 400K | 128K | `max_content_tokens: 390000` |
| Azure | gpt-4 | 8K | - | `max_content_tokens: 7000` |
| Google | gemini-2.5-flash | 1,048K | 65K | `max_content_tokens: 1000000` |

**Note**: Set input slightly below actual limit to account for prompt overhead. Gemini 2.5 Flash is ideal for large test suites.

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
  model_type: "ChatOpenAI"
  parameters:
    model: "gpt-5-nano"
    api_key: "${OPENAI_API_KEY}"

llm_config:
  question_prompt: |
    Find the bug, explain why it happened, and suggest fixes.
    Be specific about keyword names and line numbers when possible.
EOF

# Set credentials
export OPENAI_API_KEY="sk-..."

# Run analysis
result-companion -o output.xml -c my_config.yaml

# View results
open log_with_ai_analysis.html
```

## Environment Variables Reference

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Azure OpenAI
export AZURE_DEPLOYMENT_NAME="gpt-4"
export AZURE_API_BASE="https://resource.openai.azure.com/"
export AZURE_API_KEY="..."

# Google
export GOOGLE_API_KEY="..."

# AWS Bedrock
export AWS_BEDROCK_MODEL_ID="anthropic.claude-v2"
export AWS_REGION="us-west-2"
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
```

## Additional Resources

LangChain documentation for each model type:

- [OllamaLLM](https://python.langchain.com/docs/integrations/llms/ollama/)
- [AzureChatOpenAI](https://python.langchain.com/docs/integrations/chat/azure_chat_openai/)
- [BedrockLLM](https://python.langchain.com/api_reference/aws/llms/langchain_aws.llms.bedrock.BedrockLLM.html)
- [ChatGoogleGenerativeAI](https://python.langchain.com/docs/integrations/chat/google_generative_ai)
- [ChatOpenAI](https://python.langchain.com/docs/integrations/chat/openai/)

---

**Default Configuration**: See [`result_companion/core/configs/default_config.yaml`](../result_companion/core/configs/default_config.yaml) for all available options.
