# Examples & Configuration

Quick-start configurations for different LLM providers and use cases.

## Quick Configs

### Local (Ollama) - Default

No config needed! Just run:
```bash
result-companion -o output.xml
```

The default uses Ollama with `deepseek-r1:1.5b` model.

### OpenAI

```yaml
# openai_config.yaml
llm_factory:
  model_type: "ChatOpenAI"
  parameters:
    model: "gpt-4o-mini"
    api_key: "${OPENAI_API_KEY}"

tokenizer:
  tokenizer: openai_tokenizer
  max_content_tokens: 120000
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
    model: "gemini-2.0-flash"
    google_api_key: "${GOOGLE_API_KEY}"
    temperature: 0

tokenizer:
  tokenizer: google_tokenizer
  max_content_tokens: 900000
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

## Custom Analysis

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

### Rate Limiting

Control parallel requests to avoid API rate limits:

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

**Note**: Local models (Ollama) don't benefit from concurrency.

### Token Limits by Model

| Provider | Model | Max Tokens | Config Setting |
|----------|-------|------------|----------------|
| Ollama | deepseek-r1 | 32K | `max_content_tokens: 30000` |
| OpenAI | gpt-4o-mini | 128K | `max_content_tokens: 120000` |
| Azure | gpt-4 | 8K | `max_content_tokens: 7000` |
| Google | gemini-2.0-flash | 1M | `max_content_tokens: 900000` |

Set slightly below actual limit to account for prompt overhead.

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
    model: "gpt-4o-mini"
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

---

**Default Configuration**: See [`result_companion/core/configs/default_config.yaml`](../result_companion/core/configs/default_config.yaml) for all available options.
