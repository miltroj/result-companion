# Using Different LLM Models with ***result-companion***

## Table of Contents
- [Using Different LLM Models with ***result-companion***](#using-different-llm-models-with-result-companion)
  - [Table of Contents](#table-of-contents)
  - [Configuration Overview](#configuration-overview)
  - [Configuration File: ***user\_config.yaml***](#configuration-file-user_configyaml)
    - [1. OllamaLLM with DeepSeek Model](#1-ollamallm-with-deepseek-model)
    - [2. AzureChatOpenAI Model](#2-azurechatopenai-model)
    - [3. BedrockLLM Model](#3-bedrockllm-model)
    - [4. ChatGoogleGenerativeAI Model](#4-chatgooglegenerativeai-model)
    - [5. ChatOpenAI with Custom Endpoint (Databricks, OpenAI-compatible APIs)](#5-chatopenai-with-custom-endpoint-databricks-openai-compatible-apis)
  - [Concurrency Configuration](#concurrency-configuration)
  - [Customizing Analysis Prompts](#customizing-analysis-prompts)
    - [Main Analysis Prompt](#main-analysis-prompt)
    - [Chunking Prompts](#chunking-prompts)
  - [Understanding Content Tokenization and Chunking](#understanding-content-tokenization-and-chunking)
    - [Setting Appropriate Token Limits](#setting-appropriate-token-limits)
  - [Environment Variables in Configuration Files](#environment-variables-in-configuration-files)
  - [Running the Application](#running-the-application)
  - [Additional Resources](#additional-resources)

This guide provides instructions on configuring and utilizing various Large Language Models (LLMs) such as ***OllamaLLM***, ***AzureChatOpenAI***, ***BedrockLLM***, and ***ChatGoogleGenerativeAI*** within your application.

## Configuration Overview

**Default Configuration**: [`result_companion/core/configs/default_config.yaml`](../result_companion/core/configs/default_config.yaml)
Built-in defaults for model, prompts, and settings. Works out-of-the-box.

**User Configuration**: `user_config.yaml` (you create this)
Your customizations that override defaults. Required when using different models or custom prompts.

**Usage**:
```sh
# Use defaults
result-companion -o output.xml -r report.html

# Override with your config
result-companion -o output.xml -r report.html -c user_config.yaml
```

## Configuration File: ***user_config.yaml***

Create `user_config.yaml` to customize model, prompts, or settings. Only specify what you want to override—unspecified values use defaults.

### 1. OllamaLLM with DeepSeek Model

```yaml
# user_config.yaml
version: 1.0

llm_factory:
  model_type: "OllamaLLM"
  parameters:
    model: "deepseek-r1:7b"
#    model: "llama3.2"

tokenizer:
  tokenizer: ollama_tokenizer
  max_content_tokens: 32000  # Context window for most Ollama models (adjust based on specific model)
```

**Note:** Ensure that the specified model (e.g., ***deepseek-r1:7b***) is available and properly configured in your environment. The `max_content_tokens` value should be adjusted based on your specific model's context window size (typically 8K-32K for most Ollama models).

### 2. AzureChatOpenAI Model

```yaml
# user_config.yaml
version: 1.0

llm_factory:
  model_type: "AzureChatOpenAI"
  parameters:
    deployment_name: "${AZURE_DEPLOYMENT_NAME}"
    api_version: "2023-05-15"
    api_base: "${AZURE_API_BASE}"
    api_key: "${AZURE_API_KEY}"

tokenizer:
  tokenizer: azure_openai_tokenizer
  max_content_tokens: 16000  # Context window for GPT-4 Turbo (8k for GPT-4, 4k for GPT-3.5)
```

**Note:** Set up the following environment variables with your Azure OpenAI credentials:
- `AZURE_DEPLOYMENT_NAME`: Your Azure OpenAI deployment name
- `AZURE_API_BASE`: Your Azure endpoint (e.g., "https://your-resource-name.openai.azure.com/")
- `AZURE_API_KEY`: Your Azure OpenAI API key

Adjust the `max_content_tokens` based on your deployed model: 4K for GPT-3.5-Turbo, 8K for GPT-4, 16K+ for GPT-4 Turbo. For more information, refer to the [AzureChatOpenAI documentation](https://python.langchain.com/docs/integrations/chat/azure_chat_openai/).

### 3. BedrockLLM Model

```yaml
# user_config.yaml
version: 1.0

llm_factory:
  model_type: "BedrockLLM"
  parameters:
    model_id: "${AWS_BEDROCK_MODEL_ID}"
    region_name: "${AWS_REGION}"
    aws_access_key_id: "${AWS_ACCESS_KEY_ID}"
    aws_secret_access_key: "${AWS_SECRET_ACCESS_KEY}"

tokenizer:
  tokenizer: bedrock_tokenizer
  max_content_tokens: 8000  # Context window varies by model (Claude: 100K+, Titan: 8K, Llama2: 4K)
```

**Note:** Set up the following environment variables with your AWS credentials:
- `AWS_BEDROCK_MODEL_ID`: Your AWS Bedrock model ID
- `AWS_REGION`: Your AWS region (e.g., "us-west-2")
- `AWS_ACCESS_KEY_ID`: Your AWS access key ID
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret access key

The `max_content_tokens` should match your specific Bedrock model (Claude models support up to 100K+ tokens, Titan up to 8K, and Llama2 up to 4K). Ensure that the credentials used have the required policies to access the Bedrock service. For more information, refer to the [BedrockLLM documentation](https://python.langchain.com/api_reference/aws/llms/langchain_aws.llms.bedrock.BedrockLLM.html).

### 4. ChatGoogleGenerativeAI Model

```yaml
# user_config.yaml
version: 1.0

llm_factory:
  model_type: "ChatGoogleGenerativeAI"
  parameters:
    model: "gemini-2.5-pro"
    temperature: 0
    google_api_key: "${GOOGLE_API_KEY}"  # Set your Google API key as an environment variable

tokenizer:
  tokenizer: google_tokenizer
  max_content_tokens: 32000  # Context window for Gemini Pro (1M for Gemini 1.5 Pro, 32K for standard)
```

**Note:** Set up the `GOOGLE_API_KEY` environment variable with your Google API key. To obtain a Google API key, visit the [Google AI Studio](https://makersuite.google.com/app/apikey). The `max_content_tokens` setting should be adjusted based on your Gemini model (standard Gemini Pro supports 32K tokens, while Gemini 1.5 Pro can support up to 1 million tokens). For more information, refer to the [ChatGoogleGenerativeAI documentation](https://python.langchain.com/docs/integrations/chat/google_generative_ai).

### 5. ChatOpenAI with Custom Endpoint (Databricks, OpenAI-compatible APIs)

```yaml
# user_config.yaml
version: 1.0

llm_factory:
  model_type: "ChatOpenAI"
  parameters:
    model: "${OPENAI_MODEL}"
    api_key: "${OPENAI_API_KEY}"
    base_url: "${OPENAI_BASE_URL}"

tokenizer:
  tokenizer: openai_tokenizer
  max_content_tokens: 16000  # Adjust based on your model
```

**Note:** This configuration works with any OpenAI-compatible API endpoint, including Databricks Model Serving. Set up the following environment variables:
- `OPENAI_MODEL`: Model name
- `OPENAI_API_KEY`: Your API token
- `OPENAI_BASE_URL`: Base URL for the API (e.g., "https://your-platform/serving-endpoints")

## Concurrency Configuration

Control how many API requests are made in parallel to avoid rate limits (HTTP 429 errors).

```yaml
# user_config.yaml
concurrency:
  test_case: 2  # Test cases processed in parallel
  chunk: 2      # Chunks per test case in parallel
```

**Maximum concurrent API requests = test_case × chunk**

| test_case | chunk | Max Concurrent Requests |
|-----------|-------|------------------------|
| 1 | 1 | 1 (sequential, safest) |
| 2 | 1 | 2 |
| 2 | 2 | 4 |
| 3 | 3 | 9 |

**CLI Override:** You can override config values via command line:

```sh
result-companion analyze -o output.xml -r report.html -c config.yaml \
  --test-concurrency 2 \
  --chunk-concurrency 1
```

**Recommendation:** Start with low values (1, 1) and increase based on your API provider's rate limits.

**Local models (Ollama):** Concurrency is supported but not recommended—local LLMs are CPU/GPU bound, and parallel requests typically don't improve performance.

## Customizing Analysis Prompts

Result Companion's analysis behavior is fully customizable through prompts. This enables use cases beyond error analysis: security audits, performance issues, test quality assessment, or custom workflows.

**Defaults**: See [`default_config.yaml`](../result_companion/core/configs/default_config.yaml) for built-in prompts.
**Override**: Create `user_config.yaml` with your custom prompts.

### Main Analysis Prompt

Override `question_prompt` in your `user_config.yaml` to change analysis behavior:

```yaml
# user_config.yaml
llm_config:
  question_prompt: |
    Your custom analysis instructions here.
    Define what to look for, how to structure output, etc.
```

**Use Cases:**
- **Error Analysis** (default): Identify test failures and root causes
- **Security Audit**: Look for security vulnerabilities in test execution
- **Performance Review**: Analyze timing and resource usage patterns
- **Code Quality**: Assess test structure and maintainability

**Example - Security Focus:**
```yaml
# user_config.yaml
llm_config:
  question_prompt: |
    Analyze this Robot Framework test for security concerns:
    - Hardcoded credentials or secrets
    - Insecure API calls or configurations
    - Data exposure risks

    Output: **Security Findings** | **Severity** | **Recommendations**
```

### Chunking Prompts

For large tests exceeding `max_content_tokens`, override chunking prompts in `user_config.yaml`:

```yaml
# user_config.yaml
llm_config:
  chunking:
    chunk_analysis_prompt: |
      Extract key information from this test chunk.
      {text}

    final_synthesis_prompt: |
      Synthesize findings from all chunks.
      {summary}
```

## Understanding Content Tokenization and Chunking

The `max_content_tokens` parameter in the configuration file is crucial for handling large test results. This parameter determines:

1. **Maximum Context Size**: The maximum number of tokens that can be processed in a single API call to the LLM.
2. **Chunking Behavior**: When test results exceed this limit, the content is automatically split into smaller chunks and processed separately.
3. **Result Aggregation**: The analysis results from individual chunks are then aggregated to provide a comprehensive view.

### Setting Appropriate Token Limits

Each model has different context window limitations:

| Model Type | Recommended Setting | Notes |
|------------|---------------------|-------|
| OllamaLLM | 8,000-32,000 | Depends on specific model (Llama2: ~4K, Llama3/DeepSeek: ~8K-32K) |
| AzureChatOpenAI | 4,000-16,000 | GPT-3.5: 4K, GPT-4: 8K, GPT-4 Turbo: 16K or 128K |
| BedrockLLM | 4,000-100,000 | Varies widely (Titan: 8K, Llama2: 4K, Claude: 100K+) |
| ChatGoogleGenerativeAI | 32,000-1,000,000 | Gemini Pro: 32K, Gemini 1.5 Pro: Up to 1M |
| ChatOpenAI | 4,000-128,000 | Depends on endpoint model (GPT-4: 8K, GPT-4 Turbo: 128K) |

Setting this parameter too low will cause unnecessary chunking which might reduce analysis quality, while setting it too high might result in token limit errors from the API.

For optimal results, set this parameter close to (but slightly below) your model's actual context window size. For example, if using GPT-4 with an 8K context window, a setting of 7,000 would be appropriate to allow for prompt overhead.

## Environment Variables in Configuration Files

The configuration files support environment variable substitution using the `${VARIABLE_NAME}` syntax. This allows you to keep sensitive information like API keys out of your configuration files, making your setup more secure.

For example, instead of hardcoding your API keys:

```yaml
api_key: "your-actual-api-key-here"
```

You can reference an environment variable:

```yaml
api_key: "${API_KEY_ENV_VAR}"
```

Before running the application, make sure to set all required environment variables:

```sh
# For Google API
export GOOGLE_API_KEY="your-google-api-key"

# For Azure OpenAI
export AZURE_DEPLOYMENT_NAME="your-deployment-name"
export AZURE_API_BASE="https://your-resource-name.openai.azure.com/"
export AZURE_API_KEY="your-azure-api-key"

# For AWS Bedrock
export AWS_BEDROCK_MODEL_ID="your-model-id"
export AWS_REGION="your-aws-region"
export AWS_ACCESS_KEY_ID="your-aws-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-aws-secret-access-key"
```

## Running the Application

**With default configuration** (built-in OllamaLLM):
```sh
result-companion -o output.xml -r log_with_llm_results.html
```

**With custom configuration** (your `user_config.yaml`):
```sh
result-companion -o output.xml -r log_with_llm_results.html -c user_config.yaml
```

The `-c` flag applies your customizations while keeping unspecified settings from defaults.

## Additional Resources

- **OllamaLLM Documentation:** For detailed information on configuring and using ***OllamaLLM***, refer to the [LangChain OllamaLLM documentation](https://python.langchain.com/docs/integrations/llms/ollama/).

- **AzureChatOpenAI Documentation:** For detailed information on configuring and using ***AzureChatOpenAI***, refer to the [LangChain AzureChatOpenAI documentation](https://python.langchain.com/docs/integrations/chat/azure_chat_openai/).

- **BedrockLLM Documentation:** For detailed information on configuring and using ***BedrockLLM***, refer to the [LangChain BedrockLLM documentation](https://python.langchain.com/api_reference/aws/llms/langchain_aws.llms.bedrock.BedrockLLM.html).

- **ChatGoogleGenerativeAI Documentation:** For detailed information on configuring and using ***ChatGoogleGenerativeAI***, refer to the [LangChain ChatGoogleGenerativeAI documentation](https://python.langchain.com/docs/integrations/chat/google_generative_ai).

- **ChatOpenAI Documentation:** For detailed information on configuring and using ***ChatOpenAI*** with custom endpoints, refer to the [LangChain ChatOpenAI documentation](https://python.langchain.com/docs/integrations/chat/openai/).

By following the configurations and instructions provided above, you can effectively integrate and utilize different LLM models within your application.
