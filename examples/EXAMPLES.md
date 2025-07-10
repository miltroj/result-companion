# Using Different LLM Models with ***result-companion***

This guide provides instructions on configuring and utilizing various Large Language Models (LLMs) such as ***OllamaLLM***, ***AzureChatOpenAI***, ***BedrockLLM***, and ***ChatGoogleGenerativeAI*** within your application. By specifying the appropriate parameters in the ***user_config.yaml*** file, you can invoke these models during runtime using the ***result-companion*** command.

## Configuration File: ***user_config.yaml***

The ***user_config.yaml*** file is used to define the LLM model and its parameters. Below are example configurations for each supported model.

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

After configuring the ***user_config.yaml*** file with the desired model parameters and setting any required environment variables, run the application using the following command:

```sh
result-companion -o output.xml -r log_with_llm_results.html -c user_config.yaml
```

This command will execute the application, utilizing the specified LLM model as defined in your configuration file.

## Additional Resources

- **OllamaLLM Documentation:** For detailed information on configuring and using ***OllamaLLM***, refer to the [LangChain OllamaLLM documentation](https://python.langchain.com/docs/integrations/llms/ollama/).

- **AzureChatOpenAI Documentation:** For detailed information on configuring and using ***AzureChatOpenAI***, refer to the [LangChain AzureChatOpenAI documentation](https://python.langchain.com/docs/integrations/chat/azure_chat_openai/).

- **BedrockLLM Documentation:** For detailed information on configuring and using ***BedrockLLM***, refer to the [LangChain BedrockLLM documentation](https://python.langchain.com/api_reference/aws/llms/langchain_aws.llms.bedrock.BedrockLLM.html).

- **ChatGoogleGenerativeAI Documentation:** For detailed information on configuring and using ***ChatGoogleGenerativeAI***, refer to the [LangChain ChatGoogleGenerativeAI documentation](https://python.langchain.com/docs/integrations/chat/google_generative_ai).

By following the configurations and instructions provided above, you can effectively integrate and utilize different LLM models within your application.
