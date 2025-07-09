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
  max_content_tokens: 140000
```

**Note:** Ensure that the specified model (e.g., ***deepseek-r1:7b***) is available and properly configured in your environment.

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
  max_content_tokens: 4000
```

**Note:** Set up the following environment variables with your Azure OpenAI credentials:
- `AZURE_DEPLOYMENT_NAME`: Your Azure OpenAI deployment name
- `AZURE_API_BASE`: Your Azure endpoint (e.g., "https://your-resource-name.openai.azure.com/")
- `AZURE_API_KEY`: Your Azure OpenAI API key

For more information, refer to the [AzureChatOpenAI documentation](https://python.langchain.com/docs/integrations/chat/azure_chat_openai/).

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
  max_content_tokens: 4000
```

**Note:** Set up the following environment variables with your AWS credentials:
- `AWS_BEDROCK_MODEL_ID`: Your AWS Bedrock model ID
- `AWS_REGION`: Your AWS region (e.g., "us-west-2")
- `AWS_ACCESS_KEY_ID`: Your AWS access key ID
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret access key

Ensure that the credentials used have the required policies to access the Bedrock service. For more information, refer to the [BedrockLLM documentation](https://python.langchain.com/api_reference/aws/llms/langchain_aws.llms.bedrock.BedrockLLM.html).

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
  max_content_tokens: 140000
```

**Note:** Set up the `GOOGLE_API_KEY` environment variable with your Google API key. To obtain a Google API key, visit the [Google AI Studio](https://makersuite.google.com/app/apikey). For more information, refer to the [ChatGoogleGenerativeAI documentation](https://python.langchain.com/docs/integrations/chat/google_generative_ai).

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
