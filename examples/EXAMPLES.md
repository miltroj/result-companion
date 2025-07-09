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
    deployment_name: "your-deployment-name"
    api_version: "2023-05-15"
    api_base: "https://your-resource-name.openai.azure.com/"
    api_key: "your-azure-openai-api-key"

tokenizer:
  tokenizer: azure_openai_tokenizer
  max_content_tokens: 4000
```

**Note:** Replace the placeholders (***your-deployment-name***, ***your-resource-name***, and ***your-azure-openai-api-key***) with your actual Azure OpenAI deployment details. For more information, refer to the [AzureChatOpenAI documentation](https://python.langchain.com/docs/integrations/chat/azure_chat_openai/).

### 3. BedrockLLM Model

```yaml
# user_config.yaml
version: 1.0

llm_factory:
  model_type: "BedrockLLM"
  parameters:
    model_id: "your-model-id"
    region_name: "your-aws-region"
    aws_access_key_id: "your-aws-access-key-id"
    aws_secret_access_key: "your-aws-secret-access-key"

tokenizer:
  tokenizer: bedrock_tokenizer
  max_content_tokens: 4000
```

**Note:** Replace the placeholders (***your-model-id***, ***your-aws-region***, ***your-aws-access-key-id***, and ***your-aws-secret-access-key***) with your actual AWS credentials and Bedrock model details. Ensure that the credentials used have the required policies to access the Bedrock service. For more information, refer to the [BedrockLLM documentation](https://python.langchain.com/api_reference/aws/llms/langchain_aws.llms.bedrock.BedrockLLM.html).

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

**Note:** You need to set up the `GOOGLE_API_KEY` environment variable with your Google API key. To obtain a Google API key, visit the [Google AI Studio](https://makersuite.google.com/app/apikey). For more information, refer to the [ChatGoogleGenerativeAI documentation](https://python.langchain.com/docs/integrations/chat/google_generative_ai).

## Running the Application

After configuring the ***user_config.yaml*** file with the desired model parameters, run the application using the following command:

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
