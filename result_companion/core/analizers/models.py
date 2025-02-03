from typing import Callable, Tuple

from langchain_aws import BedrockLLM
from langchain_ollama.llms import OllamaLLM
from langchain_openai import AzureChatOpenAI

MODELS = Tuple[OllamaLLM | AzureChatOpenAI | BedrockLLM, Callable]
