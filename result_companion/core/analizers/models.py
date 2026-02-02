"""Model type definitions.

This module previously contained LangChain model type definitions.
With the migration to LiteLLM, model types are no longer needed as
LiteLLM uses a unified interface with model strings.

LiteLLM model format: provider/model-name
Examples:
- ollama_chat/llama2
- openai/gpt-4o
- anthropic/claude-3-sonnet
- gemini/gemini-2.0-flash
- bedrock/anthropic.claude-v2
"""
