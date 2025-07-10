import math
from dataclasses import dataclass

import tiktoken

from result_companion.core.parsers.config import TokenizerModel
from result_companion.core.utils.logging_config import logger


@dataclass
class Chunking:
    chunk_size: int
    number_of_chunks: int
    raw_text_len: int
    tokens_from_raw_text: int
    tokenized_chunks: int


def azure_openai_tokenizer(text: str) -> int:
    """Tokenizer for Azure OpenAI models using tiktoken."""
    # TODO: check if not starting something on import
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))


def ollama_tokenizer(text: str) -> int:
    """Placeholder tokenizer for Ollama (custom implementation required)."""
    return len(text) // 4


def bedrock_tokenizer(text: str) -> int:
    """Placeholder tokenizer for Bedrock LLM (custom implementation required)."""
    return len(text) // 5


def google_tokenizer(text: str) -> int:
    """Tokenizer for Google Generative AI models.

    Google's tokenization is approximately 4 characters per token on average,
    but we use the tiktoken cl100k_base encoding which is close to Google's tokenization.
    """
    try:
        # Use cl100k_base encoding which is similar to Google's tokenization
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Failed to use cl100k_base encoding for Google tokenizer: {e}")
        # Fallback to approximate tokenization (4 chars per token)
        return len(text) // 4


tokenizer_mappings = {
    "azure_openai_tokenizer": azure_openai_tokenizer,
    "ollama_tokenizer": ollama_tokenizer,
    "bedrock_tokenizer": bedrock_tokenizer,
    "google_tokenizer": google_tokenizer,
}


def calculate_overall_chunk_size(
    raw_text: str, actual_tokens_from_text: int, max_tokens_acceptable: int
) -> Chunking:
    raw_text_len = len(raw_text)
    N_tokenized_chunks = math.ceil(actual_tokens_from_text / max_tokens_acceptable)
    if max_tokens_acceptable > actual_tokens_from_text:
        return Chunking(
            chunk_size=0,
            number_of_chunks=0,
            raw_text_len=raw_text_len,
            tokens_from_raw_text=actual_tokens_from_text,
            tokenized_chunks=N_tokenized_chunks,
        )
    chunk_size = raw_text_len / N_tokenized_chunks
    logger.info(
        f"Chunk size: {chunk_size}, Number of chunks: {N_tokenized_chunks}, Raw text length: {raw_text_len}"
    )
    return Chunking(
        chunk_size=chunk_size,
        number_of_chunks=N_tokenized_chunks,
        raw_text_len=raw_text_len,
        tokens_from_raw_text=actual_tokens_from_text,
        tokenized_chunks=N_tokenized_chunks,
    )


def calculate_chunk_size(
    test_case: dict, system_prompt: str, tokenizer_from_config: TokenizerModel
) -> Chunking:
    LLM_fed_text = str(test_case) + system_prompt
    tokenizer = tokenizer_mappings[tokenizer_from_config.tokenizer]
    max_content_tokens = tokenizer_from_config.max_content_tokens
    text_to_tokens = tokenizer(LLM_fed_text)
    return calculate_overall_chunk_size(
        actual_tokens_from_text=text_to_tokens,
        max_tokens_acceptable=max_content_tokens,
        raw_text=LLM_fed_text,
    )
