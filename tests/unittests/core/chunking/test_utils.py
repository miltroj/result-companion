from unittest.mock import patch

from result_companion.core.chunking.utils import (
    anthropic_tokenizer,
    azure_openai_tokenizer,
    calculate_chunk_size,
    calculate_overall_chunk_size,
    google_tokenizer,
    tokenizer_mappings,
)
from result_companion.core.parsers.config import TokenizerModel, TokenizerTypes


def test_calculating_overall_chunk_size() -> None:
    chunk = calculate_overall_chunk_size(
        actual_tokens_from_text=20, max_tokens_acceptable=10, raw_text="a" * 100
    )
    assert chunk.chunk_size == 50
    assert chunk.number_of_chunks == 2
    assert chunk.raw_text_len == 100


def test_calculating_overall_chunk_size_when_there_is_no_need_for_chunking() -> None:
    chunk = calculate_overall_chunk_size(
        actual_tokens_from_text=2, max_tokens_acceptable=100, raw_text="a" * 10
    )
    assert chunk.chunk_size == 0
    assert chunk.number_of_chunks == 0
    assert chunk.raw_text_len == 10


def test_calculating_final_chunk_size() -> None:
    test_case = "t" * 999
    system_prompt = "1"
    tokenizer = TokenizerModel(tokenizer="ollama_tokenizer", max_content_tokens=100)

    chunk = calculate_chunk_size(test_case, system_prompt, tokenizer)

    assert chunk.raw_text_len == 1000
    assert chunk.tokens_from_raw_text == 1000 // 4
    assert chunk.tokenized_chunks == 3
    assert chunk.chunk_size == 333


def test_chunk_correctly_even_distributed_tokens() -> None:
    one_thousand_characters = "1" * 1000
    chunk = calculate_overall_chunk_size(
        actual_tokens_from_text=1000,
        max_tokens_acceptable=500,
        raw_text=one_thousand_characters,
    )
    assert chunk.chunk_size == 500
    assert chunk.number_of_chunks == 2
    assert chunk.raw_text_len == 1000
    assert chunk.tokens_from_raw_text == 1000


def test_chunking_not_even_distribution() -> None:
    raw_text = "1" * 11
    chunk = calculate_overall_chunk_size(
        actual_tokens_from_text=7, max_tokens_acceptable=2, raw_text=raw_text
    )

    assert chunk.raw_text_len == 11
    assert chunk.tokens_from_raw_text == 7
    assert chunk.tokenized_chunks == 4
    assert chunk.chunk_size == 2


@patch("tiktoken.get_encoding")
def test_google_tokenizer(mock_get_encoding):
    """Test the google_tokenizer function with mocking."""
    # Test with mocked tiktoken encoding
    mock_encode = mock_get_encoding.return_value.encode
    mock_encode.return_value = [1, 2, 3, 4, 5]  # Simulating 5 tokens

    result = google_tokenizer("sample text for testing")

    assert result == 5
    mock_get_encoding.assert_called_once_with("cl100k_base")
    mock_encode.assert_called_once_with("sample text for testing")

    # Test the fallback mechanism when tiktoken raises an exception
    with patch("tiktoken.get_encoding", side_effect=Exception("Test exception")):
        result = google_tokenizer("test string")  # 11 characters
        assert result == 11 // 4  # Should be 2 tokens using the fallback method


@patch("tiktoken.encoding_for_model")
def test_azure_openai_tokenizer(mock_encoding_for_model):
    """Test the azure_openai_tokenizer function using mocking."""
    # Set up the mock
    mock_encode = mock_encoding_for_model.return_value.encode
    mock_encode.return_value = [100, 200, 300]  # Simulating 3 tokens

    # Test with a sample text
    result = azure_openai_tokenizer("hello world")

    # Verify the results
    assert result == 3
    mock_encoding_for_model.assert_called_once_with("gpt-3.5-turbo")
    mock_encode.assert_called_once_with("hello world")


@patch("tiktoken.get_encoding")
def test_anthropic_tokenizer(mock_get_encoding):
    """Test the anthropic_tokenizer function with mocking."""
    mock_encode = mock_get_encoding.return_value.encode
    mock_encode.return_value = [1, 2, 3, 4]

    result = anthropic_tokenizer("test text")

    assert result == 4
    mock_get_encoding.assert_called_once_with("cl100k_base")
    mock_encode.assert_called_once_with("test text")

    with patch("tiktoken.get_encoding", side_effect=Exception("Test exception")):
        result = anthropic_tokenizer("test")
        assert result == 1


def test_all_tokenizer_types_have_mapping():
    """Ensure all TokenizerTypes have a corresponding tokenizer function."""
    for tokenizer_type in TokenizerTypes:
        assert (
            tokenizer_type.value in tokenizer_mappings
        ), f"Missing tokenizer mapping for {tokenizer_type}"
