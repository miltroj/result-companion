from result_companion.core.chunking.utils import (
    calculate_chunk_size,
    calculate_overall_chunk_size,
)
from result_companion.core.parsers.config import TokenizerModel


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
    assert chunk.chunk_size == 333.3333333333333


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
    assert chunk.chunk_size == 2.75
