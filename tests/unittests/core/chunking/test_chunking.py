import asyncio
from unittest.mock import patch

import pytest
from langchain.prompts import PromptTemplate
from langchain_community.llms.fake import FakeListLLM
from langchain_core.output_parsers import StrOutputParser

from result_companion.core.chunking.chunking import (
    accumulate_llm_results_for_summarizaton_chain,
    build_sumarization_chain,
    condense_summaries_if_needed,
    split_text_into_chunks_using_text_splitter,
    summarize_test_case,
)
from result_companion.core.chunking.utils import Chunking


def test_recursive_text_splitting():
    text = "This is a test text"
    chunk_size = 5
    overlap = 2

    chunks = split_text_into_chunks_using_text_splitter(text, chunk_size, overlap)
    assert chunks == ["This", "is a", "test", "text"]


@pytest.mark.asyncio
async def test_building_fake_chain() -> None:
    llm = FakeListLLM(
        responses=["response1", "response2"],
        model="fake",
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
        top_k=50,
        stop_sequence=None,
    )

    prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "You are analyzing chunk of system test logs. Identify potential errors and failure root causes "
            "from the following chunk of json text, do not propose solutions focus on facts:\n\n{text}"
        ),
    )

    chain = build_sumarization_chain(prompt, llm)

    response1 = await chain.ainvoke({"text": "This is a test"})
    assert response1 == "response1"
    response2 = await chain.ainvoke({"text": "This is a test"})
    assert response2 == "response2"


@pytest.mark.asyncio
@patch("result_companion.core.chunking.chunking.build_sumarization_chain")
async def test_executing_summarization_chain(mock_building_chain) -> None:
    text_prompt = PromptTemplate(
        input_variables=["text"],
        template=("Summarize the following text:\n\n{text}"),
    )

    summary_prompt = PromptTemplate(
        input_variables=["summary"],
        template=("Write test summary:\n\n{summary}"),
    )

    fake_llm = FakeListLLM(
        responses=["chunk 1", "chunk 2", "final summary"],
        model="fake",
    )

    fake_text_chain = text_prompt | fake_llm | StrOutputParser()
    fake_summary_chain = summary_prompt | fake_llm | StrOutputParser()

    mock_building_chain.side_effect = [fake_text_chain, fake_summary_chain]

    test_case = {"name": "test_case_name", "content": "This is a test"}

    chunk_prompt = "Analyze: {text}"
    final_prompt = "Synthesize: {summary}"

    chunks = ["text chunk1", "text chunk2"]

    condense_prompt = "Condense: {text}"
    result = await summarize_test_case(
        test_case, chunks, fake_llm, chunk_prompt, final_prompt, condense_prompt
    )
    assert result == ("final summary", "test_case_name", ["text chunk1", "text chunk2"])


@pytest.mark.asyncio
async def test_splitting_into_chunks_and_accumulatiing_summary_results() -> None:
    test_case_text = "a" * 10
    test_case = {"name": "chunking_test_case_name", "content": test_case_text}
    chunk_prompt = "Analyze: {text}"
    final_prompt = "Synthesize: {summary}"

    fake_llm = FakeListLLM(
        responses=["chunk 1", "chunk 2", "final summary"],
        model="fake",
    )

    chunking_strategy = Chunking(
        chunk_size=5,
        number_of_chunks=2,
        raw_text_len=len(test_case_text),
        tokens_from_raw_text=len(test_case_text) // 4,
        tokenized_chunks=2,
    )
    # chunk size and test_case_text len determines the number of api calls
    test_case_len = len(str(test_case))
    assert test_case_len == 60

    condense_prompt = "Condense: {text}"
    result = await accumulate_llm_results_for_summarizaton_chain(
        test_case,
        chunk_prompt,
        final_prompt,
        condense_prompt,
        chunking_strategy,
        fake_llm,
    )
    expected_chunks = [
        "{'nam",
        "e':",
        "'chu",
        "nking",
        "_test",
        "_case",
        "_name",
        "',",
        "'con",
        "tent'",
        ":",
        "'aaa",
        "aaaaa",
        "aa'}",
    ]

    assert result == ("final summary", "chunking_test_case_name", expected_chunks)
    # add assert for the number of api calls


@pytest.mark.asyncio
async def test_summarize_test_case_respects_chunk_concurrency():
    """Test that chunk_concurrency limits parallel processing."""
    max_concurrent = 0
    current_concurrent = 0
    lock = asyncio.Lock()

    async def tracking_ainvoke(input_dict):
        nonlocal max_concurrent, current_concurrent
        async with lock:
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
        await asyncio.sleep(0.01)  # Simulate work
        async with lock:
            current_concurrent -= 1
        return "chunk result"

    test_case = {"name": "concurrency_test", "content": "test"}
    chunks = ["chunk1", "chunk2", "chunk3", "chunk4"]

    with patch(
        "result_companion.core.chunking.chunking.build_sumarization_chain"
    ) as mock_chain:
        mock_chain.return_value.ainvoke = tracking_ainvoke

        await summarize_test_case(
            test_case,
            chunks,
            mock_chain,
            "Analyze: {text}",
            "Synthesize: {summary}",
            "Condense: {text}",
            chunk_concurrency=2,
        )

    assert max_concurrent <= 2


@pytest.mark.asyncio
async def test_summarize_uses_config_prompts():
    """Test that summarize_test_case uses prompts from config."""
    test_case = {"name": "config_test", "content": "test"}
    chunks = ["chunk1", "chunk2"]

    chunk_prompt = "Custom chunk analysis: {text}"
    final_prompt = "Custom final synthesis: {summary}"

    fake_llm = FakeListLLM(
        responses=["chunk1_result", "chunk2_result", "final_result"],
        model="fake",
    )

    condense_prompt = "Condense: {text}"
    result, name, returned_chunks = await summarize_test_case(
        test_case, chunks, fake_llm, chunk_prompt, final_prompt, condense_prompt
    )

    assert result == "final_result"
    assert name == "config_test"
    assert returned_chunks == ["chunk1", "chunk2"]


@pytest.mark.asyncio
async def test_condense_summaries_returns_aggregated_when_fits_in_limit():
    """Test that summaries are returned as-is when they fit within token limit."""
    summaries = ["Summary 1", "Summary 2"]
    final_prompt = "Final: {summary}"
    condense_prompt = "Condense: {text}"

    def fake_tokenizer(text: str) -> int:
        return len(text) // 4

    fake_llm = FakeListLLM(responses=["condensed"], model="fake")

    result = await condense_summaries_if_needed(
        summaries,
        final_prompt,
        condense_prompt,
        fake_tokenizer,
        max_tokens=1000,
        llm=fake_llm,
    )

    assert "Summary 1" in result
    assert "Summary 2" in result


@pytest.mark.asyncio
async def test_condense_summaries_returns_early_at_max_depth():
    """Guard: depth=3 returns immediately without LLM condensation."""
    # Large summaries that would normally trigger condensation
    summaries = ["X" * 200 for _ in range(4)]
    fake_llm = FakeListLLM(responses=["should_not_appear"], model="fake")

    # High token limit ensures only depth check triggers early return
    result = await condense_summaries_if_needed(
        summaries,
        "P" * 10,
        "C: {text}",
        lambda t: len(t),
        max_tokens=1000,
        llm=fake_llm,
        depth=3,
    )

    # LLM was NOT called - original content preserved, no "should_not_appear"
    assert "should_not_appear" not in result
    assert "Chunk 1/4" in result


@pytest.mark.asyncio
async def test_condense_summaries_returns_early_when_available_tokens_non_positive():
    """Guard: available_tokens <= 0 returns immediately without LLM call."""
    summaries = ["X" * 200 for _ in range(4)]
    fake_llm = FakeListLLM(responses=["should_not_appear"], model="fake")

    # Tokenizer: 1 char = 1 token. Prompt=400, max=50 -> available = 50-400-500 = -850
    result = await condense_summaries_if_needed(
        summaries, "X" * 400, "C: {text}", lambda t: len(t), max_tokens=50, llm=fake_llm
    )

    assert "should_not_appear" not in result
    assert "Chunk 1/4" in result


@pytest.mark.asyncio
async def test_condense_summaries_returns_early_for_single_summary():
    """Guard: single summary returns immediately without processing."""
    summaries = ["Only one summary here"]
    fake_llm = FakeListLLM(responses=["llm should not be called"], model="fake")

    result = await condense_summaries_if_needed(
        summaries, "F", "C: {text}", lambda t: len(t), max_tokens=5, llm=fake_llm
    )

    assert "Only one summary here" in result
    assert "Chunk 1/1" in result
    assert "llm should not be called" not in result


@pytest.mark.asyncio
async def test_condense_summaries_returns_early_for_empty_list():
    """Guard: empty list returns empty aggregated string."""
    fake_llm = FakeListLLM(responses=["should not be called"], model="fake")

    result = await condense_summaries_if_needed(
        [], "F", "C: {text}", lambda t: len(t), max_tokens=100, llm=fake_llm
    )

    assert result == ""
    assert "should not be called" not in result


@pytest.mark.asyncio
async def test_condense_summaries_returns_when_grouping_would_not_reduce():
    """Guard: if all summaries fit in one group, returns without condensing."""
    summaries = ["A", "B"]
    fake_llm = FakeListLLM(responses=["summarize should not be called"], model="fake")

    # available_tokens very high -> all summaries fit in one group
    result = await condense_summaries_if_needed(
        summaries, "F", "C: {text}", lambda t: len(t), max_tokens=10000, llm=fake_llm
    )

    assert "A" in result and "B" in result
    assert "summarize should not be called" not in result


@pytest.mark.asyncio
async def test_condense_summaries_calls_llm_to_condense_when_exceeding_limit():
    """Condensation path: groups summaries and calls LLM when limit exceeded."""
    # 4 long summaries that exceed the available token budget
    summaries = ["X" * 200 for _ in range(4)]
    fake_llm = FakeListLLM(
        responses=["condensed_1", "condensed_2", "final"], model="fake"
    )

    # 1 char = 1 token; prompt=10, buffer=500 -> available=490; aggregated ~800+ tokens
    result = await condense_summaries_if_needed(
        summaries,
        "P" * 10,
        "Condense: {text}",
        lambda t: len(t),
        max_tokens=1000,
        llm=fake_llm,
    )

    assert "condensed" in result or "Group" in result
    assert "condensed_1" in result or "condensed_2" in result
    assert "final" not in result  # groups = [[sum1, sum2], [sum3, sum4]]
    # 2 groups ["[Group 1] condensed_1", "[Group 2] condensed_2"]
