import asyncio
from unittest.mock import patch

import pytest
from langchain.prompts import PromptTemplate
from langchain_community.llms.fake import FakeListLLM
from langchain_core.output_parsers import StrOutputParser

from result_companion.core.chunking.chunking import (
    accumulate_llm_results_for_summarizaton_chain,
    build_sumarization_chain,
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

    question_prompt = "question"

    chunks = ["text chunk1", "text chunk2"]

    result = await summarize_test_case(test_case, chunks, fake_llm, question_prompt)
    assert result == ("final summary", "test_case_name", ["text chunk1", "text chunk2"])


@pytest.mark.asyncio
async def test_splitting_into_chunks_and_accumulatiing_summary_results() -> None:
    test_case_text = "a" * 10
    test_case = {"name": "chunking_test_case_name", "content": test_case_text}
    question_prompt = "question"
    chain = "empty chain"

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

    result = await accumulate_llm_results_for_summarizaton_chain(
        test_case, question_prompt, chain, chunking_strategy, fake_llm
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
            test_case, chunks, mock_chain, "question", chunk_concurrency=2
        )

    assert max_concurrent <= 2
