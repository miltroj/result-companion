import pytest

from result_companion.analizers.common import run_llm_based_analysis_and_stream_results


@pytest.mark.asyncio
async def test_run_llm_based_analysis_and_stream_results():

    class TestAsyncGenerator:
        def __init__(self, responses):
            self.responses = responses

        async def astream(self, *args, **kwargs):
            for response in self.responses:
                yield response

    test_cases = [
        {"name": "Test Case 1", "context": "Context 1"},
        {"name": "Test Case 2", "context": "Context 2"},
    ]

    question_from_config_file = "What is the result?"

    mck_chain = TestAsyncGenerator(["chunk_1 ", "chunk_2"])

    llm_results = await run_llm_based_analysis_and_stream_results(
        test_cases, question_from_config_file, mck_chain
    )

    expected_results = {
        "Test Case 1": "chunk_1 chunk_2",
        "Test Case 2": "chunk_1 chunk_2",
    }
    assert llm_results == expected_results
