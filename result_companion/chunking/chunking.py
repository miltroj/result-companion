import asyncio
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableSerializable
from result_companion.chunking.utils import Chunking
from result_companion.entrypoints.run_factory_splitting import MODELS
from langchain_core.output_parsers import StrOutputParser
from typing import Tuple


def build_sumarization_chain(prompt: PromptTemplate, model: MODELS) -> RunnableSerializable:
    return prompt | model | StrOutputParser()


def split_text_into_chunks_using_text_splitter(text: str, chunk_size: int, overlap: int) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap,length_function=len,
    is_separator_regex=False,)
    return splitter.split_text(text)


async def accumulate_llm_results_for_summarizaton_chain(test_case: dict, question_from_config_file: str, chain: RunnableSerializable, chunking_strategy: Chunking, llm: MODELS) -> Tuple[str, str, list]:
    chunks = split_text_into_chunks_using_text_splitter(str(test_case), chunking_strategy.chunk_size, chunking_strategy.chunk_size // 10)
    return await summarize_test_case(test_case, chunks, llm, question_from_config_file, chain)


async def process_chunk(chunk: str, summarization_chain: LLMChain) -> str:
    print(f"Processing chunk of length {len(chunk)}")
    return await summarization_chain.ainvoke({"text": chunk})


async def summarize_test_case(test_case, chunks, llm, question_prompt, chain):
    print(f"### For test case {test_case['name']}, {len(chunks)=}", )

    summarization_prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "You are analyzing chunk of system test logs. Identify potential errors and failure root causes "
            "from the following chunk of json text, do not propose solutions focus on facts:\n\n{text}"
        )
    )

    summarization_chain = build_sumarization_chain(summarization_prompt, llm)

    summaries = await asyncio.gather(*[process_chunk(chunk, summarization_chain) for chunk in chunks])
    aggregated_summary = "\n".join(summaries)

    final_prompt = PromptTemplate(
        input_variables=["summary"],
        template=(
            "Based on the aggregated summary below, provide a detailed analysis of the system test logs. "
            "Highlight spotted problems in specific keywords, root causes, and recommendations for resolving them:\n\n{summary}"
        )
    )

    final_analysis_chain = build_sumarization_chain(final_prompt, llm)

    final_result = await final_analysis_chain.ainvoke({"summary": aggregated_summary})
    return final_result, test_case["name"], chunks
