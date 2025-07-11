import asyncio
from typing import Tuple

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable
from tqdm import tqdm

from result_companion.core.analizers.models import MODELS
from result_companion.core.chunking.utils import Chunking

# Import progress-related utilities
from result_companion.core.utils.progress import setup_progress_logging


def build_sumarization_chain(
    prompt: PromptTemplate, model: MODELS
) -> RunnableSerializable:
    return prompt | model | StrOutputParser()


def split_text_into_chunks_using_text_splitter(
    text: str, chunk_size: int, overlap: int
) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text)


async def accumulate_llm_results_for_summarizaton_chain(
    test_case: dict,
    question_from_config_file: str,
    chain: RunnableSerializable,
    chunking_strategy: Chunking,
    llm: MODELS,
) -> Tuple[str, str, list]:
    chunks = split_text_into_chunks_using_text_splitter(
        str(test_case), chunking_strategy.chunk_size, chunking_strategy.chunk_size // 10
    )
    return await summarize_test_case(
        test_case, chunks, llm, question_from_config_file, chain
    )


async def process_chunk(chunk: str, summarization_chain: LLMChain) -> str:
    # Use tqdm's write function to output logs above the progress bar
    tqdm.write(f"Processing chunk of length {len(chunk)}")
    return await summarization_chain.ainvoke({"text": chunk})


async def summarize_test_case(test_case, chunks, llm, question_prompt, chain):
    # Configure progress-friendly logging
    progress_logger = setup_progress_logging()

    progress_logger.info(
        f"### For test case {test_case['name']}, {len(chunks)=}",
    )
    # TODO: move to default_config.yaml
    summarization_prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "You are analyzing chunk of system test logs. Identify potential errors and failure root causes "
            "from the following chunk of json text, do not propose solutions focus on facts:\n\n{text}"
        ),
    )

    summarization_chain = build_sumarization_chain(summarization_prompt, llm)

    # Process chunks with progress tracking
    # Setup progress bar for chunk processing
    chunk_tasks = []
    for chunk in chunks:
        chunk_tasks.append(process_chunk(chunk, summarization_chain))

    if len(chunks) > 1:
        # Only show progress for multiple chunks
        with tqdm(
            total=len(chunks),
            desc=f"Processing chunks for {test_case['name']}",
            leave=False,
            position=0,  # Keep at bottom of screen
            dynamic_ncols=True,  # Adjust width based on terminal
            miniters=1,  # Update on each iteration
        ) as pbar:
            # Create a list to hold results
            summaries = []
            pending = [asyncio.create_task(task) for task in chunk_tasks]

            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    summaries.append(task.result())
                    pbar.update(1)
    else:
        # For single chunk just process normally
        summaries = await asyncio.gather(*chunk_tasks)

    aggregated_summary = "\n".join(summaries)

    final_prompt = PromptTemplate(
        input_variables=["summary"],
        template=(
            "Based on the aggregated summary below, provide a detailed analysis of the system test logs. "
            "Highlight spotted problems in specific keywords, root causes, and recommendations for resolving them:\n\n{summary}"
        ),
    )

    final_analysis_chain = build_sumarization_chain(final_prompt, llm)

    final_result = await final_analysis_chain.ainvoke({"summary": aggregated_summary})
    return final_result, test_case["name"], chunks
