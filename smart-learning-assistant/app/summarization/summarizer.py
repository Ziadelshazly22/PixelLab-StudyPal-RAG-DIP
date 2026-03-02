"""
app/summarization/summarizer.py
--------------------------------
Builds a map-reduce chapter summarisation chain using LangChain.

The chain:
  1. MAP    – summarise each chunk individually (Gemini 2.0 Flash)
  2. REDUCE – combine chunk summaries into a coherent chapter summary
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

_MAP_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a concise academic summariser. "
            "Summarise the following DIP textbook excerpt in 3-5 bullet points, "
            "preserving key equations, definitions, and algorithm names:\n\n{text}",
        ),
    ]
)

_REDUCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert in Digital Image Processing. "
            "Combine the following partial summaries into a single, coherent "
            "chapter summary (≤400 words). Highlight the most important concepts "
            "and any mathematical foundations:\n\n{summaries}",
        ),
    ]
)


@lru_cache(maxsize=1)
def build_summarization_chain():
    """
    Return a runnable summarisation chain.

    Returns
    -------
    Runnable
        Accepts a list of ``Document`` objects and returns a summary string.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains.summarize import load_summarize_chain

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.1,
    )

    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=_MAP_PROMPT,
        combine_prompt=_REDUCE_PROMPT,
        verbose=False,
    )
    return chain
