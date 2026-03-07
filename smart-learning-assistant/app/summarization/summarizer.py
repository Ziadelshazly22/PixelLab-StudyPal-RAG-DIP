# -*- coding: utf-8 -*-
"""
app/summarization/summarizer.py
--------------------------------
Chapter summarisation and study-question generation for the DIP AI Tutor.

Functions
---------
get_source_chunks(source_filename)
    Retrieve all ChromaDB chunks belonging to one source PDF, sorted by page.

summarize_document(source_filename)
    Map-reduce summarisation over all chunks of a document.
    Resilient to Gemini 429 rate limits via tenacity retry.

generate_study_questions(source_filename, n)
    Generate *n* exam-style questions from a representative sample of the document.
"""

from __future__ import annotations

import logging
import re

from dotenv import load_dotenv
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates for map-reduce summarisation
# ---------------------------------------------------------------------------

_MAP_TEMPLATE = """\
You are summarising a section of a Digital Image Processing textbook.
Preserve: key concepts, algorithm names, mathematical formulas (LaTeX), and any cited theorems.
Be concise but technically precise:

{text}"""

_COMBINE_TEMPLATE = """\
You are a DIP expert creating a comprehensive study guide chapter summary from individual section summaries.
Organise by: (1) Core Concepts, (2) Key Algorithms & Formulas,
(3) Practical Applications, (4) Common Exam Topics.
Maintain all LaTeX notation:

{text}"""


# ---------------------------------------------------------------------------
# Helper: retrieve all chunks for one source file
# ---------------------------------------------------------------------------

def get_source_chunks(source_filename: str) -> list[Document]:
    """Retrieve all ChromaDB chunks that belong to *source_filename*, sorted by page.

    Uses a metadata ``where`` filter so only chunks from the requested file are
    returned — no similarity scoring is applied.

    Args:
        source_filename: Exact filename stored in chunk metadata (e.g.
            ``"Digital_Image_Processing_Gonzalez_Woods_4th_Ed.pdf"``).

    Returns:
        List of :class:`~langchain_core.documents.Document` objects sorted
        ascending by ``metadata["page"]``.

    Raises:
        ValueError: If *source_filename* matches no chunks in the store.

    Example:
        >>> docs = get_source_chunks("Gonzalez_Woods_4th_Ed.pdf")
        >>> print(len(docs), "chunks retrieved")
    """
    from app.ingestion.pipeline import load_vectorstore

    vs = load_vectorstore()

    # Primary path requested by spec: similarity_search with empty query + metadata filter.
    try:
        docs = vs.similarity_search("", k=500, filter={"source": source_filename})
    except Exception as exc:
        logger.warning(
            "similarity_search retrieval failed (%s); falling back to metadata get().", exc
        )

        # Fallback path: retrieve by metadata where-clause directly.
        raw = vs._collection.get(
            where={"source": source_filename},
            limit=500,
            include=["documents", "metadatas"],
        )
        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(raw["documents"], raw["metadatas"])
            if text and text.strip()
        ]

    # Some Chroma versions return empty results for empty-query similarity search.
    if not docs:
        raw = vs._collection.get(
            where={"source": source_filename},
            limit=500,
            include=["documents", "metadatas"],
        )
        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(raw["documents"], raw["metadatas"])
            if text and text.strip()
        ]

    if not docs:
        raise ValueError(
            f"No chunks found for source '{source_filename}'. "
            "Verify the filename matches the 'source' metadata stored during ingestion."
        )

    docs.sort(key=lambda d: d.metadata.get("page", 0))
    logger.info("Retrieved %d chunks for '%s'.", len(docs), source_filename)
    return docs


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    reraise=True,
)
def _invoke_summarize_chain(chain, docs: list[Document]) -> str:
    """Invoke the summarisation chain with tenacity retry on Gemini 429."""
    result = chain.invoke(docs)
    # load_summarize_chain returns either a str or {"output_text": str}
    if isinstance(result, dict):
        return result.get("output_text", str(result))
    return str(result)


def summarize_document(source_filename: str) -> str:
    """Generate a map-reduce academic summary for *source_filename*.

    Retrieves all chunks from ChromaDB, optionally samples every 3rd chunk
    if the total text exceeds 100,000 characters (to stay within Gemini's
    context window), then runs a map-reduce summarisation chain.

    Gemini 429 rate-limit errors are retried up to 3 times with exponential
    back-off (4 s → 8 s → 60 s) via :func:`tenacity`.

    Args:
        source_filename: Exact filename as stored in chunk metadata.

    Returns:
        Markdown-formatted summary string headed with
        ``# Summary: {source_filename}``.

    Raises:
        ValueError: Propagated from :func:`get_source_chunks` if no chunks found.

    Example:
        >>> summary = summarize_document("Gonzalez_Woods_4th_Ed.pdf")
        >>> print(summary[:200])
    """
    from langchain_classic.chains.summarize import load_summarize_chain
    from langchain_core.prompts import PromptTemplate
    from app.chains.rag_chain import get_llm

    docs = get_source_chunks(source_filename)

    # Sample every 3rd chunk if content is very large
    total_chars = sum(len(d.page_content) for d in docs)
    if total_chars > 100_000:
        docs = docs[::3]
        logger.info(
            "Sampling every 3rd chunk for '%s' (total chars: %d > 100,000).",
            source_filename, total_chars,
        )

    map_prompt = PromptTemplate(input_variables=["text"], template=_MAP_TEMPLATE)
    combine_prompt = PromptTemplate(input_variables=["text"], template=_COMBINE_TEMPLATE)

    chain = load_summarize_chain(
        llm=get_llm(),
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        verbose=False,
    )

    logger.info(
        "Starting map-reduce summarisation for '%s' (%d chunks).",
        source_filename, len(docs),
    )
    summary_text = _invoke_summarize_chain(chain, docs)
    logger.info("Summarisation complete for '%s'.", source_filename)

    return f"# Summary: {source_filename}\n\n{summary_text}"


# ---------------------------------------------------------------------------
# Study question generation
# ---------------------------------------------------------------------------

def generate_study_questions(source_filename: str, n: int = 5) -> list[str]:
    """Generate *n* exam-style study questions from *source_filename*.

    Samples up to 20 evenly-spaced representative chunks from the document,
    concatenates their text, and prompts the LLM to produce *n* questions
    covering conceptual, mathematical, and applied question types.

    Args:
        source_filename: Exact filename as stored in chunk metadata.
        n: Number of questions to generate (default 5; range 1–10).

    Returns:
        List of *n* question strings (without numbering prefixes).

    Raises:
        ValueError: Propagated from :func:`get_source_chunks` if no chunks found.

    Example:
        >>> questions = generate_study_questions("Gonzalez_Woods_4th_Ed.pdf", n=5)
        >>> for q in questions:
        ...     print("-", q)
    """
    from app.chains.rag_chain import get_llm

    docs = get_source_chunks(source_filename)

    # Sample up to 20 evenly-spaced chunks for representative coverage
    sample_size = min(20, len(docs))
    step = max(1, len(docs) // sample_size)
    sampled = docs[::step][:sample_size]
    context_text = "\n\n".join(d.page_content for d in sampled)

    prompt = (
        f"Based on the following Digital Image Processing content, generate exactly {n} "
        "exam-style questions. Mix question types:\n"
        "  • Conceptual (define / explain)\n"
        "  • Mathematical (derive / calculate)\n"
        "  • Applied (implement / compare)\n"
        "Format as a numbered list. Provide only the questions — no answers.\n\n"
        f"CONTENT:\n{context_text}\n\n"
        f"EXACTLY {n} EXAM QUESTIONS:"
    )

    llm = get_llm()
    response = llm.invoke(prompt)
    raw_text = response.content if hasattr(response, "content") else str(response)

    # Parse numbered list: "1. Question text" → "Question text"
    questions: list[str] = []
    for line in raw_text.splitlines():
        line = line.strip()
        match = re.match(r"^\d+[\.\)]\s+(.+)", line)
        if match:
            questions.append(match.group(1))
        elif line and not re.match(r"^(EXACTLY|CONTENT|exam|question)", line, re.I):
            # Catch un-numbered lines that look like questions
            if line.endswith("?") or (len(line) > 20 and len(questions) < n):
                questions.append(line)

    questions = [q for q in questions if q][:n]
    logger.info(
        "Generated %d study questions for '%s'.", len(questions), source_filename
    )
    return questions
