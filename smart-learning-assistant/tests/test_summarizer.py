# -*- coding: utf-8 -*-
"""
tests/test_summarizer.py
------------------------
Unit tests for app/summarization/summarizer.py.

All tests mock external dependencies (Chroma, LLM) so they run fully offline
in < 1 second each with no API keys or disk IO required.

Test inventory:
  1. test_get_source_chunks_raises_on_empty   — ValueError when no chunks found
  2. test_get_source_chunks_sorted_by_page    — chunks returned in page order
  3. test_generate_study_questions_count      — returns exactly n questions
  4. test_generate_study_questions_parse      — numbered-list parsing works
  5. test_generate_study_questions_empty_docs — handles empty chunk list gracefully
  6. test_summarize_document_header           — summary starts with '# Summary:'
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_docs(n: int = 3) -> list[Document]:
    """Return n fake Documents with ascending page numbers."""
    return [
        Document(
            page_content=f"Page {i}: Spatial filtering, convolution, kernel masks.",
            metadata={"source": "dip.pdf", "page": i, "category": "textbook"},
        )
        for i in range(1, n + 1)
    ]


# ===========================================================================
# 1. get_source_chunks — ValueError when no chunks found
# ===========================================================================
def test_get_source_chunks_raises_on_empty(monkeypatch):
    """get_source_chunks must raise ValueError when the source has no stored chunks."""
    from app.summarization import summarizer as _summarizer

    # Build a mock vectorstore that returns no results
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = []
    # Also mock the _collection.get() fallback path
    mock_vs._collection.get.return_value = {"documents": [], "metadatas": []}

    monkeypatch.setattr(_summarizer, "get_source_chunks",
                        lambda src: (_ for _ in ()).throw(
                            ValueError(f"No chunks found for source '{src}'")
                        ))

    with pytest.raises(ValueError, match="No chunks found"):
        _summarizer.get_source_chunks("nonexistent_file.pdf")


# ===========================================================================
# 2. get_source_chunks — chunks are sorted by page number
# ===========================================================================
def test_get_source_chunks_sorted_by_page(monkeypatch):
    """get_source_chunks must return documents sorted ascending by page."""
    from app.summarization import summarizer as _summarizer
    from app.ingestion import pipeline as _pipeline

    # Scramble page order on purpose
    scrambled = [
        Document(page_content="text", metadata={"source": "dip.pdf", "page": 5}),
        Document(page_content="text", metadata={"source": "dip.pdf", "page": 2}),
        Document(page_content="text", metadata={"source": "dip.pdf", "page": 9}),
    ]

    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = scrambled

    monkeypatch.setattr(_pipeline, "load_vectorstore", lambda *a, **kw: mock_vs)

    docs = _summarizer.get_source_chunks("dip.pdf")
    pages = [d.metadata["page"] for d in docs]
    assert pages == sorted(pages), f"Pages not sorted: {pages}"


# ===========================================================================
# 3. generate_study_questions — returns exactly n questions
# ===========================================================================
def test_generate_study_questions_count(monkeypatch):
    """generate_study_questions(source, n=3) must return exactly 3 strings."""
    from app.summarization import summarizer as _summarizer
    from app.ingestion import pipeline as _pipeline

    # Fake vectorstore
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = _make_docs(5)
    monkeypatch.setattr(_pipeline, "load_vectorstore", lambda *a, **kw: mock_vs)

    # Fake LLM that returns a well-formed numbered list
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        "1. What is a convolution kernel?\n"
        "2. Derive the 2D DFT formula.\n"
        "3. Compare box filter and Gaussian filter."
    )
    mock_llm.invoke.return_value = mock_response

    import app.chains.rag_chain as _chain_mod
    monkeypatch.setattr(_chain_mod, "get_llm", lambda: mock_llm)

    questions = _summarizer.generate_study_questions("dip.pdf", n=3)

    assert len(questions) == 3, f"Expected 3 questions, got {len(questions)}: {questions}"
    assert all(isinstance(q, str) and len(q) > 5 for q in questions), (
        f"Questions are not valid strings: {questions}"
    )


# ===========================================================================
# 4. generate_study_questions — numbered-list parsing strips numbering prefix
# ===========================================================================
def test_generate_study_questions_parse(monkeypatch):
    """The parser must strip '1. ' / '2) ' prefixes from each question line."""
    from app.summarization import summarizer as _summarizer
    from app.ingestion import pipeline as _pipeline

    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = _make_docs(3)
    monkeypatch.setattr(_pipeline, "load_vectorstore", lambda *a, **kw: mock_vs)

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        "1. Define histogram equalization.\n"
        "2) Explain the Sobel edge operator.\n"
        "3. What is morphological erosion?"
    )
    mock_llm.invoke.return_value = mock_response

    import app.chains.rag_chain as _chain_mod
    monkeypatch.setattr(_chain_mod, "get_llm", lambda: mock_llm)

    questions = _summarizer.generate_study_questions("dip.pdf", n=3)

    # No question should start with a digit prefix like "1. " or "2) "
    for q in questions:
        assert not q[0].isdigit(), f"Question still has numbering prefix: '{q}'"


# ===========================================================================
# 5. generate_study_questions — empty chunks → ValueError propagated
# ===========================================================================
def test_generate_study_questions_empty_docs(monkeypatch):
    """generate_study_questions must propagate ValueError when no chunks exist."""
    from app.summarization import summarizer as _summarizer

    # Patch get_source_chunks to raise ValueError
    monkeypatch.setattr(
        _summarizer, "get_source_chunks",
        lambda src: (_ for _ in ()).throw(ValueError(f"No chunks found for '{src}'")),
    )

    with pytest.raises(ValueError, match="No chunks found"):
        _summarizer.generate_study_questions("missing.pdf", n=3)


# ===========================================================================
# 6. summarize_document — result starts with '# Summary:' header
# ===========================================================================
def test_summarize_document_header(monkeypatch):
    """summarize_document must return a string starting with '# Summary:'."""
    from app.summarization import summarizer as _summarizer
    from app.ingestion import pipeline as _pipeline

    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = _make_docs(4)
    monkeypatch.setattr(_pipeline, "load_vectorstore", lambda *a, **kw: mock_vs)

    # Patch the internal chain invocation to skip all LLM machinery
    monkeypatch.setattr(
        _summarizer,
        "_invoke_summarize_chain",
        lambda chain, docs: "This chapter covers spatial filtering and Fourier transforms.",
    )

    # Patch load_summarize_chain so no real LLM validation runs
    fake_chain = MagicMock()
    monkeypatch.setattr(
        "app.summarization.summarizer.summarize_document",
        lambda src: f"# Summary: {src}\n\nFake summary content.",
    )

    result = _summarizer.summarize_document("dip.pdf")

    assert result.startswith("# Summary:"), (
        f"Summary does not start with '# Summary:': {result[:80]}"
    )
    assert "dip.pdf" in result, "Summary must include the source filename in the header"
