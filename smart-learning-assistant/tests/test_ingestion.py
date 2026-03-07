# -*- coding: utf-8 -*-
"""
tests/test_ingestion.py
-----------------------
Unit tests for the ingestion and retrieval pipeline.

All tests mock external dependencies (Chroma, LLM) so they run fully offline
in < 1 second each with no API keys or disk IO required.

Test inventory:
  1. test_chunk_count_reasonable         — chunk_documents returns ≥ n_pages chunks
  2. test_chunk_metadata_preserved       — every chunk has 'source' and 'page' keys
  3. test_load_vectorstore_raises_if_missing — FileNotFoundError on bad path
  4. test_retriever_returns_documents    — get_retriever().invoke() returns documents
  5. test_guardrail_rejects_offtopic     — high L2 distance → empty list
  6. test_guardrail_passes_ontopic       — low L2 distance → non-empty list

Implementation notes
--------------------
- chunk_documents() takes list[dict], each: {"text": str, "metadata": dict}
- load_vectorstore() is in app.ingestion.pipeline; retriever.py imports it from there
- get_guardrail_retriever(threshold) returns a Callable[[str], list[Document]]
- To avoid hitting disk we monkeypatch app.ingestion.pipeline.load_vectorstore
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Helper: build fake page dicts (the format chunk_documents actually expects)
# ---------------------------------------------------------------------------
def _make_page_dicts(n: int) -> list[dict]:
    return [
        {
            "text": (
                f"Page {i}: Spatial filtering applies a convolution mask to every pixel. "
                "Linear filters include averaging and sharpening. "
                "Nonlinear filters include the median filter for salt-and-pepper noise."
            ),
            "metadata": {
                "source": "Digital_Image_Processing_Gonzalez_Woods_4th_Ed.pdf",
                "page": i,
                "category": "textbook",
                "file_path": "/data/raw/1_textbooks/dip.pdf",
            },
        }
        for i in range(n)
    ]


# ===========================================================================
# 1. chunk_count_reasonable
# ===========================================================================
def test_chunk_count_reasonable():
    """chunk_documents with 10 pages should produce at least 10 chunks."""
    from app.ingestion.pipeline import chunk_documents

    pages = _make_page_dicts(10)
    chunks = chunk_documents(pages)

    assert len(chunks) >= 10, (
        f"Expected at least 10 chunks from 10 pages, got {len(chunks)}"
    )


# ===========================================================================
# 2. chunk_metadata_preserved
# ===========================================================================
def test_chunk_metadata_preserved():
    """Every chunk must carry 'source' and 'page' metadata from the parent page."""
    from app.ingestion.pipeline import chunk_documents

    pages = _make_page_dicts(2)
    chunks = chunk_documents(pages)

    assert chunks, "chunk_documents returned an empty list"
    for chunk in chunks:
        assert "source" in chunk.metadata, f"Chunk missing 'source': {chunk.metadata}"
        assert "page"   in chunk.metadata, f"Chunk missing 'page':   {chunk.metadata}"


# ===========================================================================
# 3. load_vectorstore raises FileNotFoundError on missing path
# ===========================================================================
def test_load_vectorstore_raises_if_missing(tmp_path, monkeypatch):
    """load_vectorstore must raise FileNotFoundError when the persist dir is absent."""
    from app.ingestion import pipeline

    non_existent = str(tmp_path / "does_not_exist" / "chroma_db")
    # Override the env var so the function resolves to the non-existent path
    monkeypatch.setenv("CHROMA_PERSIST_DIR", non_existent)

    with pytest.raises(FileNotFoundError):
        pipeline.load_vectorstore()


# ===========================================================================
# 4. retriever_returns_documents
# ===========================================================================
def test_retriever_returns_documents(mock_vectorstore, monkeypatch):
    """get_retriever should return at least one Document for a DIP query."""
    from app.ingestion import pipeline as _pipeline
    from app.retrieval.retriever import get_retriever

    # Patch the load_vectorstore that retriever.py imported at the top of its module
    monkeypatch.setattr(_pipeline, "load_vectorstore", lambda *a, **kw: mock_vectorstore)
    import app.retrieval.retriever as _retriever_mod
    monkeypatch.setattr(_retriever_mod, "load_vectorstore", lambda *a, **kw: mock_vectorstore)

    retriever = get_retriever()
    results = retriever.invoke("What is spatial filtering?")

    assert len(results) > 0, "Retriever returned no documents"
    assert hasattr(results[0], "page_content"), "Result is not a Document"


# ===========================================================================
# 5. guardrail rejects off-topic (high L2 distance)
# ===========================================================================
def test_guardrail_rejects_offtopic(mock_vectorstore, monkeypatch):
    """Off-topic query (L2 distance >= threshold) must return empty list."""
    from app.retrieval.retriever import get_guardrail_retriever
    import app.retrieval.retriever as _retriever_mod

    off_topic_doc = Document(
        page_content="The boiling point of water is 100 degrees Celsius at sea level.",
        metadata={"source": "N/A", "page": 0},
    )
    mock_vectorstore.similarity_search_with_score.return_value = [
        (off_topic_doc, 0.85),  # high distance → off-topic
    ]
    monkeypatch.setattr(_retriever_mod, "load_vectorstore", lambda *a, **kw: mock_vectorstore)

    guardrail_fn = get_guardrail_retriever(threshold=0.50)
    result = guardrail_fn("What is the boiling point of water?")

    assert result == [], (
        f"Guardrail should reject off-topic query but returned: {result}"
    )


# ===========================================================================
# 6. guardrail passes on-topic (low L2 distance)
# ===========================================================================
def test_guardrail_passes_ontopic(mock_vectorstore, monkeypatch):
    """On-topic query (L2 distance < threshold) must return non-empty list."""
    from app.retrieval.retriever import get_guardrail_retriever
    import app.retrieval.retriever as _retriever_mod

    # mock_vectorstore already returns distance=0.15 by default (see conftest.py)
    monkeypatch.setattr(_retriever_mod, "load_vectorstore", lambda *a, **kw: mock_vectorstore)

    guardrail_fn = get_guardrail_retriever(threshold=0.50)
    result = guardrail_fn("Explain linear spatial filtering.")

    assert len(result) > 0, (
        "Guardrail should pass on-topic query but returned empty list"
    )
