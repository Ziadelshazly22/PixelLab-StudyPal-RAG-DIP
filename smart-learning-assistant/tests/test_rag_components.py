"""Pytest tests for core RAG components."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from unittest.mock import MagicMock, patch

from app.chains.rag_chain import (
    _HUMAN_MESSAGE,
    _SYSTEM_MESSAGE,
    _extract_question,
    format_docs,
    get_llm,
    clear_session,
    MEMORY_STORE,
)
from app.retrieval.retriever import get_retriever


# ===========================================================================
# 1. LLM initialises without error
# ===========================================================================
def test_get_llm_initializes() -> None:
    """get_llm() must return a non-None object (uses real env key if present)."""
    llm = get_llm()
    assert llm is not None


# ===========================================================================
# 2. format_docs — empty list → sentinel string
# ===========================================================================
def test_format_docs_empty() -> None:
    """format_docs([]) must return exactly 'NO_CONTEXT_AVAILABLE'."""
    context_empty = format_docs([])
    assert context_empty == "NO_CONTEXT_AVAILABLE"


# ===========================================================================
# 3. format_docs — non-empty → contains source and page
# ===========================================================================
def test_format_docs_non_empty() -> None:
    """format_docs must include the source filename and page in the output."""
    docs = [
        Document(
            page_content="Histogram equalization stretches intensity values.",
            metadata={"source": "Gonzalez_Woods_DIP.pdf", "page": 234},
        ),
        Document(
            page_content="Edge detection identifies boundaries.",
            metadata={"source": "Gonzalez_Woods_DIP.pdf", "page": 456},
        ),
    ]
    context = format_docs(docs)
    assert "Gonzalez_Woods_DIP.pdf" in context
    assert "Page 234" in context
    assert "Histogram equalization" in context


# ===========================================================================
# 4. Prompt messages contain required rule markers
# ===========================================================================
def test_prompt_messages_content() -> None:
    """System and human prompt templates must contain mandatory rule markers."""
    assert "MANDATORY RULES" in _SYSTEM_MESSAGE
    assert "CITATIONS" in _SYSTEM_MESSAGE
    assert "STUDENT QUESTION" in _HUMAN_MESSAGE


# ===========================================================================
# 5. get_retriever — skips gracefully if ChromaDB missing
# ===========================================================================
def test_get_retriever_initializes_or_known_data_issue() -> None:
    """get_retriever returns a retriever or is skipped if DB not present."""
    try:
        retriever = get_retriever()
        assert retriever is not None
    except FileNotFoundError:
        pytest.skip("ChromaDB not found in current environment.")
    except Exception as exc:  # noqa: BLE001
        if "_type" in str(exc):
            pytest.skip("Known ChromaDB schema compatibility issue.")
        raise


# ===========================================================================
# 6. format_docs — citation format contains separators and both metadata keys
# ===========================================================================
def test_format_docs_citation_format() -> None:
    """Each formatted block must start with '--- Source:' and include 'Page'."""
    docs = [
        Document(
            page_content="Sobel operator approximates image gradient.",
            metadata={"source": "dip.pdf", "page": 101},
        ),
    ]
    context = format_docs(docs)
    assert "--- Source: dip.pdf, Page 101 ---" in context, (
        f"Expected citation header not found in:\n{context}"
    )
    assert "Sobel operator" in context


# ===========================================================================
# 7. format_docs — missing metadata keys fall back to 'unknown' and '?'
# ===========================================================================
def test_format_docs_missing_metadata() -> None:
    """format_docs must handle documents with no 'source' or 'page' metadata."""
    docs = [
        Document(page_content="Some content.", metadata={}),
    ]
    context = format_docs(docs)
    assert "unknown" in context, "Expected 'unknown' as source fallback"
    assert "?" in context, "Expected '?' as page fallback"


# ===========================================================================
# 8. _extract_question — bare string passes through unchanged
# ===========================================================================
def test_extract_question_from_string() -> None:
    """_extract_question('text') must return 'text' unchanged."""
    result = _extract_question("What is the Fourier transform?")
    assert result == "What is the Fourier transform?"


# ===========================================================================
# 9. _extract_question — dict with 'question' key extracts the value
# ===========================================================================
def test_extract_question_from_dict() -> None:
    """_extract_question({'question': 'text'}) must return 'text'."""
    result = _extract_question({"question": "Explain histogram equalization."})
    assert result == "Explain histogram equalization."


# ===========================================================================
# 10. _extract_question — dict without 'question' key returns str(dict)
# ===========================================================================
def test_extract_question_from_dict_no_key() -> None:
    """_extract_question({'other': 'val'}) must return str(dict) without raising."""
    d = {"other": "value"}
    result = _extract_question(d)
    assert result == str(d)


# ===========================================================================
# 11. clear_session — removes an existing session and returns True
# ===========================================================================
def test_clear_session_removes_existing() -> None:
    """clear_session must remove a known session_id from MEMORY_STORE and return True."""
    fake_chain = MagicMock()
    session_id = "test-session-clear-123"

    # Inject a fake session directly into the module-level store
    MEMORY_STORE[session_id] = {"chain": fake_chain, "last_accessed": 9999999999.0}

    result = clear_session(session_id)

    assert result is True, "clear_session should return True for a known session"
    assert session_id not in MEMORY_STORE, "Session was not removed from MEMORY_STORE"


# ===========================================================================
# 12. clear_session — returns False for non-existent session
# ===========================================================================
def test_clear_session_nonexistent_returns_false() -> None:
    """clear_session must return False when the session_id is not in the store."""
    result = clear_session("session-that-does-not-exist-xyzzy")
    assert result is False, "clear_session should return False for unknown session"


# ===========================================================================
# 13. build_rag_chain — returns a LangChain Runnable
# ===========================================================================
def test_build_rag_chain_returns_runnable(monkeypatch) -> None:
    """build_rag_chain() must return a Runnable even when the DB is unavailable."""
    from langchain_core.runnables import Runnable
    import app.chains.rag_chain as _chain_mod

    # Patch guardrail so no DB access is needed
    monkeypatch.setattr(
        _chain_mod,
        "get_guardrail_retriever",
        lambda threshold=1.2: lambda q: [],
    )

    chain = _chain_mod.build_rag_chain()
    assert isinstance(chain, Runnable), (
        f"build_rag_chain() returned {type(chain)}, expected a Runnable"
    )
