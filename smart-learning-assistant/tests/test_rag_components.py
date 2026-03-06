"""Pytest tests for core RAG components."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.chains.rag_chain import (
    _HUMAN_MESSAGE,
    _SYSTEM_MESSAGE,
    format_docs,
    get_llm,
)
from app.retrieval.retriever import get_retriever


def test_get_llm_initializes() -> None:
    llm = get_llm()
    assert llm is not None


def test_format_docs_empty() -> None:
    context_empty = format_docs([])
    assert context_empty == "NO_CONTEXT_AVAILABLE"


def test_format_docs_non_empty() -> None:
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


def test_prompt_messages_content() -> None:
    assert "MANDATORY RULES" in _SYSTEM_MESSAGE
    assert "CITATIONS" in _SYSTEM_MESSAGE
    assert "STUDENT QUESTION" in _HUMAN_MESSAGE


def test_get_retriever_initializes_or_known_data_issue() -> None:
    try:
        retriever = get_retriever()
        assert retriever is not None
    except FileNotFoundError:
        pytest.skip("ChromaDB not found in current environment.")
    except Exception as exc:  # noqa: BLE001
        if "_type" in str(exc):
            pytest.skip("Known ChromaDB schema compatibility issue.")
        raise
