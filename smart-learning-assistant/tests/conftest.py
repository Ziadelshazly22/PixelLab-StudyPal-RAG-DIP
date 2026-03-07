# -*- coding: utf-8 -*-
"""
tests/conftest.py
-----------------
Shared pytest fixtures for the DIP AI Tutor test suite.

All fixtures use unittest.mock so no real Chroma DB or LLM calls are made.
Tests are therefore fully offline and run in < 1 second each.
"""

import os
import sys

# Ensure project root is on sys.path regardless of how pytest is invoked
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Reusable fake document
# ---------------------------------------------------------------------------
_FAKE_DOC = Document(
    page_content=(
        "Spatial filtering applies a convolution mask to every pixel in the image. "
        "Linear spatial filters include smoothing (averaging) and sharpening filters. "
        "Nonlinear filters, such as the median filter, are better for salt-and-pepper noise."
    ),
    metadata={
        "source": "Digital_Image_Processing_Gonzalez_Woods_4th_Ed.pdf",
        "page": 42,
    },
)


# ---------------------------------------------------------------------------
# mock_vectorstore fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_vectorstore():
    """Return a MagicMock that mimics a loaded Chroma vectorstore.

    Behavior:
    - _collection.count()                 -> 500
    - similarity_search(query)            -> [_FAKE_DOC, _FAKE_DOC]
    - similarity_search_with_score(query) -> [(_FAKE_DOC, 0.15)]
    - as_retriever().invoke(query)        -> [_FAKE_DOC]
    """
    vs = MagicMock()

    # Collection metadata
    vs._collection = MagicMock()
    vs._collection.count.return_value = 500

    # Plain similarity search
    vs.similarity_search.return_value = [_FAKE_DOC, _FAKE_DOC]

    # Scored search — low distance means ON-topic
    vs.similarity_search_with_score.return_value = [(_FAKE_DOC, 0.15)]

    # Retriever
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [_FAKE_DOC]
    vs.as_retriever.return_value = mock_retriever

    return vs


# ---------------------------------------------------------------------------
# mock_llm fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_llm():
    """Return a MagicMock that mimics a ChatGoogleGenerativeAI instance.

    .invoke() returns a message object whose .content is a plausible DIP answer.
    """
    llm = MagicMock()

    msg = MagicMock()
    msg.content = (
        "Spatial filtering involves applying a convolution kernel to each pixel of "
        "an image. Linear filters include the box filter and Gaussian filter, while "
        "nonlinear filters include the median filter, which is effective against "
        "salt-and-pepper noise. [Source: Digital_Image_Processing_Gonzalez_Woods_4th_Ed.pdf, Page 42]"
    )
    llm.invoke.return_value = msg

    return llm
