"""
app/retrieval/retriever.py
--------------------------
Builds a LangChain-compatible retriever backed by a persistent ChromaDB
collection.  Uses Maximal Marginal Relevance (MMR) search for diversity.

Embedding strategy
------------------
- Primary  : Google ``text-embedding-004``  (when GOOGLE_API_KEY is set)
- Fallback : ``all-MiniLM-L6-v2``           (sentence-transformers, local)

The embedding model used here **must match** the one used during ingestion;
otherwise the stored vectors will be incompatible with the query vector.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

_CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
_COLLECTION = os.getenv("COLLECTION_NAME", "dip_knowledge_base")
_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")


def _get_embeddings():
    """
    Return the embedding model matching the one used during ingestion.

    Strategy
    --------
    - If ``GOOGLE_API_KEY`` is present, use Google ``text-embedding-004``.
    - Otherwise fall back to local ``all-MiniLM-L6-v2`` (no API key required).
    """
    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=_EMBEDDING_MODEL)

    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def build_retriever(k: int = 6, fetch_k: int = 20):
    """
    Return an MMR retriever over the persisted ChromaDB collection.

    Parameters
    ----------
    k : int
        Number of documents to return per query.
    fetch_k : int
        Candidate pool size for MMR re-ranking.

    Returns
    -------
    VectorStoreRetriever
    """
    from langchain_chroma import Chroma

    embeddings = _get_embeddings()
    vector_store = Chroma(
        collection_name=_COLLECTION,
        embedding_function=embeddings,
        persist_directory=_CHROMA_DIR,
    )
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k},
    )
    return retriever
