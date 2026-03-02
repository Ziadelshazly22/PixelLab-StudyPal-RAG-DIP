"""
app/retrieval/retriever.py
--------------------------
Builds a LangChain-compatible retriever backed by a persistent ChromaDB
collection.  Uses Maximal Marginal Relevance (MMR) search for diversity.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

_CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
_COLLECTION = os.getenv("COLLECTION_NAME", "dip_knowledge_base")


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
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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
