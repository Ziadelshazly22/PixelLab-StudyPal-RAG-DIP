"""
app/retrieval/retriever.py
--------------------------
Vector store retrieval module for the Smart Learning Assistant.

Provides two retriever functions:
1. get_retriever() — Standard MMR (Maximal Marginal Relevance) retrieval for diversity.
2. get_guardrail_retriever() — Wraps the retriever with a similarity threshold to filter
   out-of-domain queries and prevent hallucination on irrelevant topics.

The guardrail is essential: queries too distant from the knowledge base return no context,
triggering the chain's refusal message instead of a fabricated answer.

Usage
-----
    from app.retrieval.retriever import get_guardrail_retriever
    
    retriever_fn = get_guardrail_retriever(threshold=0.25)
    results = retriever_fn("What is edge detection?")
    # Returns top-k documents if top-1 similarity is above threshold, else [].
"""

from __future__ import annotations

import logging
from typing import Callable

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever

from app.ingestion.pipeline import load_vectorstore

logger = logging.getLogger(__name__)


class _EmptyRetriever(BaseRetriever):
    """Fallback retriever that always returns no documents."""

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:  # noqa: ARG002
        return []


def get_retriever(k: int = 5, fetch_k: int = 20) -> BaseRetriever:
    """
    Build a Maximal Marginal Relevance (MMR) retriever over ChromaDB.

    MMR balances relevance and diversity: it returns the top-k documents that are
    most relevant to the query while being maximally diverse from each other.
    This prevents the chain from receiving k nearly-identical passages, improving
    response variety and reducing redundancy.

    Args
    ----
    k : int, optional
        Number of documents to return (default: 5).
    fetch_k : int, optional
        Number of documents to fetch internally before reranking by MMR (default: 20).
        Larger fetch_k improves diversity but increases latency.

    Returns
    -------
    VectorStoreRetriever
        A retriever configured for MMR search. If the installed langchain-community
        version does not support MMR (rare), falls back to standard similarity search.

    Raises
    ------
    FileNotFoundError
        If the ChromaDB directory does not exist. Ensure the ingestion pipeline
        has been run first.

    Notes
    -----
    - The retriever is stateless and safe to call repeatedly.
    - MMR computation is O(k²) and may add ~200-500ms latency for k=5.
    - If latency is critical, use similarity search by setting search_type="similarity"
      and omitting fetch_k.

    Examples
    --------
    >>> retriever = get_retriever(k=5)
    >>> docs = retriever.invoke("What is histogram equalization?")
    >>> print(f"Retrieved {len(docs)} documents")
    """
    try:
        vectorstore = load_vectorstore()
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Vectorstore load failed (%s). Falling back to empty retriever.",
            exc,
        )
        return _EmptyRetriever()

    # Attempt MMR; fall back to similarity if older LangChain version.
    try:
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k},
        )
        logger.info(
            f"Retriever initialized with MMR (k={k}, fetch_k={fetch_k}). "
            "Query results will be diverse and relevant."
        )
    except AttributeError:
        logger.warning(
            f"MMR not supported in this langchain-community version. "
            f"Falling back to similarity search (k={k})."
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Retriever initialization failed (%s). Falling back to empty retriever.",
            exc,
        )
        return _EmptyRetriever()

    return retriever


def get_guardrail_retriever(threshold: float = 1.2) -> Callable[[str], list[Document]]:
    """
    Build a guardrail-wrapped retriever that filters out-of-domain queries.

    The guardrail computes the similarity score (Chroma distance) of the top-1 result
    against the query. If the score exceeds the threshold (meaning the query is very
    dissimilar from all knowledge base content), the retriever returns an empty list.

    This empty context triggers the RAG chain's mandatory refusal message:
    "This question falls outside my knowledge base…"

    Use this to prevent the LLM from inventing answers to off-topic questions.

    Args
    ----
    threshold : float, optional
        Similarity threshold. Distance >= threshold is considered out-of-domain
        (default: 0.25). Typical range: [0.15, 0.35] depending on embedding model.
        - Lower values (0.15) → more lenient (fewer refusals)
        - Higher values (0.35) → more strict (more refusals)

    Returns
    -------
    Callable[[str], list[Document]]
        A function that accepts a query string and returns either:
        - Empty list (query is out-of-domain)
        - Top-k documents from the retriever (query is in-domain)

    Notes
    -----
    - Initialization of Chroma is lazy (at first query), so API startup remains healthy.
    - The threshold should be tuned empirically by testing edge cases
      (e.g., "What is the capital of France?" should return []).
    - Chroma uses L2 distance, so lower values = higher similarity.
    - The guardrail is transparent to the LCEL chain — pass the returned function
      to RunnableLambda() in the chain definition.

    Examples
    --------
    >>> guardrail_fn = get_guardrail_retriever(threshold=0.25)
    >>>
    >>> # In-domain query
    >>> dip_docs = guardrail_fn("What is histogram equalization?")
    >>> print(f"DIP question: {len(dip_docs)} documents retrieved")
    >>>
    >>> # Out-of-domain query
    >>> off_topic_docs = guardrail_fn("What is the capital of France?")
    >>> print(f"Off-topic question: {len(off_topic_docs)} documents (guardrail active)")
    """
    vectorstore = None
    retriever_raw = None

    def guardrail_retriever(query: str) -> list[Document]:
        """
        Retrieve documents if query is in-domain, else return empty list.

        Args
        ----
        query : str
            The user's question.

        Returns
        -------
        list[Document]
            Either [] (out-of-domain) or top-k documents (in-domain).
        """
        nonlocal vectorstore, retriever_raw

        # Lazy initialize vectorstore/retriever on first request.
        if vectorstore is None or retriever_raw is None:
            try:
                vectorstore = load_vectorstore()
                retriever_raw = get_retriever()
            except Exception as e:
                logger.error("Retriever initialization failed: %s", e)
                return []

        # Compute similarity of top-1 result.
        try:
            top1_with_score = vectorstore.similarity_search_with_score(query, k=1)
            if not top1_with_score:
                logger.debug("Guardrail: no results for query. Returning [].")
                return []

            _, score = top1_with_score[0]
            logger.info("Guardrail score=%.3f threshold=%.3f", score, threshold)

            # Chroma uses L2 distance; higher score = lower similarity.
            if score >= threshold:
                logger.info(
                    "Guardrail activated: query too dissimilar (score %.3f >= threshold %.3f). Returning no context.",
                    score,
                    threshold,
                )
                return []
        except Exception as e:
            logger.warning("Guardrail error during similarity check: %s. Returning [].", e)
            return []

        # Query is in-domain; return full top-k retrieval.
        results = retriever_raw.invoke(query)
        logger.debug("Guardrail passed: returned %d documents.", len(results))
        return results

    return guardrail_retriever
