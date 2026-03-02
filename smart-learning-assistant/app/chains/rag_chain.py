"""
app/chains/rag_chain.py
-----------------------
Builds the core Retrieval-Augmented Generation (RAG) chain using
LangChain LCEL (LangChain Expression Language).

Strategy:
  - Primary LLM  : Gemini 2.0 Flash  (fast, grounded answers)
  - Fallback LLM : DeepSeek-R1       (mathematical reasoning)
  - Retriever    : ChromaDB MMR      (maximal marginal relevance)
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are an expert AI tutor specialising in Digital Image Processing (DIP). "
    "Answer the student's question using ONLY the retrieved context below. "
    "If the answer is not in the context, say so clearly and suggest where to look. "
    "Cite the source document and page number when available.\n\n"
    "Context:\n{context}\n"
)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


# ---------------------------------------------------------------------------
# Helper: format retrieved docs into a single string
# ---------------------------------------------------------------------------
def _format_docs(docs: list) -> str:
    parts = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[{src} – p.{page}]\n{doc.page_content}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Chain factory
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def build_rag_chain(retriever=None):  # type: ignore[assignment]
    """
    Return a runnable RAG chain.

    Parameters
    ----------
    retriever : BaseRetriever, optional
        A LangChain-compatible retriever.  When ``None`` the function returns
        a *stub* chain that can be imported without a live ChromaDB instance –
        useful during development and testing.

    Returns
    -------
    Runnable
        LCEL chain: ``{"context": ..., "question": ...} | prompt | llm | parser``
    """
    if retriever is None:
        # Stub: echo the question back so the app starts without a vector store
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(
            lambda x: (
                f"[STUB] RAG chain not initialised. "
                f"Your question was: {x.get('question', x)}"
            )
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.2,
    )

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain
