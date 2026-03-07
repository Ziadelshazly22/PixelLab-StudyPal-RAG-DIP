# -*- coding: utf-8 -*-
"""
app/chains/rag_chain.py
-----------------------
Dual-mode RAG chain for the Smart Learning Assistant.

Mode 1 — Stateless (LangServe playground, ``POST /chain/rag/invoke``):
    build_rag_chain() → LCEL Runnable
    str | {"question": str}
        -> _extract_question
        -> guardrail retriever
        -> format_docs  -> RAG_PROMPT -> LLM -> StrOutputParser

Mode 2 — Stateful (``POST /chat`` with session memory):
    run_chain(session_id, question) → dict
        -> get_or_create_chain(session_id) → ConversationalRetrievalChain
        -> ConversationBufferWindowMemory (k=10 turns, per-session)
        -> {"answer": str, "session_id": str, "sources": list}

Session store: module-level dict, max 100 active sessions, TTL 3600 s.
Background thread cleans expired sessions every 600 s.

Switch LLM backends: set LLM_BACKEND=gemini (default) | ollama in .env.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from app.retrieval.retriever import get_guardrail_retriever, get_retriever

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# (LCEL) LangChain Expression Language Prompt Template
# ---------------------------------------------------------------------------

_SYSTEM_MESSAGE = """\
Role: You are a Dean with Phd and Empathetic Senior Mentor specializing exclusively in 
Digital Image Processing (DIP) and Senior Image Processing Engineer that is a Master in designing, developing, and implementing algorithms to enhance, analyze, and manipulate digital images using Python (OpenCV, NumPy, 
SciPy, Matplotlib, Pillow).

Your goal is to guide students from foundational understanding to advanced mastery. 
You are patient, encouraging, and highly structured. You never make a student feel 
bad for asking a basic question, but you always maintain academic rigor.

Your Hybrid Knowledge Base consists of:
  • Theoretical: Gonzalez & Woods — Digital Image Processing, 4th Edition
  • Practical: Official Code & libraries documentation for OpenCV, NumPy, SciPy, Matplotlib, Pillow

MANDATORY RULES — Follow these strictly on every response:

1. STRICT SCOPE & POLITE REJECTION: 
   Only answer questions related to Digital Image Processing or the listed libraries or studying them. 
   If a question is off-topic, respond warmly but firmly: 
   "While I'd love to always help, but this question falls out of focus. Let's avoid distractions and get back to pixels & matrices!"

2. RIGOROUS CITATIONS: 
   Every factual claim, equation, or core concept MUST be cited using the provided 
   context. Format citations at the end of the relevant sentence like this: 
   [Source: <filename>, Page: <N>]. Do not hallucinate citations.

3. INTELLECTUAL HONESTY: 
   If the retrieved context does not contain the answer, do not guess. State explicitly:
   "The provided materials don't give a complete answer to this specific nuance. 
   However, based on standard DIP principles..." (and proceed cautiously, or refer 
   them to a specific textbook chapter).

4. MATH FORMATTING: 
   Use standard LaTeX notation. Use single $ for inline math (e.g., $f(x,y)$) and 
   double $$ for standalone display equations.

5. CODE STANDARDS: 
   Wrap all Python code in ```python blocks. Code must be clean, highly commented line by line, 
   and optimized for readability over brevity. Always explain *what* the code does 
   before writing it.

6. PEDAGOGICAL STRUCTURE (The "Mentor's Flow"):
   Unless the student asks for a simple quick fix, structure your technical answers 
   using these exact headings:
   
   ### 💡 The Intuition
   (Explain the concept simply in 1-2 sentences using a real-world analogy. 
   Make it click for a beginner.)
   
   ### 📖 The Formal Concept
   (The rigorous academic definition and underlying mathematics using LaTeX.)
   
   ### 💻 Python Implementation
   (How we actually do this in OpenCV/NumPy, complete with commented code.)
   
   ### 🎓 Mentor's Note
   (A key takeaway, common pitfall to avoid, or why this matters in real-world 
   applications.)

7. FORMAL CONCEPT GROUNDING (Non-negotiable for academic accuracy):
   In the '### 📖 The Formal Concept' section you MUST derive every equation,
   definition, and technical claim DIRECTLY from the RETRIEVED CONTEXT provided
   in the human message. Do NOT add formulas, theorems, or facts from your training
   data that are absent from those passages. If the context is incomplete, state
   exactly which aspect is missing rather than filling the gap from memory.
   The Intuition, Python Implementation, and Mentor's Note sections may
   draw on broader knowledge — but the Formal Concept section must be
   entirely traceable to the retrieved passages.
"""

_HUMAN_MESSAGE = """\
RETRIEVED CONTEXT:
{context}

STUDENT QUESTION: {question}

Take a deep breath and break this down step-by-step for the student. Provide a 
thorough, academically rigorous, yet accessible answer following the Mentor's Flow.
"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_MESSAGE),
        ("human", _HUMAN_MESSAGE),
    ]
)


# ---------------------------------------------------------------------------
# Format retrieved documents
# ---------------------------------------------------------------------------

def format_docs(docs: list[Document]) -> str:
    """
    Format a list of LangChain Documents into a context string for the prompt.

    If no documents are provided (guardrail filtered the query), returns the string
    "NO_CONTEXT_AVAILABLE", which will cause the chain to activate the mandatory
    refusal message in the system prompt.

    Args
    ----
    docs : list[Document]
        List of documents retrieved by the retriever.

    Returns
    -------
    str
        Either:
        - "NO_CONTEXT_AVAILABLE" (if docs is empty)
        - Formatted context string: "--- Source: {source}, Page {page} ---\n{content}\n"

    Examples
    --------
    >>> from langchain_core.documents import Document
    >>> docs = [
    ...     Document(
    ...         page_content="Edge detection is...",
    ...         metadata={"source": "Gonzalez_Woods_DIP.pdf", "page": 234}
    ...     )
    ... ]
    >>> context = format_docs(docs)
    >>> print(context)
    --- Source: Gonzalez_Woods_DIP.pdf, Page 234 ---
    Edge detection is...
    """
    if not docs:
        return "NO_CONTEXT_AVAILABLE"

    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        content = doc.page_content
        parts.append(f"--- Source: {source}, Page {page} ---\n{content}")

    return "\n\n".join(parts)


def _extract_question(x: str | dict) -> str:
    """Normalize both direct ``str`` calls and LangServe ``{"question": str}`` dicts."""
    if isinstance(x, dict):
        return x.get("question", str(x))
    return str(x)


# ---------------------------------------------------------------------------
# LLM selector
# ---------------------------------------------------------------------------

def get_llm() -> BaseLanguageModel:
    """Return the active LLM backend.

    Gemini 2.0 Flash is used when ``LLM_BACKEND=gemini`` and DeepSeek-R1
    via Ollama is used when ``LLM_BACKEND=ollama``.

    Raises
    ------
    RuntimeError  If required env var is missing.
    ValueError    If LLM_BACKEND is not "gemini" or "ollama".
    """
    backend = os.getenv("LLM_BACKEND", "gemini").lower().strip()

    if backend == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set but LLM_BACKEND=gemini")

        model_name = os.getenv("LLM_MODEL", "gemini-2.0-flash")
        logger.info("Using %s as primary LLM (free tier, demo/dev environment)", model_name)
        return ChatGoogleGenerativeAI(
            model=model_name,
            api_key=api_key,  # langchain-google-genai 4.x expects plain str
            temperature=0.2,
            max_retries=0,    # fail fast on quota errors — no retry storm
        )

    elif backend == "groq":
        from langchain_groq import ChatGroq

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY not set but LLM_BACKEND=groq")

        groq_model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        logger.info(
            "Using Groq (%s) as primary LLM (free tier, no billing required)",
            groq_model,
        )
        # Gate every LLM call to stay under Groq's 6 000 TPM free-tier ceiling.
        # llama-3.1-8b-instant responses average ~600 tokens (2 calls/question).
        # 0.06 req/s × 60 = 3.6 calls/min × 600 tok ≈ 2 160 tok/min (< 6 000).
        try:
            from langchain_core.rate_limiters import InMemoryRateLimiter
            _groq_limiter = InMemoryRateLimiter(
                requests_per_second=0.06,
                check_every_n_seconds=0.1,
                max_bucket_size=1,
            )
            return ChatGroq(
                model=groq_model,
                api_key=groq_api_key,
                temperature=0.2,
                max_tokens=2048,
                max_retries=2,
                rate_limiter=_groq_limiter,
            )
        except (ImportError, TypeError):
            # Older langchain-core without InMemoryRateLimiter or rate_limiter kwarg.
            return ChatGroq(
                model=groq_model,
                api_key=groq_api_key,
                temperature=0.2,
                max_tokens=2048,
                max_retries=2,
            )

    elif backend == "ollama":
        from langchain_community.chat_models import ChatOllama

        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-r1")

        logger.info(
            f"Using DeepSeek-R1 (Ollama) as LLM "
            f"(fully local, production target: {ollama_url})"
        )
        return ChatOllama(
            model=deepseek_model,
            base_url=ollama_url,
            temperature=0.2,
        )

    else:
        raise ValueError(
            f"Unknown LLM_BACKEND '{backend}'. "
            f"Must be 'gemini' or 'ollama'."
        )


# ---------------------------------------------------------------------------
# RAG Chain Factory
# ---------------------------------------------------------------------------

def build_rag_chain() -> Runnable:
    """
    Build and return the complete RAG chain using LCEL.

        This chain follows the architecture:
            Input (question string)
                -> [Guardrail Retriever] filters out-of-domain queries
                -> [Format Docs] creates context string for prompt
                -> [Prompt] injects context and question
                -> [LLM] generates response (Gemini or DeepSeek)
                -> [Output Parser] extracts string from LLM response
                -> Output (answer string)

    The guardrail is critical: if a query is too dissimilar from the knowledge base,
    it returns an empty list. This forces the prompt to respond with the mandatory
    refusal message, preventing LLM hallucination on off-topic questions.

    Returns
    -------
    Runnable
        LCEL chain that accepts a question string and returns an answer string.

    Notes
    -----
    - Chain is configured for LangSmith tracing with run_name="DIP_RAG_Chain".
    - Accepts both a bare ``str`` and a LangServe ``{"question": str}`` dict.
    - Guardrail threshold: 0.25 (L2 distance).  Tune in retriever.py if needed.
    """

    # Fallback retrievers used when ChromaDB schema compat check fails.
    def _passthrough_retriever(query: str) -> list[Document]:
        return get_retriever().invoke(query)

    def _empty_retriever(query: str) -> list[Document]:  # noqa: ARG001
        return []

    try:
        # Threshold 1.2 is calibrated for all-MiniLM-L6-v2 (384-dim) L2 distances:
        #   in-domain DIP content: 0.58–0.81  → passes (< 1.2)
        #   off-topic queries:     1.46–1.54  → blocked (≥ 1.2)
        guardrail_fn = get_guardrail_retriever(threshold=1.2)
        logger.info("Guardrail retriever initialized.")
    except KeyError as exc:
        logger.warning("ChromaDB schema compat issue: %s — passthrough fallback.", exc)
        guardrail_fn = _passthrough_retriever
    except Exception as exc:
        logger.error("Retriever init failed: %s — stub fallback.", exc)
        guardrail_fn = _empty_retriever

    retriever: Runnable = RunnableLambda(guardrail_fn)

    chain = (
        RunnableLambda(_extract_question)
        | RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
        )
        | RAG_PROMPT
        | get_llm()
        | StrOutputParser()
    ).with_config(run_name="DIP_RAG_Chain")

    logger.info("RAG chain built successfully.")
    return chain


# ---------------------------------------------------------------------------
# Session memory store
# ---------------------------------------------------------------------------

#: module-level store  →  {session_id: {"chain": ..., "last_accessed": float}}
MEMORY_STORE: dict[str, dict[str, Any]] = {}
_STORE_LOCK = threading.Lock()
_MAX_SESSIONS = 100


def get_or_create_chain(session_id: str):
    """Return an existing ``ConversationalRetrievalChain`` or create one for *session_id*.

    Each session has its own ``ConversationBufferWindowMemory`` (k=10) so
    conversation history never leaks between students.  When the store exceeds
    ``_MAX_SESSIONS``, the least-recently-accessed session is evicted.

    Args:
        session_id: Opaque string identifier (UUID recommended).

    Returns:
        ``ConversationalRetrievalChain`` configured for multi-turn tutoring.

    Example:
        >>> chain = get_or_create_chain("abc-123")
        >>> result = chain({"question": "What is spatial filtering?"})
        >>> result["answer"]
        '...'
    """
    from langchain_classic.chains import ConversationalRetrievalChain
    from langchain_classic.memory import ConversationBufferWindowMemory

    with _STORE_LOCK:
        if session_id in MEMORY_STORE:
            MEMORY_STORE[session_id]["last_accessed"] = time.time()
            logger.debug("Session cache hit: %s", session_id)
            return MEMORY_STORE[session_id]["chain"]

        # ── Enforce max session cap ──────────────────────────────────────
        if len(MEMORY_STORE) >= _MAX_SESSIONS:
            oldest_sid = min(
                MEMORY_STORE, key=lambda s: MEMORY_STORE[s]["last_accessed"]
            )
            del MEMORY_STORE[oldest_sid]
            logger.info("Evicted LRU session: %s (store was full)", oldest_sid)

        # ── Build per-session memory ─────────────────────────────────────
        memory = ConversationBufferWindowMemory(
            k=10,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

        # Use CONV_PROMPT which includes {chat_history} placeholder
        conv_chain = ConversationalRetrievalChain.from_llm(
            llm=get_llm(),
            retriever=get_retriever(),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": CONV_PROMPT},
            verbose=False,
        )

        MEMORY_STORE[session_id] = {
            "chain": conv_chain,
            "last_accessed": time.time(),
        }
        logger.info("Created new session: %s (active sessions: %d)", session_id, len(MEMORY_STORE))
        return conv_chain


def run_chain(session_id: str, question: str) -> dict:
    """Invoke the conversational chain for *session_id* and return a structured result.

    Args:
        session_id: Session identifier — must match a prior or new session.
        question: The student's question text.

    Returns:
        dict with keys:
            ``answer``     (str)  — LLM response with citations.
            ``session_id`` (str)  — echoed back for client tracking.
            ``sources``    (list) — list of ``{"source": str, "page": int/str,
                                    "page_content": str}`` dicts from retrieved
                                    documents. ``page_content`` is the raw chunk
                                    text; included so the evaluation pipeline
                                    can pass real context strings to RAGAS.

    Example:
        >>> result = run_chain("abc-123", "What is the Sobel operator?")
        >>> print(result["answer"])
        >>> print(result["sources"])
    """
    chain = get_or_create_chain(session_id)
    result = chain.invoke({"question": question})
    source_docs = result.get("source_documents", [])
    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "page_content": doc.page_content,  # included for RAGAS context evaluation
        }
        for doc in source_docs
    ]
    return {
        "answer": result.get("answer", ""),
        "session_id": session_id,
        "sources": sources,
    }


def clear_session(session_id: str) -> bool:
    """Remove *session_id* and its conversation history from the store.

    Args:
        session_id: Session to remove.

    Returns:
        ``True`` if the session existed and was removed; ``False`` otherwise.

    Example:
        >>> clear_session("abc-123")
        True
    """
    with _STORE_LOCK:
        if session_id in MEMORY_STORE:
            del MEMORY_STORE[session_id]
            logger.info("Session cleared: %s", session_id)
            return True
        logger.debug("clear_session: session not found: %s", session_id)
        return False


def cleanup_expired_sessions(ttl_seconds: int = 3600) -> None:
    """Evict sessions that have not been accessed within *ttl_seconds*.

    Schedules itself to run again every 600 seconds via a daemon
    ``threading.Timer`` so no external scheduler is needed.

    Args:
        ttl_seconds: Idle timeout in seconds (default 3600 = 1 hour).
    """
    cutoff = time.time() - ttl_seconds
    with _STORE_LOCK:
        expired = [sid for sid, meta in MEMORY_STORE.items() if meta["last_accessed"] < cutoff]
        for sid in expired:
            del MEMORY_STORE[sid]
    if expired:
        logger.info("Session cleanup: removed %d expired session(s).", len(expired))

    # Re-schedule — daemon=True means the timer won't block server shutdown
    t = threading.Timer(600.0, cleanup_expired_sessions, kwargs={"ttl_seconds": ttl_seconds})
    t.daemon = True
    t.start()


# ---------------------------------------------------------------------------
# Conversational prompt (includes {chat_history})
# ---------------------------------------------------------------------------

_CONV_SYSTEM_MESSAGE = (
    _SYSTEM_MESSAGE
    + "\n\nWhen continuing a conversation, use the chat history below to "
    "understand context and avoid repeating information already covered."
)

_CONV_HUMAN_MESSAGE = """\
CHAT HISTORY:
{chat_history}

RETRIEVED CONTEXT:
{context}

STUDENT QUESTION: {question}

Take a deep breath and break this down step-by-step for the student.
"""

CONV_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _CONV_SYSTEM_MESSAGE),
        ("human", _CONV_HUMAN_MESSAGE),
    ]
)

# Start the background cleanup timer on module import
cleanup_expired_sessions()
logger.info("Session cleanup daemon started (TTL=3600s, interval=600s).")
