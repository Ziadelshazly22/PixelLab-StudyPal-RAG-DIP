"""
app/chains/rag_chain.py
-----------------------
Builds the core Retrieval-Augmented Generation (RAG) chain using
LangChain LCEL (LangChain Expression Language).

The RAG chain follows this architecture:
  Input (question)
    ↓
  [Guardrail Retriever] → empty list OR top-k documents
    ↓
  [Format Docs] → "NO_CONTEXT_AVAILABLE" OR formatted context string
    ↓
  [Prompt] → system message + context + question
    ↓
  [LLM] → Gemini 2.0 Flash (primary) OR ChatOllama (fallback DeepSeek-R1)
    ↓
  [Output Parser] → str
    ↓
  Output (answer)

Dual-LLM strategy:
  - Primary  : ChatGoogleGenerativeAI(model="gemini-2.0-flash")
              Fast inference, generous free tier, good for demos
  - Fallback : ChatOllama(model="deepseek-r1", base_url=OLLAMA_BASE_URL)
              Fully local, zero API cost, production target for university server

Switching is automatic: set LLM_BACKEND="ollama" in .env to use DeepSeek.

Non-functional requirements:
  - End-to-end response time: < 5 seconds
  - Hallucination prevention: guardrail filters out-of-domain queries
  - Citation requirement: every factual claim must cite source [Source: filename, Page N]
"""

from __future__ import annotations

import logging
import os
from typing import Callable

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.base_language import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    Runnable,
)

from app.retrieval.retriever import get_guardrail_retriever

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LCEL Prompt Template
# ---------------------------------------------------------------------------

_SYSTEM_MESSAGE = """\
You are an expert AI Tutor specialising exclusively in Digital Image Processing (DIP),
computer vision, and their implementation using Python libraries (OpenCV, NumPy, SciPy,
Matplotlib, Pillow).

Your knowledge base consists of:
  • Gonzalez & Woods — Digital Image Processing, 4th Edition
  • OpenCV, NumPy, SciPy, Matplotlib, and Pillow official documentation

MANDATORY RULES — follow every rule on every response:

1. SCOPE: Only answer questions about DIP, computer vision, image processing algorithms,
   or the listed Python libraries. For any off-topic question, respond exactly:
   "This question falls outside my knowledge base (Digital Image Processing and related 
   libraries). Please consult your course materials or a general search engine."

2. CITATIONS: Every factual claim must be followed by [Source: <filename>, Page <N>].
   Never make uncited assertions.

3. HONESTY: If the provided context does not contain enough information to answer 
   confidently, say: "The provided context does not fully cover this. I recommend 
   consulting Chapter <X> of Gonzalez & Woods directly."

4. MATH: Format all equations using LaTeX inline notation: $equation$

5. CODE: Wrap all code examples in ```python blocks with a one-line comment 
   explaining what the code demonstrates.

6. STRUCTURE: For technical questions, use this structure:
   — Concept definition (1-2 sentences)
   — Mathematical formulation (if applicable)
   — Practical implementation (code if relevant)
   — Key insight or exam-relevant point
"""

_HUMAN_MESSAGE = """\
RETRIEVED CONTEXT:
{context}

STUDENT QUESTION: {question}

Provide a thorough, academically rigorous answer following all mandatory rules above.
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


# ---------------------------------------------------------------------------
# LLM selector
# ---------------------------------------------------------------------------

def get_llm() -> BaseLanguageModel:
    """
    Select and return the appropriate LLM based on LLM_BACKEND environment variable.

    The primary LLM (Gemini 2.0 Flash) is excellent for demos and development due to
    its generous free tier. However, for production deployment on the university server,
    switching to DeepSeek-R1 (via Ollama) provides zero API cost and full local control.

    Args
    ----
    None

    Returns
    -------
    BaseChatModel
        Either ChatGoogleGenerativeAI (if LLM_BACKEND="gemini") or
        ChatOllama (if LLM_BACKEND="ollama").

    Raises
    ------
    ValueError
        If LLM_BACKEND is set to an unsupported value.
    RuntimeError
        If the required environment variables (GOOGLE_API_KEY for Gemini,
        OLLAMA_BASE_URL for Ollama) are not set.

    Examples
    --------
    >>> # Use default (Gemini)
    >>> llm = get_llm()
    >>> response = llm.invoke("What is histogram equalization?")
    >>> 
    >>> # Switch to Ollama by setting environment variable
    >>> import os
    >>> os.environ["LLM_BACKEND"] = "ollama"
    >>> llm = get_llm()  # now ChatOllama with DeepSeek-R1
    """
    backend = os.getenv("LLM_BACKEND", "gemini").lower().strip()

    if backend == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set but LLM_BACKEND=gemini")

        logger.info("Using Gemini 2.0 Flash as primary LLM (free tier, demo/dev environment)")
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=api_key,
            temperature=0.2,
            convert_system_message_to_human=True,
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
        ↓
      [Guardrail Retriever] → filters out-of-domain queries
        ↓
      [Format Docs] → creates context string for prompt
        ↓
      [Prompt] → injects context and question
        ↓
      [LLM] → generates response (Gemini or DeepSeek)
        ↓
      [Output Parser] → extracts string from LLM response
        ↓
      Output (answer string)

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
    - Input and output are both simple strings (not dicts).
    - Latency target: < 5 seconds end-to-end.
    - For testing, the retriever always provides context; guardrail threshold
      can be adjusted in app/retrieval/retriever.py.

    Examples
    --------
    >>> chain = build_rag_chain()
    >>> answer = chain.invoke("What is histogram equalization?")
    >>> print(answer)
    >>> 
    >>> # With LangSmith tracing
    >>> from langsmith import trace
    >>> with trace():
    ...     answer = chain.invoke("What is Fourier transform in image processing?")
    """
    # Initialise guardrail retriever (will reject out-of-domain queries)
    guardrail_fn = get_guardrail_retriever(threshold=0.25)

    # Wrap guardrail function in RunnableLambda to make it compatible with LCEL
    retriever: Runnable = RunnableLambda(guardrail_fn)

    # Build the chain using RunnableParallel
    # This passes both context and question to the prompt template
    chain = (
        RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
        )
        | RAG_PROMPT
        | get_llm()
        | StrOutputParser()
    )

    # Add LangSmith tracing metadata for debugging and monitoring
    chain = chain.with_config(run_name="DIP_RAG_Chain")

    logger.info("RAG chain built successfully. Ready to invoke.")
    return chain
