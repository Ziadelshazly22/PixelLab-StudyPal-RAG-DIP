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
              Fast inference, generous free tier, good for the demo and development environment
  - Fallback : ChatOllama(model="deepseek-r1", base_url=OLLAMA_BASE_URL)
              Fully local, zero API cost, production target for university server later

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
   "While I'd love to always help,but this question falls is out of focus, Let's avoid distractions and  get back to pixels 
   & matrices!"

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
