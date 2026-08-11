# -*- coding: utf-8 -*-
"""
main.py
-------
FastAPI + LangServe entry point for the Smart Learning Assistant.

Start-up sequence
-----------------
1. Load environment variables from .env
2. Register auxiliary REST routes   (app/api/router.py)
3. Register LangServe chain route   POST /chain/rag/invoke
4. Mount the Gradio UI              /ui  (separate process: python app/ui/interface.py)

Run:
    .venv\\Scripts\\python.exe -m uvicorn main:app --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")  # must run before any LangChain imports

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.api.router import router as api_router
from app.chains.rag_chain import build_rag_chain, run_chain, run_chain_with_doc, clear_session
from app.summarization.summarizer import summarize_document, generate_study_questions

logger = logging.getLogger(__name__)


def _is_quota_error(exc: Exception) -> bool:
    """Heuristic checker for provider quota/rate-limit errors."""
    text = str(exc).lower()
    return (
        "resourceexhausted" in text
        or "quota" in text
        or "rate limit" in text
        or "429" in text
    )


def _is_connection_error(exc: Exception) -> bool:
    """Heuristic checker for provider/network connectivity failures."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        token in text
        for token in (
            "apiconnectionerror",
            "connecterror",
            "connection error",
            "getaddrinfo failed",
            "name or service not known",
            "temporary failure in name resolution",
        )
    )


def _internal_error_detail(prefix: str, exc: Exception, *, limit: int = 240) -> str:
    """Build a short error detail string that preserves the exception type."""
    message = f"{type(exc).__name__}: {exc}"
    if len(message) > limit:
        message = message[: limit - 3] + "..."
    return f"{prefix} ({message})"


# ---------------------------------------------------------------------------
# Lifespan — log LLM backend on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler — runs on server startup and shutdown.

    Startup:
        1. Pre-warm the embedding model and vectorstore so lru_cache is
           populated before the first request arrives.  This eliminates the
           8-15 second cold-load penalty on the first /chat call.
        2. Log which LLM backend is active.
    Shutdown: Log and clean up.
    """
    # Pre-warm embedding model + vectorstore (populates lru_cache once,
    # at startup, so every subsequent request is served from cache).
    try:
        from app.ingestion.pipeline import get_embedding_model, load_vectorstore
        logger.info("⏳ Pre-warming embedding model and vectorstore...")
        get_embedding_model()   # loads all-MiniLM-L6-v2 once
        load_vectorstore()                       # opens ChromaDB once
        logger.info("✅ Embedding model and vectorstore ready.")
    except Exception as _warm_err:  # noqa: BLE001
        logger.warning(f"Warm-up skipped (vectorstore may not exist yet): {_warm_err}")

    llm_backend = os.getenv("LLM_BACKEND", "groq")
    logger.info(f"🚀 Server ready. LLM backend: {llm_backend.upper()}")
    yield
    logger.info("Server shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Smart Learning Assistant",
    description=(
        "RAG-Powered AI Tutor grounded in the Gonzalez & Woods DIP textbook "
        "and verified code documentation.  "
        "Dual-LLM strategy: Groq llama-3.1-8b-instant (demo/dev) + DeepSeek-R1-Distill-Qwen-14B via Ollama (campus). "
        "Powered by LangChain LCEL(LangChain Expression Language ) and ChromaDB."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):  # noqa: ANN001
    """Convert provider quota/rate-limit and connectivity failures into 503s."""
    if _is_quota_error(exc):
        return JSONResponse(
            status_code=503,
            content={
                "detail": (
                    "LLM quota/rate limit reached. Please retry later or switch backend "
                    "to Ollama for local inference."
                )
            },
        )
    if _is_connection_error(exc):
        return JSONResponse(
            status_code=503,
            content={
                "detail": _internal_error_detail(
                    "LLM provider is unreachable",
                    exc,
                )
                + " Try again later or switch LLM_BACKEND=ollama if you have a local Ollama server.",
            },
        )
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )

# ---------------------------------------------------------------------------
# CORS – allow all origins during development; tighten in production
# ---------------------------------------------------------------------------
# TODO: restrict origins before production deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Favicon — served at /favicon.ico and /favicon.svg so every browser tab
# picks up the 🤖 icon regardless of mount path
# ---------------------------------------------------------------------------
_FAVICON = Path(__file__).resolve().parent / "app" / "ui" / "favicon.svg"


@app.get("/favicon.ico", include_in_schema=False)
@app.get("/favicon.svg", include_in_schema=False)
async def favicon() -> FileResponse:
    return FileResponse(_FAVICON, media_type="image/svg+xml")


@app.get("/manifest.json", include_in_schema=False)
async def web_app_manifest() -> JSONResponse:
    """PWA Web App Manifest — prevents 404 log noise from browsers."""
    return JSONResponse({
        "name": "DIP AI Tutor",
        "short_name": "DIP Tutor",
        "description": "Smart Learning Assistant for Digital Image Processing",
        "start_url": "/ui",
        "display": "standalone",
        "background_color": "#ffffff",
        "theme_color": "#1f2937",
        "icons": [],
    })

# ---------------------------------------------------------------------------
# Auxiliary REST routes
# ---------------------------------------------------------------------------
app.include_router(api_router)

# ---------------------------------------------------------------------------
# Stateless RAG chain route  (replaces LangServe — same input/output schema)
# POST /chain/rag/invoke  { "input": "<question>" }  →  { "output": "<answer>" }
# ---------------------------------------------------------------------------
class _ChainRequest(BaseModel):
    input: str


@app.post("/chain/rag/invoke", tags=["rag"])
async def chain_rag_invoke(body: _ChainRequest) -> dict:
    """Stateless RAG chain — same interface as the old LangServe endpoint."""
    loop = asyncio.get_running_loop()
    try:
        chain = build_rag_chain()
        result = await loop.run_in_executor(None, chain.invoke, body.input)
    except Exception as exc:  # noqa: BLE001
        if _is_connection_error(exc):
            raise HTTPException(
                status_code=503,
                detail=(
                    _internal_error_detail(
                        "LLM provider is unreachable during chain invocation",
                        exc,
                    )
                    + " Check network/DNS or switch LLM_BACKEND=ollama if available."
                ),
            ) from exc
        if _is_quota_error(exc):
            raise HTTPException(
                status_code=503,
                detail="LLM quota/rate limit reached. Please retry later.",
            ) from exc
        logger.error("/chain/rag/invoke failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=_internal_error_detail("Chain invocation failed", exc),
        ) from exc
    return {"output": result}


# ---------------------------------------------------------------------------
# Health  (top-level – used by Docker / load-balancer probes)
# ---------------------------------------------------------------------------
@app.get("/health", tags=["ops"])
async def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
@app.get("/", tags=["root"])
async def root() -> dict:
    return {
        "message": "Smart Learning Assistant is running 🚀",
        "docs": "/docs",
        "ui": "/ui",
    }


# ---------------------------------------------------------------------------
# Chat  — stateful conversational RAG (per-session memory)
# ---------------------------------------------------------------------------

class _ChatRequest(BaseModel):
    question: str
    session_id: str
    doc_context: str = ""    # extracted text from a session-attached file (never stored in KB)
    doc_filename: str = ""  # original filename shown in source citations


class _SummarizeRequest(BaseModel):
    source: str
    include_questions: bool = True
    n_questions: int = 5


@app.post("/chat", tags=["chat"])
async def chat(body: _ChatRequest) -> dict:
    """Conversational RAG with per-session ``ConversationalRetrievalChain``.

    Returns ``{"answer": str, "session_id": str, "sources": list}``.
    Times out after 90 s and returns HTTP 503 so the client gets a clear error
    instead of hanging indefinitely (httpx default per-call timeout is 600 s).
    """
    loop = asyncio.get_running_loop()
    try:
        if body.doc_context and body.doc_context.strip():
            # Session document attached — bypass the condense-question step so
            # the attached content is never stripped before the LLM sees it.
            _chain_fn = lambda: run_chain_with_doc(  # noqa: E731
                body.session_id,
                body.question,
                body.doc_context,
                doc_filename=body.doc_filename or "Attached Session Document",
            )
        else:
            _chain_fn = lambda: run_chain(body.session_id, body.question)  # noqa: E731

        result = await asyncio.wait_for(
            loop.run_in_executor(None, _chain_fn),
            timeout=90.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Request timed out after 90 s. "
                "The LLM may be rate-limited — wait 30 s and try again, "
                "or set LLM_BACKEND=ollama for fully local inference."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        if _is_connection_error(exc):
            raise HTTPException(
                status_code=503,
                detail=(
                    _internal_error_detail(
                        "LLM provider is unreachable during chat",
                        exc,
                    )
                    + " Check network/DNS or switch LLM_BACKEND=ollama if available."
                ),
            ) from exc
        if _is_quota_error(exc):
            raise HTTPException(
                status_code=503,
                detail=(
                    "LLM quota/rate limit reached. Please retry later or switch backend "
                    "to Ollama."
                ),
            ) from exc
        logger.error("/chat failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=_internal_error_detail(
                "Chat request failed due to an internal processing error",
                exc,
            ),
        ) from exc
    return result


@app.delete("/chat/{session_id}", tags=["chat"])
async def clear_chat(session_id: str) -> dict:
    """Delete the conversation memory buffer for *session_id*."""
    removed = clear_session(session_id)
    return {"status": "cleared" if removed else "not_found", "session_id": session_id}


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------

@app.post("/summarize", tags=["summarization"])
async def summarize(body: _SummarizeRequest) -> dict:
    """Map-reduce document summarisation + optional study-question generation.

    Returns ``{"summary": str, "study_questions": list[str], "source": str}``.
    Raises HTTP 503 if the operation exceeds 120 s.
    """
    loop = asyncio.get_running_loop()
    try:
        summary_future = loop.run_in_executor(None, summarize_document, body.source)
        summary: str = await asyncio.wait_for(summary_future, timeout=120.0)

        questions: list[str] = []
        if body.include_questions:
            questions_future = loop.run_in_executor(
                None, generate_study_questions, body.source, body.n_questions
            )
            questions = await asyncio.wait_for(questions_future, timeout=60.0)

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Summarisation timed out. "
                "The document may be too large — try again or increase the timeout."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        if _is_connection_error(exc):
            raise HTTPException(
                status_code=503,
                detail=(
                    _internal_error_detail(
                        "LLM provider is unreachable during summarisation",
                        exc,
                    )
                    + " Check network/DNS or switch LLM_BACKEND=ollama if available."
                ),
            ) from exc
        if _is_quota_error(exc):
            raise HTTPException(
                status_code=503,
                detail=(
                    "LLM quota/rate limit reached during summarisation. "
                    "Please retry later or switch backend to Ollama."
                ),
            ) from exc
        logger.error("/summarize failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Summarisation failed due to an internal processing error.",
        ) from exc

    return {"summary": summary, "study_questions": questions, "source": body.source}


# ---------------------------------------------------------------------------
# Gradio UI  (mounted at /ui)
# UI runs as a separate process: python app/ui/interface.py
# ---------------------------------------------------------------------------
try:
    import gradio as gr
    from app.ui.interface import build_interface

    gradio_app = build_interface()
    app = gr.mount_gradio_app(app, gradio_app, path="/ui")
except Exception as _ui_err:  # noqa: BLE001
    import warnings
    warnings.warn(f"Gradio UI could not be mounted: {_ui_err}", stacklevel=1)


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

