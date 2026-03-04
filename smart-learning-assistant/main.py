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
    .venv\\Scripts\\python.exe -m uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")  # must run before any LangChain/Google imports

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

import asyncio
import concurrent.futures

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from pydantic import BaseModel

from app.api.router import router as api_router
from app.chains.rag_chain import build_rag_chain, run_chain, clear_session
from app.summarization.summarizer import summarize_document, generate_study_questions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — log LLM backend on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler — runs on server startup and shutdown.

    Startup: Log which LLM backend is active (Gemini or Ollama).
    Shutdown: (reserved for cleanup if needed)
    """
    llm_backend = os.getenv("LLM_BACKEND", "gemini")
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
        "Dual-LLM strategy: Gemini 2.0 Flash (primary) + DeepSeek-R1 (fallback). "
        "Powered by LangChain, LangServe, and ChromaDB."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
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
# Auxiliary REST routes
# ---------------------------------------------------------------------------
app.include_router(api_router)

# ---------------------------------------------------------------------------
# LangServe RAG chain route
# ---------------------------------------------------------------------------
add_routes(app, build_rag_chain(), path="/chain/rag")


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


class _SummarizeRequest(BaseModel):
    source: str
    include_questions: bool = True
    n_questions: int = 5


@app.post("/chat", tags=["chat"])
async def chat(body: _ChatRequest) -> dict:
    """Conversational RAG with per-session ``ConversationalRetrievalChain``.

    Returns ``{"answer": str, "session_id": str, "sources": list}``.
    """
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool, run_chain, body.session_id, body.question
        )
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
    loop = asyncio.get_event_loop()
    try:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = loop.run_in_executor(pool, summarize_document, body.source)
            summary: str = await asyncio.wait_for(future, timeout=120.0)

        questions: list[str] = []
        if body.include_questions:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = loop.run_in_executor(
                    pool, generate_study_questions, body.source, body.n_questions
                )
                questions = await asyncio.wait_for(future, timeout=60.0)

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Summarisation timed out. "
                "The document may be too large — try again or increase the timeout."
            ),
        )

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

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

