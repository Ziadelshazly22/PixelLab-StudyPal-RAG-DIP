"""
main.py
-------
FastAPI + LangServe entry point for the Smart Learning Assistant.

Start-up sequence
-----------------
1. Load environment variables from .env
2. Register auxiliary REST routes  (app/api/router.py)
3. Register LangServe chain routes (/rag, /summarize)
4. Mount the Gradio UI at /ui

Run:
    python main.py
    # or
    uvicorn main:app --reload
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()  # must run before any LangChain/Google imports

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import router as api_router

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
)

# ---------------------------------------------------------------------------
# CORS – allow all origins during development; tighten in production
# ---------------------------------------------------------------------------
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
# LangServe chain routes
# Uncomment each block once the underlying chain/retriever is initialised.
# ---------------------------------------------------------------------------
# from langserve import add_routes
# from app.chains.rag_chain import build_rag_chain
# from app.retrieval.retriever import build_retriever
#
# retriever = build_retriever()
# rag_chain = build_rag_chain(retriever=retriever)
# add_routes(app, rag_chain, path="/rag")

# ---------------------------------------------------------------------------
# Gradio UI  (mounted at /ui)
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

