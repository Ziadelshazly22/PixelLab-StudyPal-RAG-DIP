# -*- coding: utf-8 -*-
"""
app/api/router.py
-----------------
Auxiliary REST endpoints for the Smart Learning Assistant.

Routes (no prefix — registered directly on the FastAPI app):
    POST /ingest                — Upload a PDF and ingest it into ChromaDB
    GET  /status                — Knowledge-base and server health summary
    POST /settings/llm_backend  — Switch LLM backend at runtime (gemini | ollama)
    GET  /api/health            — Legacy liveness probe
    GET  /api/info              — Legacy service metadata

Chain-specific routes are registered in main.py via LangServe add_routes.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()  # No prefix — routes live at root level

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_UPLOADS_DIR = Path("./data/raw/uploads")
_LARGE_FILE_THRESHOLD = 5 * 1024 * 1024  # 5 MB


# ---------------------------------------------------------------------------
# Internal: single-file ingestion helper
# ---------------------------------------------------------------------------

def _run_single_file_ingestion(file_path: Path) -> dict:
    """Run extract → chunk → embed for one PDF.  Returns stats dict."""
    from app.ingestion.pipeline import (
        chunk_documents,
        embed_and_store,
        extract_text_from_pdf,
    )

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    filename = file_path.name

    logger.info("Ingestion started: %s", filename)

    pages = extract_text_from_pdf(str(file_path), category="uploads")
    if not pages:
        logger.warning("No extractable pages in %s", filename)
        return {"chunks_added": 0, "pages_processed": 0}

    chunks = chunk_documents(pages)
    embed_and_store(chunks, persist_dir=persist_dir)

    logger.info(
        "Ingestion complete: %s — %d pages, %d chunks",
        filename, len(pages), len(chunks),
    )
    return {"chunks_added": len(chunks), "pages_processed": len(pages)}


def _background_ingest(file_path: Path) -> None:
    """BackgroundTasks wrapper — swallows exceptions so the worker stays alive."""
    try:
        _run_single_file_ingestion(file_path)
    except Exception as exc:
        logger.error(
            "Background ingestion failed for %s: %s",
            file_path.name, exc, exc_info=True,
        )


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

@router.post("/ingest", tags=["ingestion"], summary="Upload a PDF and ingest it")
async def ingest_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
) -> dict:
    """
    Accept a single PDF upload, persist it to ``data/raw/uploads/``, and run
    the ingestion pipeline (extract → chunk → embed → ChromaDB).

    - **≤ 5 MB** — processed inline; response includes final chunk count.
    - **> 5 MB** — queued as a background task; returns immediately with
      ``"status": "processing"``.  Poll ``GET /status`` for the updated
      chunk count.

    Raises
    ------
    400
        If the uploaded file is not a ``.pdf``.
    500
        If the file cannot be saved or ingestion fails (inline path only).
    """
    # ── Validate file type ──────────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are accepted. Got: '{file.filename}'",
        )

    # ── Ensure upload directory exists ──────────────────────────────────
    _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Read and save file ──────────────────────────────────────────────
    file_bytes = await file.read()
    file_size = len(file_bytes)
    save_path = _UPLOADS_DIR / file.filename

    try:
        save_path.write_bytes(file_bytes)
    except OSError as exc:
        logger.error("Could not save uploaded file %s: %s", file.filename, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Could not save file: {exc}",
        ) from exc

    logger.info(
        "Upload received: %s (%.2f MB)", file.filename, file_size / (1024 * 1024)
    )

    # ── Large file → background task ────────────────────────────────────
    if file_size > _LARGE_FILE_THRESHOLD:
        background_tasks.add_task(_background_ingest, save_path)
        logger.info("Queued background ingestion for: %s", file.filename)
        return {
            "status": "processing",
            "message": (
                "Ingestion started in background. "
                "Poll GET /status for the updated chunk count."
            ),
            "filename": file.filename,
        }

    # ── Small file → inline ingestion ───────────────────────────────────
    try:
        result = _run_single_file_ingestion(save_path)
    except Exception as exc:
        logger.error("Inline ingestion failed for %s: %s", file.filename, exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {exc}",
        ) from exc

    return {
        "status": "success",
        "filename": file.filename,
        "chunks_added": result["chunks_added"],
        "pages_processed": result["pages_processed"],
    }


# ---------------------------------------------------------------------------
# GET /status
# ---------------------------------------------------------------------------

@router.get("/status", tags=["ops"], summary="Knowledge-base and server status")
async def get_status() -> dict:
    """
    Returns a snapshot of the system state:

    - ``llm_backend``    — active LLM (from ``LLM_BACKEND`` env var)
    - ``embedding_model``— active embedding model
    - ``total_chunks``   — total vectors stored in ChromaDB (−1 on error)
    - ``collection``     — ChromaDB collection name
    - ``server_time``    — current UTC timestamp (ISO 8601)
    """
    llm_backend = os.getenv("LLM_BACKEND", "gemini")
    embedding_model = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    collection_name = "dip_knowledge_base"

    total_chunks: int = -1
    sources: list[str] = []

    # Use load_vectorstore() — it auto-patches the SQLite config_json_str and
    # selects the correct embedding model from the stored dimension.
    try:
        from app.ingestion.pipeline import load_vectorstore
        vs = load_vectorstore(persist_dir=persist_dir)
        total_chunks = vs._collection.count()

        # Source extraction via metadata get (no embedding needed)
        if total_chunks > 0:
            raw = vs._collection.get(include=["metadatas"])
            metas = raw.get("metadatas", []) if isinstance(raw, dict) else []
            sources = sorted({
                m.get("source")
                for m in metas
                if isinstance(m, dict) and m.get("source")
            })
    except FileNotFoundError:
        total_chunks = 0
        logger.warning("/status: ChromaDB not found — run the ingestion pipeline first.")
    except Exception as exc:
        logger.warning("/status: Could not read chunk count: %s", exc)

    return {
        "llm_backend": llm_backend,
        "embedding_model": embedding_model,
        "total_chunks": total_chunks,
        "collection": collection_name,
        "sources": sources,
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# POST /settings/llm_backend
# ---------------------------------------------------------------------------

class _LLMBackendRequest(BaseModel):
    backend: str


@router.post(
    "/settings/llm_backend",
    tags=["settings"],
    summary="Switch LLM backend at runtime",
)
async def set_llm_backend(body: _LLMBackendRequest) -> dict:
    """
    Update the active LLM backend by setting ``os.environ["LLM_BACKEND"]``.

    Accepted values: ``"gemini"`` | ``"ollama"``

    .. note::
        The LangServe chain mounted at ``/chain/rag`` is initialised **once**
        at server startup.  Runtime environment changes will not affect the
        already-instantiated chain object.  **A server restart is required**
        to apply the new backend to existing chain instances.

    Raises
    ------
    400
        If ``backend`` is not ``"gemini"`` or ``"ollama"``.
    """
    _VALID = {"gemini", "ollama"}
    backend = body.backend.strip().lower()

    if backend not in _VALID:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid backend '{backend}'. Must be one of: {sorted(_VALID)}",
        )

    os.environ["LLM_BACKEND"] = backend
    logger.info("LLM_BACKEND updated to: %s", backend)

    return {
        "status": "updated",
        "backend": backend,
        "note": "Restart server to apply to existing chain instances",
    }


# ---------------------------------------------------------------------------
# Legacy helper endpoints (backward compatibility)
# ---------------------------------------------------------------------------

@router.get("/api/health", tags=["ops"], summary="Health check (legacy)")
async def health() -> dict:
    """Liveness probe — returns 200 when the service is running."""
    return {"status": "ok"}


@router.get("/api/info", tags=["ops"], summary="Service metadata (legacy)")
async def info() -> dict:
    """Returns version and model information."""
    return {
        "service": "Smart Learning Assistant",
        "version": "0.1.0",
        "models": ["gemini-2.0-flash", "deepseek-r1"],
    }
