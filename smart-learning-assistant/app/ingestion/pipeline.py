"""
app/ingestion/pipeline.py
-------------------------
Document ingestion pipeline for the Smart Learning Assistant.

Purpose
-------
Loads academic PDFs from ``data/raw/``, extracts text page-by-page,
chunks each page into semantically coherent pieces, embeds every chunk,
and persists the resulting vectors to a local ChromaDB collection
(``dip_knowledge_base``).

Inputs
------
PDF files organised under ``data/raw/`` in three category sub-folders::

    data/raw/
    ├── 1_textbooks/        → category = "textbook"
    ├── 2_core_vision/      → category = "core_vision"
    └── 3_python_utilities/ → category = "utilities"

Outputs
-------
Persistent ChromaDB collection at ``CHROMA_PERSIST_DIR``
(default: ``./data/chroma_db``).  Each stored chunk carries metadata::

    {
        "source":      "filename.pdf",
        "page":        <int>,
        "category":    "textbook" | "core_vision" | "utilities",
        "file_path":   "/absolute/path/to/file.pdf",
        "chunk_index": <int>,
    }

Usage
-----
    # From Python
    from app.ingestion.pipeline import run_ingestion_pipeline
    stats = run_ingestion_pipeline()

    # From CLI (see scripts/run_ingestion.py)
    python scripts/run_ingestion.py
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_COLLECTION_NAME = "dip_knowledge_base"

_SUBDIR_TO_CATEGORY: dict[str, str] = {
    "1_textbooks": "textbook",
    "2_core_vision": "core_vision",
    "3_python_utilities": "utilities",
}


# ---------------------------------------------------------------------------
# FUNCTION 1 — Text extraction
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str, category: str) -> list[Document]:
    """
    Extract text from a PDF file, one :class:`~langchain_core.documents.Document`
    per page.

    Args:
        pdf_path: Absolute path to the PDF file.
        category: One of ``"textbook"``, ``"core_vision"``, ``"utilities"`` —
                  stored in chunk metadata for downstream filtering.

    Returns:
        List of LangChain Document objects with metadata::

            {"source": filename, "page": int, "category": str, "file_path": str}

        Returns an **empty list** on failure — never raises.
    """
    import fitz  # PyMuPDF
    import pdfplumber

    filename = Path(pdf_path).name
    documents: list[Document] = []

    try:
        # ── Primary extraction: PyMuPDF ──────────────────────────────────
        pdf = fitz.open(pdf_path)
        total_text = ""

        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text("text")

            if len(text.strip()) < 50:
                # Image-only / math-diagram page — skip silently
                logger.debug(
                    "Skipping page %d of %s (< 50 chars, likely image-only)",
                    page_num + 1,
                    filename,
                )
                continue

            total_text += text
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": page_num + 1,
                        "category": category,
                        "file_path": pdf_path,
                    },
                )
            )

        pdf.close()

        # ── Fallback: pdfplumber if total text is suspiciously small ──────
        if total_text.strip() and len(total_text.strip()) < 500:
            logger.warning(
                "PyMuPDF yielded < 500 chars for %s — retrying with pdfplumber.",
                filename,
            )
            documents = []
            with pdfplumber.open(pdf_path) as pdf_pl:
                for page_num, page in enumerate(pdf_pl.pages):
                    text = page.extract_text() or ""
                    if len(text.strip()) < 50:
                        logger.debug(
                            "pdfplumber: skipping page %d of %s (< 50 chars)",
                            page_num + 1,
                            filename,
                        )
                        continue
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": filename,
                                "page": page_num + 1,
                                "category": category,
                                "file_path": pdf_path,
                            },
                        )
                    )

        logger.info("Extracted %d pages from %s", len(documents), filename)

    except Exception:
        logger.error("Failed to extract text from %s", pdf_path, exc_info=True)
        return []

    return documents


# ---------------------------------------------------------------------------
# FUNCTION 2 — Chunking
# ---------------------------------------------------------------------------
def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into overlapping chunks, preserving all metadata.

    Args:
        documents: Output from :func:`extract_text_from_pdf`.

    Returns:
        List of chunk Documents with all original metadata plus
        ``{"chunk_index": int}``.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    raw_chunks = splitter.split_documents(documents)

    chunks: list[Document] = []
    for idx, chunk in enumerate(raw_chunks):
        chunk.metadata["chunk_index"] = idx
        chunks.append(chunk)

    logger.info(
        "Created %d chunks from %d pages", len(chunks), len(documents)
    )
    return chunks


# ---------------------------------------------------------------------------
# FUNCTION 3 — Embeddings
# ---------------------------------------------------------------------------
def get_embeddings(use_google: bool = True):
    """
    Return the embedding model.

    Strategy
    --------
    - Primary  : :class:`~langchain_google_genai.GoogleGenerativeAIEmbeddings`
                 (``EMBEDDING_MODEL`` from ``.env``, default ``models/text-embedding-004``)
    - Fallback : :class:`~langchain_community.embeddings.HuggingFaceEmbeddings`
                 (``all-MiniLM-L6-v2``, fully local, no API key required)

    Args:
        use_google: Set ``False`` to skip straight to the local fallback.
    """
    if use_google:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            model = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
            logger.info("Using Google embeddings: %s", model)
            return GoogleGenerativeAIEmbeddings(model=model)
        except Exception:
            logger.warning(
                "Google embeddings failed, falling back to SentenceTransformers",
                exc_info=True,
            )

    from langchain_community.embeddings import HuggingFaceEmbeddings

    logger.info("Using SentenceTransformers: all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# FUNCTION 4 — Already-processed sources
# ---------------------------------------------------------------------------
def get_processed_sources(chroma_client: Any) -> set[str]:
    """
    Return the set of source filenames already stored in the Chroma collection.

    Used for **incremental ingestion** — avoids re-processing PDFs that were
    previously ingested in an earlier run.

    Args:
        chroma_client: A ``chromadb.PersistentClient`` instance.

    Returns:
        Set of filenames (e.g. ``{"Gonzalez_Woods_4th.pdf", "numpy-user.pdf"}``).
        Returns an empty set if the collection does not yet exist.
    """
    try:
        collection = chroma_client.get_collection(_COLLECTION_NAME)
        results = collection.get(include=["metadatas"])
        return {
            meta["source"]
            for meta in results.get("metadatas", [])
            if meta and "source" in meta
        }
    except Exception:
        logger.debug("Collection not found — treating as empty.", exc_info=False)
        return set()


# ---------------------------------------------------------------------------
# FUNCTION 5 — Embed & store
# ---------------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _add_with_retry(vector_store, batch: list[Document]) -> None:
    """Add one batch to Chroma, with automatic tenacity retries."""
    vector_store.add_documents(batch)


def embed_and_store(
    chunks: list[Document],
    embeddings,
    persist_dir: str,
    batch_size: int = 50,
) -> int:
    """
    Embed *chunks* and persist them to the ChromaDB collection in batches.

    Args:
        chunks:      Output from :func:`chunk_documents`.
        embeddings:  Embedding model from :func:`get_embeddings`.
        persist_dir: Path to the ``chroma_db`` directory.
        batch_size:  Chunks per embedding API call (default 50).
                     A 1-second sleep between batches keeps throughput
                     within Google's ~1 500 req/min rate limit.

    Returns:
        Number of chunks successfully stored.
    """
    from langchain_chroma import Chroma
    from tqdm import tqdm

    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=_COLLECTION_NAME,
    )

    stored = 0
    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]

    for batch in tqdm(batches, desc="Embedding batches", unit="batch"):
        try:
            _add_with_retry(vector_store, batch)
            stored += len(batch)
            time.sleep(1)  # respect Google embedding rate limit
        except Exception:
            logger.error(
                "Failed to store batch after 3 retries — skipping %d chunks.",
                len(batch),
                exc_info=True,
            )

    total = vector_store._collection.count()
    logger.info("Stored %d chunks; total collection size: %d", stored, total)
    return stored


# ---------------------------------------------------------------------------
# FUNCTION 6 — Main orchestrator
# ---------------------------------------------------------------------------
def run_ingestion_pipeline(
    raw_dir: str = "./data/raw",
    persist_dir: str | None = None,
) -> dict:
    """
    Full ingestion pipeline: scan → extract → chunk → embed → persist.

    Skips PDFs whose filenames are already present in the collection
    (incremental / resumable runs).

    Args:
        raw_dir:     Root directory containing the three category sub-folders.
        persist_dir: ChromaDB persistence directory.
                     Defaults to ``$CHROMA_PERSIST_DIR`` or ``./data/chroma_db``.

    Returns:
        Statistics dictionary::

            {
                "processed_files": int,
                "skipped_files":   int,
                "total_chunks":    int,
                "errors":          list[str],
            }
    """
    import chromadb

    if persist_dir is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "processed_files": 0,
        "skipped_files": 0,
        "total_chunks": 0,
        "errors": [],
    }

    # Initialise ChromaDB client and embeddings once for the whole run
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    processed_sources = get_processed_sources(chroma_client)
    embeddings = get_embeddings(use_google=True)

    raw_path = Path(raw_dir)

    for subdir_name, category in _SUBDIR_TO_CATEGORY.items():
        subdir = raw_path / subdir_name
        if not subdir.exists():
            logger.warning("Sub-directory not found, skipping: %s", subdir)
            continue

        pdf_files = sorted(subdir.glob("**/*.pdf"))
        logger.info(
            "Scanning %s (%d PDF(s) found) → category=%s",
            subdir_name,
            len(pdf_files),
            category,
        )

        for pdf_path in pdf_files:
            filename = pdf_path.name

            if filename in processed_sources:
                logger.info("Skipping already-processed: %s", filename)
                stats["skipped_files"] += 1
                continue

            try:
                docs = extract_text_from_pdf(str(pdf_path), category)
                if not docs:
                    stats["errors"].append(f"No text extracted from {filename}")
                    continue

                chunks = chunk_documents(docs)
                stored = embed_and_store(chunks, embeddings, persist_dir)
                stats["processed_files"] += 1
                stats["total_chunks"] += stored

            except Exception as exc:
                msg = f"{filename}: {exc}"
                logger.error("Pipeline error — %s", msg, exc_info=True)
                stats["errors"].append(msg)

    logger.info(
        "Ingestion complete: %d files processed, %d skipped, %d total chunks",
        stats["processed_files"],
        stats["skipped_files"],
        stats["total_chunks"],
    )
    return stats


# ---------------------------------------------------------------------------
# FUNCTION 7 — Load vector store (for retriever)
# ---------------------------------------------------------------------------
def load_vectorstore(persist_dir: str | None = None):
    """
    Return the existing Chroma vector store for use by the retriever.

    Called by :mod:`app.retrieval.retriever` at query time — does **not**
    re-ingest any documents.

    Args:
        persist_dir: Path to the persisted ChromaDB directory.
                     Defaults to ``$CHROMA_PERSIST_DIR`` or ``./data/chroma_db``.

    Returns:
        :class:`~langchain_chroma.Chroma` instance ready for similarity search.
    """
    from langchain_chroma import Chroma

    if persist_dir is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

    embeddings = get_embeddings(use_google=True)
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=_COLLECTION_NAME,
    )


# ---------------------------------------------------------------------------
# CLI entry-point (direct execution)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    run_ingestion_pipeline()
