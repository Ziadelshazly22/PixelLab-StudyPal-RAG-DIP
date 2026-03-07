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

Dual-environment support
------------------------
All paths are configurable via environment variables — no hardcoded paths.
This file runs identically in Google Colab (heavy ingestion) and on a
local / university server (CLI), with only ``CHROMA_PERSIST_DIR`` and
``GOOGLE_API_KEY`` differing between environments.
"""

from __future__ import annotations

import functools
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
# SECTION 1 — PDF Text Extraction
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str, category: str) -> list[dict]:
    """Extract text from a PDF file, returning one dict per usable page.

    Opens the file with PyMuPDF (fitz) first.  If the entire document yields
    fewer than 100 characters (scanned / image-only PDF), the function
    transparently falls back to ``pdfplumber`` for layout-aware extraction.

    Pages with fewer than 50 characters after stripping (image-only,
    diagram-only, or blank pages) are silently skipped and counted — they
    do **not** raise an error.

    The internal key ``_image_only_skipped`` is injected into the *first*
    page's metadata so the caller can harvest aggregate skip counts without
    changing the public shape of each page dict.

    Args:
        pdf_path: Absolute path to the PDF file.
        category: One of ``"textbook"``, ``"core_vision"``, ``"utilities"``.
            Stored in every chunk's metadata for downstream filtering.

    Returns:
        List of page dicts, one per extractable page::

            [
                {
                    "text": str,
                    "metadata": {
                        "source":    "filename.pdf",
                        "page":      1,          # 1-based
                        "category":  "textbook",
                        "file_path": "/abs/path/file.pdf",
                    },
                },
                ...
            ]

        Returns an **empty list** on any unrecoverable error — never raises.

    Raises:
        Nothing.  ``fitz.FileDataError`` and all other exceptions are caught,
        logged, and result in an empty return value.

    Example:
        >>> pages = extract_text_from_pdf("data/raw/1_textbooks/dip.pdf", "textbook")
        >>> print(pages[0]["metadata"])
        {'source': 'dip.pdf', 'page': 1, 'category': 'textbook', 'file_path': '...'}
    """
    import fitz  # PyMuPDF
    import pdfplumber

    filename = Path(pdf_path).name
    pages: list[dict] = []
    image_only_skipped = 0

    try:
        # ── Primary extraction: PyMuPDF ──────────────────────────────────
        pdf = fitz.open(pdf_path)
        total_text = ""

        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text("text")

            if len(text.strip()) < 50:
                logger.debug(
                    "Skipping page %d of %s (< 50 chars, likely image-only).",
                    page_num + 1,
                    filename,
                )
                image_only_skipped += 1
                continue

            total_text += text
            pages.append(
                {
                    "text": text,
                    "metadata": {
                        "source": filename,
                        "page": page_num + 1,
                        "category": category,
                        "file_path": pdf_path,
                    },
                }
            )

        pdf.close()

        # ── Fallback: pdfplumber when fitz total < 100 chars ─────────────
        if len(total_text.strip()) < 100:
            logger.warning(
                "PyMuPDF yielded < 100 chars for %s — retrying with pdfplumber.",
                filename,
            )
            pages = []
            image_only_skipped = 0
            with pdfplumber.open(pdf_path) as pdf_pl:
                for page_num, page_pl in enumerate(pdf_pl.pages):
                    text = page_pl.extract_text() or ""
                    if len(text.strip()) < 50:
                        logger.debug(
                            "pdfplumber: skipping page %d of %s (< 50 chars).",
                            page_num + 1,
                            filename,
                        )
                        image_only_skipped += 1
                        continue
                    pages.append(
                        {
                            "text": text,
                            "metadata": {
                                "source": filename,
                                "page": page_num + 1,
                                "category": category,
                                "file_path": pdf_path,
                            },
                        }
                    )

        logger.info(
            "Extracted %d pages from %s  (%d image-only pages skipped).",
            len(pages),
            filename,
            image_only_skipped,
        )

    except fitz.FileDataError as exc:
        logger.error(
            "fitz.FileDataError reading %s: %s — skipping file.", filename, exc
        )
        return []
    except Exception:
        logger.error(
            "Failed to extract text from %s", pdf_path, exc_info=True
        )
        return []

    # Embed skip count in first page metadata for aggregation by the caller.
    if pages:
        pages[0]["metadata"]["_image_only_skipped"] = image_only_skipped

    return pages


# ---------------------------------------------------------------------------
# SECTION 2 — Chunking
# ---------------------------------------------------------------------------
def chunk_documents(pages: list[dict]) -> list[Document]:
    """Split page dicts into overlapping LangChain Document chunks.

    Converts each page dict (``{"text": str, "metadata": dict}``) to a
    :class:`~langchain_core.documents.Document`, then applies
    :class:`~langchain_text_splitters.RecursiveCharacterTextSplitter`
    with parameters tuned for technical / academic text:

    - ``chunk_size=800`` — fits Google ``text-embedding-004``'s sweet spot
      and keeps LaTeX formula context intact within a chunk.
    - ``chunk_overlap=150`` — ensures sentence-boundary continuity across
      chunk borders.
    - Separator hierarchy: paragraph → newline → sentence → word → char.

    Args:
        pages: Output from :func:`extract_text_from_pdf`.  Each element is::

            {"text": str, "metadata": {"source": str, "page": int, ...}}

    Returns:
        List of :class:`~langchain_core.documents.Document` objects.
        Every chunk inherits the parent page's metadata and gains an
        additional ``"chunk_index"`` key (0-based, scoped to the whole run).
        The internal ``_image_only_skipped`` key is stripped before storage.

    Example:
        >>> pages = extract_text_from_pdf("dip.pdf", "textbook")
        >>> chunks = chunk_documents(pages)
        >>> chunks[0].metadata
        {'source': 'dip.pdf', 'page': 1, 'category': 'textbook',
         'file_path': '...', 'chunk_index': 0}
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    # Convert page dicts → Documents, stripping internal stats keys
    documents: list[Document] = [
        Document(
            page_content=p["text"],
            metadata={k: v for k, v in p["metadata"].items() if not k.startswith("_")},
        )
        for p in pages
    ]

    raw_chunks = splitter.split_documents(documents)

    for idx, chunk in enumerate(raw_chunks):
        chunk.metadata["chunk_index"] = idx

    logger.info(
        "Created %d chunks from %d pages.", len(raw_chunks), len(pages)
    )
    return raw_chunks


# ---------------------------------------------------------------------------
# SECTION 3 — Embedding model
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=4)
def get_embedding_model(use_google: bool = True):
    """Return the active embedding model (Google primary, HuggingFace fallback).

    Results are cached per (use_google,) so the SentenceTransformer weights are
    loaded exactly once per process — subsequent calls return the same instance.

    Strategy
    --------
    1. **Primary** — :class:`~langchain_google_genai.GoogleGenerativeAIEmbeddings`
       using ``EMBEDDING_MODEL`` from ``.env`` (default ``models/text-embedding-004``).
       Requires ``GOOGLE_API_KEY`` to be set and ``langchain-google-genai < 2.0``
       installed (v2+ routes to the v1beta endpoint where the model is unavailable).
    2. **Fallback** — :class:`~langchain_community.embeddings.HuggingFaceEmbeddings`
       with ``all-MiniLM-L6-v2``.  Fully local, no API key required.

    Args:
        use_google: Set ``False`` to force the local HuggingFace fallback.

    Returns:
        A LangChain-compatible embeddings object implementing
        ``embed_documents`` / ``embed_query``.

    Example:
        >>> emb = get_embedding_model()
        >>> vectors = emb.embed_documents(["What is convolution?"])
        >>> len(vectors[0])
        768
    """
    if use_google and os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            model = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
            logger.info("Using Google embeddings: %s", model)
            return GoogleGenerativeAIEmbeddings(model=model)
        except Exception:
            logger.warning(
                "Google embeddings failed — falling back to SentenceTransformers.",
                exc_info=True,
            )

    logger.info("Using SentenceTransformers fallback: all-MiniLM-L6-v2")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings  # preferred (no deprecation)
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[no-redef]
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Backward-compatible alias used by external callers and older scripts
get_embeddings = get_embedding_model


# ---------------------------------------------------------------------------
# SECTION 3 — Incremental source tracking
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
# SECTION 3 — Auth-error detection helper
# ---------------------------------------------------------------------------
def _is_auth_error(exc: BaseException) -> bool:
    """
    Return True if the exception (or any chained cause) indicates an
    unrecoverable API-key or API-version error that retrying cannot fix.

    Catches two distinct failure modes:
    - 400 INVALID_ARGUMENT / API key expired  → key is invalid/expired
    - 404 NOT_FOUND / model not found for v1beta → langchain-google-genai v2+
      installed; it routes embeddings to the v1beta endpoint where
      text-embedding-004 is not exposed (fix: pin langchain-google-genai<2.0)
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        msg = str(current).lower()
        if any(
            phrase in msg
            for phrase in (
                "api key expired",
                "api_key_invalid",
                "api key not valid",
                "key expired",
                "invalid_argument",
                # 404 raised by google-genai SDK (langchain-google-genai v2+)
                # when text-embedding-004 is unavailable at the v1beta endpoint
                "not found for api version",
                "not supported for embedcontent",
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


# ---------------------------------------------------------------------------
# SECTION 3 — Batch insertion with retry
# ---------------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _add_with_retry(vector_store, batch: list[Document]) -> None:
    """Add one batch to Chroma, with automatic tenacity retries.

    Args:
        vector_store: Initialised :class:`~langchain_chroma.Chroma` instance.
        batch:        A slice of chunk Documents to embed and store.
    """
    vector_store.add_documents(batch)


def embed_and_store(
    chunks: list[Document],
    persist_dir: str,
    batch_size: int = 50,
):
    """Embed *chunks* and persist them to the ChromaDB collection in batches.

    Creates (or opens) the ``dip_knowledge_base`` Chroma collection at
    *persist_dir*.  Chunks are sent to the Google embedding API in batches
    of *batch_size* with a 1.2-second sleep between batches to stay within
    Google's 1 500 req/min rate limit.

    On an unrecoverable API error (expired key, wrong SDK version), raises
    a :class:`RuntimeError` immediately with human-readable remediation steps.

    Args:
        chunks:      Output from :func:`chunk_documents`.
        persist_dir: Path to the persistent ChromaDB directory.
        batch_size:  Chunks per embedding API call.  Default ``50``.

    Returns:
        The :class:`~langchain_chroma.Chroma` vector-store instance (for
        downstream use or verification).

    Raises:
        RuntimeError: If an unrecoverable API key / model-version error is
            detected.  Human-readable remediation steps are included in the
            message.

    Example:
        >>> vs = embed_and_store(chunks, persist_dir="./data/chroma_db")
        >>> vs._collection.count()
        372
    """
    from langchain_chroma import Chroma
    from tqdm import tqdm

    embeddings = get_embedding_model()
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=_COLLECTION_NAME,
    )

    stored = 0
    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]

    for batch in tqdm(batches, desc="Embedding chunks", unit="batch"):
        try:
            _add_with_retry(vector_store, batch)
            stored += len(batch)
            time.sleep(1.2)  # respect Google's 1 500 req/min rate limit
        except Exception as _batch_exc:
            if _is_auth_error(_batch_exc):
                raise RuntimeError(
                    "\n❌  Embedding API error — aborting ingestion.\n\n"
                    "  Cause A — API key expired / invalid:\n"
                    "    1. Get a fresh key  →  https://aistudio.google.com/app/apikey\n"
                    "    2. Re-run the 'Update API Key' cell, then re-run ingestion.\n\n"
                    "  Cause B — langchain-google-genai v2 installed (404 v1beta):\n"
                    "    Cell 2 must pin:  langchain-google-genai<2.0\n"
                    "    Runtime → Restart session, then run ALL cells from Cell 2."
                ) from _batch_exc
            logger.error(
                "Failed to store batch after 3 retries — skipping %d chunks.",
                len(batch),
                exc_info=True,
            )

    total = vector_store._collection.count()
    logger.info(
        "Stored %d new chunks this run; total collection size: %d", stored, total
    )
    return vector_store


# ---------------------------------------------------------------------------
# SECTION 4 — Orchestrator
# ---------------------------------------------------------------------------
def run_ingestion_pipeline(
    raw_dir: str = "./data/raw",
    persist_dir: str | None = None,
    batch_size: int = 50,
) -> dict:
    """Full ingestion pipeline: scan → extract → chunk → embed → persist.

    Scans all three category sub-directories under *raw_dir*, skips files
    whose filenames are already present in the ChromaDB collection
    (incremental / resumable across sessions), and stores all new chunks.

    Auth errors (expired API key, wrong SDK version) are re-raised
    immediately so Colab / CLI callers see a clear failure message.
    All other per-file errors are caught, recorded in ``stats["errors"]``,
    and execution continues with the next file.

    Args:
        raw_dir:     Root directory containing the three category sub-folders.
            Defaults to ``./data/raw``.
        persist_dir: ChromaDB persistence directory.
            Defaults to ``$CHROMA_PERSIST_DIR`` or ``./data/chroma_db``.
        batch_size:  Chunks per embedding API batch.  Default ``50``.

    Returns:
        Statistics dictionary::

            {
                "processed_files":    int,
                "skipped_files":      int,   # already in DB
                "total_pages":        int,
                "total_chunks":       int,
                "image_only_skipped": int,   # pages with < 50 chars
                "errors":             list[str],
            }

    Example:
        >>> stats = run_ingestion_pipeline(raw_dir="/content/drive/MyDrive/.../data/raw")
        >>> print(stats["total_chunks"])
        2847
    """
    import chromadb

    if persist_dir is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "processed_files": 0,
        "skipped_files": 0,
        "total_pages": 0,
        "total_chunks": 0,
        "image_only_skipped": 0,
        "errors": [],
    }

    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection = chroma_client.get_or_create_collection(_COLLECTION_NAME)
    raw_path = Path(raw_dir)

    for subdir_name, category in _SUBDIR_TO_CATEGORY.items():
        subdir = raw_path / subdir_name
        if not subdir.exists():
            logger.warning("Sub-directory not found, skipping: %s", subdir)
            continue

        pdf_files = sorted(subdir.glob("**/*.pdf"))
        logger.info(
            "Scanning %s  →  %d PDF(s) found  (category=%s)",
            subdir_name,
            len(pdf_files),
            category,
        )

        for pdf_path in pdf_files:
            filename = pdf_path.name

            try:
                existing = collection.get(where={"source": filename}, limit=1)
                if existing.get("ids"):
                    logger.info("Skipping already-ingested: %s", filename)
                    stats["skipped_files"] += 1
                    continue
            except Exception:
                logger.debug(
                    "Per-file DB skip check failed for %s; continuing ingestion.",
                    filename,
                    exc_info=True,
                )

            try:
                pages = extract_text_from_pdf(str(pdf_path), category)
                if not pages:
                    stats["errors"].append(f"No text extracted from {filename}")
                    continue

                # Harvest image-only skip count before chunking strips the key
                img_skipped = pages[0].get("metadata", {}).pop(
                    "_image_only_skipped", 0
                )
                stats["image_only_skipped"] += img_skipped
                stats["total_pages"] += len(pages)

                chunks = chunk_documents(pages)
                embed_and_store(chunks, persist_dir, batch_size=batch_size)

                stats["processed_files"] += 1
                stats["total_chunks"] += len(chunks)

            except RuntimeError:
                # Auth / API-version errors: abort the whole pipeline
                raise
            except Exception as exc:
                msg = f"{filename}: {exc}"
                logger.error("Pipeline error — %s", msg, exc_info=True)
                stats["errors"].append(msg)

    logger.info(
        "Ingestion complete: %d files processed, %d skipped, "
        "%d pages extracted, %d chunks stored, %d image-only pages skipped.",
        stats["processed_files"],
        stats["skipped_files"],
        stats["total_pages"],
        stats["total_chunks"],
        stats["image_only_skipped"],
    )
    return stats


# ---------------------------------------------------------------------------
# SECTION 3 — Load vector store (for retrieval)
# ---------------------------------------------------------------------------
def _patch_chroma_config_json(db_path: str) -> None:
    """Idempotent: ensure collections.config_json_str has the '_type' field.

    ChromaDB 0.6+ requires ``config_json_str`` to contain
    ``{"_type": "CollectionConfigurationInternal"}`` or it crashes with
    ``KeyError: '_type'``.  Collections built on earlier versions (or on Colab
    with certain chromadb builds) store ``{}`` which triggers the bug.  This
    helper patches every affected row in-place so the application never needs
    re-ingestion to recover.
    """
    import json
    import sqlite3

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name, config_json_str FROM collections")
        rows = cur.fetchall()
        for name, cfg_str in rows:
            try:
                cfg = json.loads(cfg_str) if cfg_str else {}
            except json.JSONDecodeError:
                cfg = {}
            if cfg.get("_type") != "CollectionConfigurationInternal":
                patched = json.dumps({"_type": "CollectionConfigurationInternal"})
                cur.execute(
                    "UPDATE collections SET config_json_str=? WHERE name=?",
                    (patched, name),
                )
                logger.debug("Patched config_json_str for collection '%s'.", name)
        conn.commit()
        conn.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not patch chroma config_json_str: %s", exc)


def _detect_collection_dim(db_path: str, collection_name: str = _COLLECTION_NAME) -> int | None:
    """Read the stored vector dimension for *collection_name* from SQLite.

    Returns the integer dimension, or ``None`` if it cannot be determined.
    """
    import sqlite3

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT dimension FROM collections WHERE name=?", (collection_name,)
        )
        row = cur.fetchone()
        conn.close()
        return int(row[0]) if row and row[0] is not None else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not detect collection dimension: %s", exc)
        return None


@functools.lru_cache(maxsize=2)
def load_vectorstore(persist_dir: str | None = None):
    """Return the existing Chroma vector store for retrieval-time use.

    Result is cached per persist_dir — the Chroma client is opened once and
    reused for every subsequent retrieval call in the same process.

    This is a lightweight helper called by :mod:`app.retrieval.retriever`,
    :mod:`app.summarization.summarizer`, and :mod:`app.evaluation.metrics`.
    It does **not** re-ingest any documents.

    On every call it:

    1. Patches ``config_json_str`` in the SQLite if the ``_type`` field is
       missing (ChromaDB 0.6+ compatibility fix — idempotent).
    2. Reads the stored vector dimension and selects the matching embedding
       model automatically:
       - **384-dim** → ``all-MiniLM-L6-v2`` (HuggingFace, local)
       - **768-dim** → ``models/text-embedding-004`` (Google)
       - **unknown**  → follows normal primary/fallback logic

    Args:
        persist_dir: Path to the persisted ChromaDB directory.
            Defaults to ``$CHROMA_PERSIST_DIR`` env var or ``./data/chroma_db``.

    Returns:
        :class:`~langchain_chroma.Chroma` instance ready for similarity search.

    Raises:
        FileNotFoundError: If *persist_dir* does not exist on disk, i.e. the
            ingestion pipeline has never been run.

    Example:
        >>> vs = load_vectorstore()
        >>> vs.similarity_search("edge detection kernel", k=3)
    """
    from langchain_chroma import Chroma

    if persist_dir is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

    persist_dir = str(Path(persist_dir).resolve())

    if not Path(persist_dir).exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found at '{persist_dir}'.\n"
            "Run the ingestion pipeline first:\n"
            "  python scripts/run_ingestion.py\n"
            "or run notebooks/ingestion_colab.ipynb in Google Colab."
        )

    db_path = str(Path(persist_dir) / "chroma.sqlite3")

    # Step 1: patch config_json_str so ChromaDB 0.6+ can open old collections
    if Path(db_path).exists():
        _patch_chroma_config_json(db_path)

    # Step 2: auto-select embedding model that matches the stored dimension
    stored_dim = _detect_collection_dim(db_path) if Path(db_path).exists() else None
    if stored_dim == 384:
        # Collection was built with all-MiniLM-L6-v2 (384-dim) — must match
        logger.info(
            "Detected 384-dim collection — using all-MiniLM-L6-v2 embeddings."
        )
        embeddings = get_embedding_model(use_google=False)
    elif stored_dim == 768:
        logger.info("Detected 768-dim collection — using Google text-embedding-004.")
        embeddings = get_embedding_model(use_google=True)
    else:
        logger.info(
            "Collection dimension unknown (%s) — using default embedding strategy.",
            stored_dim,
        )
        embeddings = get_embedding_model()

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=_COLLECTION_NAME,
    )


# ---------------------------------------------------------------------------
# SECTION 4 — CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Ingest PDFs into the dip_knowledge_base ChromaDB collection."
    )
    parser.add_argument(
        "--raw-dir",
        default=os.getenv("RAW_DIR", "./data/raw"),
        help="Root directory containing the three PDF sub-folders.",
    )
    parser.add_argument(
        "--persist-dir",
        default=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"),
        help="ChromaDB persistence directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Chunks per embedding API call (default: 50).",
    )
    args = parser.parse_args()

    stats = run_ingestion_pipeline(
        raw_dir=args.raw_dir,
        persist_dir=args.persist_dir,
        batch_size=args.batch_size,
    )

    # ── Final stats table ───────────────────────────────────────────────
    sep = "─" * 38
    print(f"\n{sep}")
    print("  Ingestion complete")
    print(sep)
    print(f"  Total files processed   : {stats['processed_files']}")
    print(f"  Skipped (already in DB) : {stats['skipped_files']}")
    print(f"  Total pages extracted   : {stats['total_pages']}")
    print(f"  Image-only pages skipped: {stats['image_only_skipped']}")
    print(f"  Total chunks stored     : {stats['total_chunks']}")
    if stats["errors"]:
        print(f"  \u26a0\ufe0f  Errors ({len(stats['errors'])})")
        for err in stats["errors"]:
            print(f"    \u2022 {err}")
    print(sep)
