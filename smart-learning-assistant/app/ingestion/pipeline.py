"""
app/ingestion/pipeline.py
-------------------------
Document ingestion pipeline:

  1. Load PDFs from ``data/raw/`` using PyMuPDF (fitz) / pdfplumber
  2. Chunk text with LangChain's RecursiveCharacterTextSplitter
  3. Embed chunks with sentence-transformers (all-MiniLM-L6-v2)
  4. Persist to ChromaDB at ``CHROMA_PERSIST_DIR``
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
_COLLECTION = os.getenv("COLLECTION_NAME", "dip_knowledge_base")
_RAW_DIR = Path(__file__).parents[3] / "data" / "raw"


def _load_pdfs(directory: Path) -> list:
    """Load all PDFs in *directory* using LangChain's PyMuPDFLoader."""
    from langchain_community.document_loaders import PyMuPDFLoader

    docs = []
    pdf_files = list(directory.glob("**/*.pdf"))
    if not pdf_files:
        print(f"[ingestion] No PDF files found in {directory}")
        return docs

    for pdf_path in pdf_files:
        print(f"[ingestion] Loading: {pdf_path.name}")
        loader = PyMuPDFLoader(str(pdf_path))
        docs.extend(loader.load())

    print(f"[ingestion] Loaded {len(docs)} pages from {len(pdf_files)} PDF(s).")
    return docs


def _chunk_documents(docs: list) -> list:
    """Split documents into overlapping chunks."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[ingestion] Created {len(chunks)} chunks.")
    return chunks


def _build_vector_store(chunks: list):
    """Embed chunks and persist them in ChromaDB."""
    from langchain_chroma import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=_COLLECTION,
        persist_directory=_CHROMA_DIR,
    )
    print(f"[ingestion] Vector store persisted to: {_CHROMA_DIR}")
    return vector_store


def ingest_documents(raw_dir: str | Path | None = None) -> None:
    """
    Full ingestion pipeline: load → chunk → embed → persist.

    Parameters
    ----------
    raw_dir : str or Path, optional
        Directory containing raw PDFs.  Defaults to ``data/raw/``.
    """
    directory = Path(raw_dir) if raw_dir else _RAW_DIR
    docs = _load_pdfs(directory)
    if not docs:
        return
    chunks = _chunk_documents(docs)
    _build_vector_store(chunks)
    print("[ingestion] ✅ Ingestion complete.")


if __name__ == "__main__":
    ingest_documents()
