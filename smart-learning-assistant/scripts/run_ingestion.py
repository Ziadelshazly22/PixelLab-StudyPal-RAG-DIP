#!/usr/bin/env python3
"""
scripts/run_ingestion.py
------------------------
Ingestion Pipeline Runner — University Server / Strong Local Environment

Standalone script: no Jupyter / Colab infrastructure required.
Run from the project root:

    python scripts/run_ingestion.py
    python scripts/run_ingestion.py --raw-dir /path/to/raw --persist-dir /path/to/chroma_db
    python scripts/run_ingestion.py --batch-size 25
    python scripts/run_ingestion.py --clear-existing   # ⚠️ deletes chroma_db and rebuilds

Exit codes
----------
0 — success (no errors)
1 — pipeline crashed OR at least one per-file error occurred
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

# ── Make project root importable ─────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent          # .../scripts/
_PROJECT_ROOT = _SCRIPT_DIR.parent                     # .../smart-learning-assistant/
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Logging — console + file
# ---------------------------------------------------------------------------
def _configure_logging(log_level: str) -> None:
    log_dir = _PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "ingestion.log"

    fmt = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )
    # Silence noisy third-party loggers
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the DIP Knowledge Base ingestion pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        default=str(_PROJECT_ROOT / "data" / "raw"),
        help="Root directory containing the three PDF category sub-folders.",
    )
    parser.add_argument(
        "--persist-dir",
        default=os.getenv(
            "CHROMA_PERSIST_DIR",
            str(_PROJECT_ROOT / "data" / "chroma_db"),
        ),
        help="ChromaDB persistence directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of chunks per embedding API call (respects rate limits).",
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="⚠️  Delete the existing vector store and rebuild from scratch.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pretty-print stats table
# ---------------------------------------------------------------------------
def _print_stats(stats: dict, elapsed: float) -> None:
    width = 48
    border = "─" * width
    lines = [
        f"┌{border}┐",
        f"│{'  📊  Ingestion Summary':^{width}}│",
        f"├{border}┤",
        f"│  {'Files processed':<22} {stats['processed_files']:>22}  │",
        f"│  {'Files skipped':<22} {stats['skipped_files']:>22}  │",
        f"│  {'Total chunks stored':<22} {stats['total_chunks']:>22}  │",
        f"│  {'Errors':<22} {len(stats['errors']):>22}  │",
        f"│  {'Wall-clock time':<22} {elapsed:>19.1f}s  │",
        f"└{border}┘",
    ]
    print("\n" + "\n".join(lines) + "\n")

    if stats["errors"]:
        print("  ⚠️  Errors encountered:")
        for err in stats["errors"]:
            print(f"     • {err}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = _parse_args()
    _configure_logging(args.log_level)

    logger.info("=" * 62)
    logger.info("  Smart Learning Assistant — Ingestion Pipeline Runner")
    logger.info("=" * 62)
    logger.info("Raw dir    : %s", args.raw_dir)
    logger.info("Persist dir: %s", args.persist_dir)
    logger.info("Batch size : %d", args.batch_size)
    logger.info("Log file   : %s", _PROJECT_ROOT / "logs" / "ingestion.log")

    # ── Optional: clear existing vector store ─────────────────────────────
    if args.clear_existing:
        persist_path = Path(args.persist_dir)
        if persist_path.exists():
            shutil.rmtree(persist_path)
            logger.warning("Cleared existing vector store at %s", persist_path)
        else:
            logger.info(
                "No existing vector store at %s — nothing to clear.", persist_path
            )

    # ── Run pipeline ──────────────────────────────────────────────────────
    try:
        from app.ingestion.pipeline import run_ingestion_pipeline

        start = time.time()
        stats = run_ingestion_pipeline(
            raw_dir=args.raw_dir,
            persist_dir=args.persist_dir,
        )
        elapsed = time.time() - start

    except Exception:
        logger.critical("Ingestion pipeline crashed.", exc_info=True)
        return 1

    # ── Print summary ──────────────────────────────────────────────────────
    _print_stats(stats, elapsed)
    logger.info("Total wall-clock time: %.1fs", elapsed)

    if stats["errors"]:
        logger.warning(
            "Pipeline completed with %d error(s). "
            "Check logs/ingestion.log for details.",
            len(stats["errors"]),
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
