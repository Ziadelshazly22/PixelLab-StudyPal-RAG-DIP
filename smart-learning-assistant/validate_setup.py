# -*- coding: utf-8 -*-
"""
validate_setup.py
-----------------
Pre-flight environment validation for the DIP AI Tutor.

Checks performed
----------------
1. Python version >= 3.10
2. .env file present (GROQ_API_KEY loaded)
3. Required Python packages importable
4. ChromaDB directory exists and collection has > 0 chunks
5. Groq API key reachable (optional, skipped if --skip-api is passed)

Usage
-----
    python validate_setup.py            # full check (pings Groq API)
    python validate_setup.py --skip-api # skip live API check (CI / offline)

Exit codes
----------
    0  All checks passed.
    1  One or more checks failed — server should NOT be started.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Colors (gracefully degraded if the terminal doesn't support ANSI)
# ---------------------------------------------------------------------------
_NO_COLOR = not sys.stdout.isatty() or os.name == "nt" and os.environ.get("NO_COLOR")


def _green(s: str) -> str:
    return s if _NO_COLOR else f"\033[32m{s}\033[0m"


def _red(s: str) -> str:
    return s if _NO_COLOR else f"\033[31m{s}\033[0m"


def _yellow(s: str) -> str:
    return s if _NO_COLOR else f"\033[33m{s}\033[0m"


PASS = _green("  ✅ PASS")
FAIL = _red("  ❌ FAIL")
WARN = _yellow("  ⚠️  WARN")

_ROOT = Path(__file__).resolve().parent  # smart-learning-assistant/
_CHECKS: list[tuple[bool, str]] = []


def _record(ok: bool, label: str, detail: str = "") -> bool:
    icon = PASS if ok else FAIL
    line = f"{icon}  {label}"
    if detail:
        line += f"\n         {detail}"
    print(line)
    _CHECKS.append((ok, label))
    return ok


# ---------------------------------------------------------------------------
# Check 1 — Python version
# ---------------------------------------------------------------------------
def check_python_version() -> bool:
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 10)
    return _record(
        ok,
        f"Python version: {major}.{minor}",
        "" if ok else "Requires Python 3.10+. Install from https://python.org",
    )


# ---------------------------------------------------------------------------
# Check 2 — .env file and GROQ_API_KEY
# ---------------------------------------------------------------------------
def check_env_file() -> bool:
    env_path = _ROOT / ".env"
    if not env_path.exists():
        return _record(
            False,
            ".env file present",
            f"Missing {env_path}. Run: copy .env.example .env  then fill in GROQ_API_KEY",
        )

    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        # Fallback: parse manually
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_groq_api_key_here":
        return _record(
            False,
            ".env GROQ_API_KEY set",
            "GROQ_API_KEY is missing or still the placeholder. Edit .env and add your key.",
        )
    masked = api_key[:8] + "***"
    _record(True, ".env file present")
    return _record(True, f"GROQ_API_KEY loaded ({masked})")


# ---------------------------------------------------------------------------
# Check 3 — Required packages importable
# ---------------------------------------------------------------------------
_REQUIRED_PACKAGES = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("langchain_core", "langchain-core"),
    ("langchain_groq", "langchain-groq"),
    ("langchain_community", "langchain-community"),
    ("chromadb", "chromadb"),
    ("gradio", "gradio"),
    ("sentence_transformers", "sentence-transformers"),
    ("fitz", "pymupdf"),
    ("pdfplumber", "pdfplumber"),
    ("dotenv", "python-dotenv"),
]


def check_packages() -> bool:
    missing = []
    for module, pkg in _REQUIRED_PACKAGES:
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)

    if missing:
        return _record(
            False,
            "Required packages importable",
            f"Missing: {', '.join(missing)}. Run: pip install -r requirements.txt",
        )
    return _record(True, f"Required packages importable ({len(_REQUIRED_PACKAGES)} checked)")


# ---------------------------------------------------------------------------
# Check 4 — ChromaDB exists and has chunks
# ---------------------------------------------------------------------------
def check_chromadb() -> bool:
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env")
    except ImportError:
        pass

    chroma_dir = Path(os.getenv("CHROMA_PERSIST_DIR", str(_ROOT / "data" / "chroma_db")))

    if not chroma_dir.exists():
        return _record(
            False,
            "ChromaDB directory exists",
            f"Not found: {chroma_dir}\n"
            "         Run ingestion first:\n"
            "           python scripts/run_ingestion.py\n"
            "         or use notebooks/ingestion_colab.ipynb on Google Colab.",
        )

    try:
        import chromadb as _chromadb
        client = _chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection("dip_knowledge_base")
        count = collection.count()
        if count == 0:
            return _record(
                False,
                "ChromaDB populated",
                "Collection exists but has 0 chunks. Re-run the ingestion pipeline.",
            )
        return _record(True, f"ChromaDB populated ({count:,} chunks in dip_knowledge_base)")
    except Exception as exc:
        return _record(
            False,
            "ChromaDB accessible",
            f"Error: {exc}",
        )


# ---------------------------------------------------------------------------
# Check 5 — Groq API reachable (live call)
# ---------------------------------------------------------------------------
def check_groq_api() -> bool:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_groq_api_key_here":
        return _record(False, "Groq API reachable", "GROQ_API_KEY not set — skipping live check.")

    try:
        import requests  # type: ignore
        r = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if r.status_code == 200:
            return _record(True, "Groq API reachable (200 OK)")
        return _record(
            False,
            "Groq API reachable",
            f"HTTP {r.status_code} — check your GROQ_API_KEY at https://console.groq.com/keys",
        )
    except Exception as exc:
        return _record(False, "Groq API reachable", f"Connection error: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="DIP AI Tutor — environment pre-flight check")
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Skip the live Groq API connectivity check (useful in CI or offline environments)",
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  DIP AI Tutor — Environment Validation")
    print("=" * 60)
    print()

    check_python_version()
    check_env_file()
    check_packages()
    check_chromadb()
    if not args.skip_api:
        check_groq_api()

    # Summary
    passed = sum(1 for ok, _ in _CHECKS if ok)
    total = len(_CHECKS)
    print()
    print("=" * 60)
    if passed == total:
        print(_green(f"  ALL {total} CHECKS PASSED — environment is ready"))
        print("  Start the server:  uvicorn main:app --reload --port 8000")
        print("  Or (Windows):      Quick Start.bat")
    else:
        failed = total - passed
        print(_red(f"  {failed}/{total} CHECK(S) FAILED — fix the issues above before starting"))
    print("=" * 60)
    print()

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
