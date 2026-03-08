# -*- coding: utf-8 -*-
"""
run_all.py
----------
Production health-check and go/no-go launcher for the DIP AI Tutor.

Performs five automated checks before signalling that the app is
"READY FOR DEMO".  Designed to be run once before any live presentation
or handoff.

Checks
------
1. Environment validation  (validate_setup.py --skip-api)
2. Unit tests              (pytest tests/ -v)
3. ChromaDB populated      (> 100 chunks in dip_knowledge_base)
4. Evaluation report       (evaluation_report.md exists + non-empty)
5. README placeholders     (no "~0.XX" left in README.md)

Usage
-----
    python run_all.py

Exit codes
----------
    0  All checks green — proceed with demo.
    1  One or more checks failed — fix before presenting.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent          # smart-learning-assistant/
_REPO_ROOT = _ROOT.parent                        # repo root
_VALIDATE = _ROOT / "validate_setup.py"
_TESTS_DIR = _ROOT / "tests"
_CHROMA_DIR = Path(
    os.getenv("CHROMA_PERSIST_DIR", str(_ROOT / "data" / "chroma_db"))
)
_EVAL_REPORT = _REPO_ROOT / "evaluation_report.md"
_README = _REPO_ROOT / "README.md"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
_NO_COLOR = not sys.stdout.isatty()


def _green(s: str) -> str:
    return s if _NO_COLOR else f"\033[32m{s}\033[0m"


def _red(s: str) -> str:
    return s if _NO_COLOR else f"\033[31m{s}\033[0m"


def _yellow(s: str) -> str:
    return s if _NO_COLOR else f"\033[33m{s}\033[0m"


def _cyan(s: str) -> str:
    return s if _NO_COLOR else f"\033[36m{s}\033[0m"


# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------
_results: list[tuple[bool, str, str]] = []   # (ok, label, detail)


def _record(ok: bool, label: str, detail: str = "") -> bool:
    _results.append((ok, label, detail))
    icon = _green("✅") if ok else _red("❌")
    status = _green("PASS") if ok else _red("FAIL")
    print(f"  {icon} [{status}] {label}")
    if detail:
        for line in detail.splitlines():
            print(f"           {line}")
    return ok


# ---------------------------------------------------------------------------
# Check 1 — validate_setup.py
# ---------------------------------------------------------------------------
def check_environment() -> bool:
    print(_cyan("\n[1/5] Running environment pre-flight..."))
    result = subprocess.run(
        [sys.executable, str(_VALIDATE), "--skip-api"],
        capture_output=True,
        text=True,
    )
    ok = result.returncode == 0
    if not ok:
        # Print the last 10 lines of output to surface the failure message
        last_lines = "\n".join(result.stdout.strip().splitlines()[-10:])
        return _record(False, "Environment validation", last_lines)
    return _record(True, "Environment validation (validate_setup.py)")


# ---------------------------------------------------------------------------
# Check 2 — pytest
# ---------------------------------------------------------------------------
def check_tests() -> bool:
    print(_cyan("\n[2/5] Running test suite..."))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(_TESTS_DIR), "-v", "--tb=short", "-q"],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
    )
    output = result.stdout + result.stderr

    # Parse summary line like "5 passed, 1 failed in 12.3s"
    summary_match = re.search(
        r"(\d+ passed)?[, ]*(\d+ failed)?[, ]*(\d+ error)?",
        output,
        re.IGNORECASE,
    )
    summary = summary_match.group(0).strip() if summary_match else "unknown"

    # Also look for the cleaner "X passed" / "X failed" pattern pytest uses
    short_match = re.search(r"\d+ passed|\d+ failed|\d+ error", output)
    if short_match:
        summary = short_match.group(0)

    ok = result.returncode == 0
    detail = summary if ok else f"{summary}\n{output.strip()[-400:]}"
    return _record(ok, f"Unit tests ({summary})", "" if ok else detail)


# ---------------------------------------------------------------------------
# Check 3 — ChromaDB populated
# ---------------------------------------------------------------------------
def check_chromadb() -> bool:
    print(_cyan("\n[3/5] Checking ChromaDB collection..."))

    if not _CHROMA_DIR.exists():
        return _record(
            False,
            "ChromaDB populated",
            f"Directory not found: {_CHROMA_DIR}",
        )

    try:
        import chromadb  # type: ignore
        client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
        collection = client.get_collection("dip_knowledge_base")
        count = collection.count()
        MIN_CHUNKS = 100
        ok = count >= MIN_CHUNKS
        label = f"ChromaDB: {count:,} chunks in dip_knowledge_base"
        detail = "" if ok else f"Only {count} chunks — expected ≥ {MIN_CHUNKS}. Re-run ingestion."
        return _record(ok, label, detail)
    except Exception as exc:
        return _record(False, "ChromaDB accessible", str(exc))


# ---------------------------------------------------------------------------
# Check 4 — Evaluation report exists
# ---------------------------------------------------------------------------
def check_eval_report() -> bool:
    print(_cyan("\n[4/5] Checking evaluation report..."))

    if not _EVAL_REPORT.exists():
        return _record(
            False,
            "evaluation_report.md present",
            f"File not found: {_EVAL_REPORT}\n"
            "Run the evaluation notebook to generate it.",
        )

    content = _EVAL_REPORT.read_text(encoding="utf-8")
    if len(content.strip()) < 100:
        return _record(
            False,
            "evaluation_report.md non-empty",
            "File exists but appears empty or truncated.",
        )

    # Look for the overall RAGAS score in the file
    score_match = re.search(r"overall[^\d]*([\d.]+)", content, re.IGNORECASE)
    score_str = f" (overall={score_match.group(1)})" if score_match else ""
    return _record(True, f"evaluation_report.md present{score_str}")


# ---------------------------------------------------------------------------
# Check 5 — README has no placeholder scores
# ---------------------------------------------------------------------------
def check_readme_placeholders() -> bool:
    print(_cyan("\n[5/5] Checking README for placeholder values..."))

    if not _README.exists():
        return _record(False, "README.md present", f"Not found: {_README}")

    content = _README.read_text(encoding="utf-8")
    placeholders = re.findall(r"~0\.\d{2}", content)
    if placeholders:
        return _record(
            False,
            "README placeholders",
            f"Found placeholder values: {set(placeholders)}\n"
            "Replace them with real RAGAS scores from evaluation_report.md.",
        )
    return _record(True, "README.md has no placeholder scores")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print()
    print("=" * 62)
    print("  🎓 DIP AI Tutor — Pre-Demo Go/No-Go Health Check")
    print("=" * 62)

    check_environment()
    check_tests()
    check_chromadb()
    check_eval_report()
    check_readme_placeholders()

    # Final verdict
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    failed = total - passed

    print()
    print("=" * 62)
    if failed == 0:
        print(_green(f"  ALL {total} CHECKS PASSED"))
        print()
        print("  READY FOR DEMO: " + _green("YES ✅"))
        print()
        print("  Next steps:")
        print("    1. Start API  → Quick Start.bat   (or uvicorn main:app --reload)")
        print("    2. Open UI    → http://localhost:7860")
        print("    3. Open docs  → http://localhost:8000/docs")
        print("    4. Follow     → DEMO_SCRIPT.md")
    else:
        print(_red(f"  {failed}/{total} CHECK(S) FAILED"))
        print()
        print("  READY FOR DEMO: " + _red("NO ❌"))
        print()
        print("  Fix the failed checks above, then re-run:  python run_all.py")

    print("=" * 62)
    print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
