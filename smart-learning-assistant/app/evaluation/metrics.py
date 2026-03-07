# -*- coding: utf-8 -*-
"""
app/evaluation/metrics.py
--------------------------
Two-phase evaluation pipeline for the DIP AI Tutor.

Phase A — collect_answers()  [runs LOCALLY against the running FastAPI server]
    - Loads test_questions.json (18 questions: 15 DIP + 3 off-topic)
    - POSTs each question to /chat with a fresh UUID session (stateless, no memory bleed)
    - Records: answer, contexts (source filenames), latency
    - Validates guardrail on off-topic questions (must contain refusal phrase)
    - Saves everything to data/eval_intermediate.json

Phase B — run_ragas_scoring()  [runs in GOOGLE COLAB to avoid local quota contention]
    - Loads data/eval_intermediate.json
    - Builds a RAGAS Dataset
    - Runs faithfulness, answer_relevancy, context_precision, context_recall
    - Returns result DataFrame

generate_report()  [runs after Phase B]
    - Generates a human-readable evaluation_report.md

CLI Usage
---------
    # Phase A (local, fast — no heavy LLM calls)
    python app/evaluation/metrics.py --phase collect

    # Phase B (Colab — after uploading eval_intermediate.json to Drive)
    python app/evaluation/metrics.py --phase score
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

try:
    from tqdm import tqdm
except ImportError:  # fallback: tqdm not installed
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]          # smart-learning-assistant/
_QUESTIONS_FILE = Path(__file__).parent / "test_questions.json"
_INTERMEDIATE_FILE = _ROOT / "data" / "eval_intermediate.json"
_REPORT_FILE = _ROOT / "evaluation_report.md"

_BACKEND_URL = os.getenv("EVAL_BACKEND_URL", "http://localhost:8000")
_CHAT_URL = f"{_BACKEND_URL}/chat"
_CHAT_TIMEOUT = 120  # seconds

_REFUSAL_PHRASE = "out of focus"   # matches prompt: "this question falls out of focus"

# Delay between consecutive Gemini requests (seconds).
# Free tier ceiling is 15 RPM = 1 request / 4 s.  Default 5 s gives headroom.
# Override with EVAL_REQUEST_DELAY env var if needed.
_REQUEST_DELAY_S: float = float(os.getenv("EVAL_REQUEST_DELAY", "5"))

_RAGAS_TARGETS = {
    "faithfulness": 0.7,
    "answer_relevancy": 0.7,
    "context_precision": 0.7,
    "context_recall": 0.7,
}

_DAILY_QUOTA_ID = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
_QUOTA_LIMIT_ZERO_MSG = (
    "\n"
    "=" * 70 + "\n"
    "  QUOTA LIMIT = 0  (Google Cloud billing restriction)\n"
    "=" * 70 + "\n"
    "  The API key is valid but the Google Cloud project has quota limit = 0.\n"
    "  This is NOT a usage issue — it means the free-tier quota is blocked.\n\n"
    "  Fix (takes ~2 minutes, no charge ever):\n"
    "  1. Go to https://console.cloud.google.com/billing\n"
    "  2. Link a billing account to the project that owns your API key\n"
    "     (a free account is enough — you only need to provide a card)\n"
    "  3. Re-run this script — the free tier (200 req/day) is then unlocked\n"
    "=" * 70
)


def _is_daily_quota_exhausted(response: "requests.Response") -> bool:
    """Return True when the 503 body indicates a *daily* quota violation.

    Per-minute quota is worth retrying (reset in <60 s).
    Daily quota is NOT worth retrying — the limit resets at midnight PT.
    """
    try:
        detail = response.json().get("detail", "")
        return _DAILY_QUOTA_ID in str(detail)
    except Exception:  # noqa: BLE001
        return False


def _preflight_quota_check() -> bool:
    """Send ONE tiny test request before running the full 18-question collection.

    Returns True if the key has usable quota, False (+ prints instructions) if not.
    """
    logger.info("Pre-flight: testing Gemini quota with a 1-token probe...")
    test_payload = {"question": "hi", "session_id": str(uuid.uuid4())}
    try:
        r = requests.post(_CHAT_URL, json=test_payload, timeout=30)
        if r.status_code == 200:
            logger.info("✅ Pre-flight passed — quota available. Starting collection.")
            return True
        if r.status_code == 503 and _is_daily_quota_exhausted(r):
            print(_QUOTA_LIMIT_ZERO_MSG)
            return False
        # Any other status (4xx/5xx not quota-related) — warn but proceed
        logger.warning("Pre-flight got %s — proceeding anyway.", r.status_code)
        return True
    except requests.exceptions.ConnectionError:
        logger.error("Pre-flight: server not reachable at %s. Start the server first.", _BACKEND_URL)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("Pre-flight probe failed (%s) — proceeding anyway.", exc)
        return True


# ===========================================================================
# PHASE A — Collect answers locally
# ===========================================================================

def collect_answers(questions_path: str | Path = _QUESTIONS_FILE) -> dict:
    """POST each question to the running FastAPI /chat and collect answers.

    Each DIP question uses a fresh UUID session_id so there is zero memory
    contamination between evaluation turns. Off-topic questions are sent to
    the same endpoint and their answers are validated for the guardrail phrase.

    Returns
    -------
    dict with keys: questions, answers, contexts, ground_truths, latencies,
                    guardrail_results, topic_map
    """
    questions_path = Path(questions_path)
    if not questions_path.exists():
        raise FileNotFoundError(f"test_questions.json not found at {questions_path}")

    with open(questions_path, encoding="utf-8") as f:
        all_questions: list[dict] = json.load(f)

    dip_questions      = [q for q in all_questions if not q["is_off_topic"]]
    offtopic_questions = [q for q in all_questions if q["is_off_topic"]]

    questions_out:     list[str]        = []
    answers_out:       list[str]        = []
    contexts_out:      list[list[str]]  = []
    ground_truths_out: list[str]        = []
    latencies_out:     list[float]      = []
    topic_map:         list[str]        = []

    logger.info("=== Phase A: collecting answers for %d DIP questions ===", len(dip_questions))

    # ── Pre-flight: one test call before committing to 18 questions ──────
    if not _preflight_quota_check():
        raise SystemExit(1)  # message already printed

    bar = tqdm(
        enumerate(dip_questions, start=1),
        total=len(dip_questions),
        desc="Collecting",
        unit="q",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    daily_quota_hit = False
    for i, q in bar:
        session_id = str(uuid.uuid4())
        payload = {"question": q["question"], "session_id": session_id}

        bar.set_postfix_str(q["question"][:55] + "…", refresh=True)
        t0 = time.time()
        try:
            resp = requests.post(_CHAT_URL, json=payload, timeout=_CHAT_TIMEOUT)
            latency = time.time() - t0
            if resp.status_code == 503 and _is_daily_quota_exhausted(resp):
                daily_quota_hit = True
                logger.error(
                    "\n❌ Daily Gemini quota exhausted (GenerateRequestsPerDay limit = 0).\n"
                    "   Collection aborted after %d/%d questions.\n"
                    "   Fix: get a fresh API key from a new Google account at "
                    "https://aistudio.google.com/app/apikey\n"
                    "   Then update GOOGLE_API_KEY in .env and restart the server.",
                    i - 1, len(dip_questions),
                )
                answer = ""
                contexts = []
            else:
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("answer", "")
                raw_sources: list[Any] = data.get("sources", [])
                contexts = [str(s).split("/")[-1].split("\\")[-1] for s in raw_sources]
        except requests.exceptions.ConnectionError:
            logger.error("  Connection refused — is FastAPI running on %s?", _BACKEND_URL)
            answer = ""
            contexts = []
            latency = time.time() - t0
        except Exception as exc:  # noqa: BLE001
            logger.error("  Request failed: %s", exc)
            answer = ""
            contexts = []
            latency = time.time() - t0

        bar.set_postfix({"lat": f"{latency:.1f}s", "len": len(answer)}, refresh=True)

        questions_out.append(q["question"])
        answers_out.append(answer)
        contexts_out.append(contexts if contexts else [""])
        ground_truths_out.append(q["ground_truth"])
        latencies_out.append(latency)
        topic_map.append(q["topic"])

        if daily_quota_hit:
            break  # no point sending more questions — daily limit won't reset mid-run

        # Respect the 15-RPM free-tier ceiling: 1 req / 4 s minimum.
        # Sleep only when there are more questions to ask.
        if not daily_quota_hit and i < len(dip_questions):
            time.sleep(_REQUEST_DELAY_S)

    # ── Off-topic / guardrail validation ────────────────────────────────
    guardrail_results: list[dict] = []
    logger.info("\n=== Phase A: guardrail check for %d off-topic questions ===", len(offtopic_questions))

    if daily_quota_hit:
        logger.warning("Skipping guardrail checks — daily quota already exhausted.")

    for q in tqdm(
        offtopic_questions if not daily_quota_hit else [],
        desc="Guardrail ", unit="q", dynamic_ncols=True,
    ):
        session_id = str(uuid.uuid4())
        payload = {"question": q["question"], "session_id": session_id}
        t0 = time.time()
        try:
            resp = requests.post(_CHAT_URL, json=payload, timeout=_CHAT_TIMEOUT)
            latency = time.time() - t0
            if resp.status_code == 503 and _is_daily_quota_exhausted(resp):
                logger.error("Daily quota exhausted during guardrail checks. Stopping.")
                answer = resp.json().get("detail", "")
                passed = None
                status = "UNKNOWN (daily quota exhausted)"
            elif resp.status_code == 503:
                answer = resp.json().get("detail", "")
                passed = None
                status = "UNKNOWN (quota)"
            else:
                resp.raise_for_status()
                answer = resp.json().get("answer", "")
                passed = _REFUSAL_PHRASE.lower() in answer.lower()
                status = "PASS" if passed else "FAIL"
        except requests.exceptions.ConnectionError:
            answer = ""
            passed = False
            status = "FAIL (connection refused)"
            latency = time.time() - t0
        except Exception as exc:  # noqa: BLE001
            answer = str(exc)
            passed = False
            status = f"FAIL ({exc})"
            latency = time.time() - t0

        logger.info("  Guardrail [%s] Q: %s", status, q["question"][:60])
        guardrail_results.append({
            "question": q["question"],
            "answer": answer,
            "passed": passed,
            "status": status,
            "latency": latency,
        })

    # ── Summary ──────────────────────────────────────────────────────────
    n = len(dip_questions)
    answered = sum(1 for a in answers_out if a)
    guardrail_passed = sum(1 for g in guardrail_results if g["passed"] is True)
    mean_latency = sum(latencies_out) / n if n > 0 else 0.0

    logger.info("\n=== Collection Summary ===")
    logger.info("  DIP questions answered : %d / %d", answered, n)
    logger.info("  Guardrail PASS         : %d / %d", guardrail_passed, len(offtopic_questions))
    logger.info("  Mean latency           : %.2fs", mean_latency)

    result = {
        "questions":         questions_out,
        "answers":           answers_out,
        "contexts":          contexts_out,
        "ground_truths":     ground_truths_out,
        "latencies":         latencies_out,
        "topic_map":         topic_map,
        "guardrail_results": guardrail_results,
        "collected_at":      datetime.now(timezone.utc).isoformat(),
    }

    _INTERMEDIATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_INTERMEDIATE_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("Intermediate data saved -> %s", _INTERMEDIATE_FILE)
    return result


# ===========================================================================
# PHASE B — RAGAS scoring (designed to run in Colab)
# ===========================================================================

def run_ragas_scoring(intermediate_path: str | Path = _INTERMEDIATE_FILE) -> Any:
    """Load intermediate JSON and run RAGAS evaluation.

    Automatically selects the judge LLM:
      1. GROQ_API_KEY set  → Groq llama-3.3-70b-versatile  (free, preferred)
      2. GOOGLE_API_KEY set → Gemini 2.0 Flash             (fallback)
      3. Neither set        → RAGAS default (OpenAI — will error if no key)

    Designed for Google Colab.  Run Phase A (collect) locally first.

    Returns
    -------
    pandas.DataFrame — per-question RAGAS scores
    """
    from datasets import Dataset  # type: ignore
    from ragas import evaluate  # type: ignore
    from ragas.metrics import (  # type: ignore
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.llms import LangchainLLMWrapper  # type: ignore
    from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore

    intermediate_path = Path(intermediate_path)
    if not intermediate_path.exists():
        raise FileNotFoundError(
            f"Intermediate file not found at {intermediate_path}.\n"
            "Run Phase A first:  python app/evaluation/metrics.py --phase collect"
        )

    with open(intermediate_path, encoding="utf-8") as f:
        data = json.load(f)

    dataset = Dataset.from_dict({
        "question":    data["questions"],
        "answer":      data["answers"],
        "contexts":    data["contexts"],
        "ground_truth": data["ground_truths"],
    })

    # ── Configure judge LLM + embeddings ────────────────────────────────
    ragas_llm: Any = None
    ragas_emb: Any = None
    groq_key   = os.getenv("GROQ_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if groq_key:
        from langchain_groq import ChatGroq  # type: ignore
        try:
            from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore  # noqa: F401
        # llama-3.1-8b-instant: 500K TPD (5× more than 70b) — ideal for RAGAS judging
        ragas_llm = LangchainLLMWrapper(
            ChatGroq(model="llama-3.1-8b-instant", api_key=groq_key, temperature=0)
        )
        ragas_emb = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )
        logger.info("RAGAS judge: Groq llama-3.1-8b-instant (500K TPD) + all-MiniLM-L6-v2 embeddings")
    elif google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  # type: ignore
        ragas_llm = LangchainLLMWrapper(
            ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_key, temperature=0)
        )
        ragas_emb = LangchainEmbeddingsWrapper(
            GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=google_key)
        )
        logger.info("RAGAS judge: Gemini 2.0 Flash + text-embedding-004")
    else:
        logger.warning(
            "No GROQ_API_KEY or GOOGLE_API_KEY found — "
            "RAGAS will attempt OpenAI default (will fail without OPENAI_API_KEY)."
        )

    from ragas import RunConfig  # type: ignore

    # max_workers=2  → sequential-ish calls; avoids burst that triggers TPM/TPD limits
    # timeout=120    → give each LLM call 2 min before giving up
    run_cfg = RunConfig(timeout=120, max_workers=2)

    logger.info("Running RAGAS evaluation on %d questions...", len(data["questions"]))
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=run_cfg,
    )
    logger.info("RAGAS complete.")
    return result.to_pandas()


# ===========================================================================
# Report generation
# ===========================================================================

def generate_report(
    ragas_df: Any,
    latencies: list[float],
    guardrail_results: list[dict],
    topic_map: list[str] | None = None,
) -> str:
    """Generate evaluation_report.md and return the markdown string."""
    import numpy as np

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = [
        "# Evaluation Report — DIP AI Tutor",
        f"\n_Generated: {ts}_\n",
    ]

    # Overall scores
    lines += ["\n## Overall RAGAS Scores\n",
              "| Metric | Score | Target | Status |",
              "|--------|-------|--------|--------|"]
    for m in metrics:
        if m in ragas_df.columns:
            score  = ragas_df[m].mean()
            target = _RAGAS_TARGETS[m]
            status = "✅" if score >= target else "❌"
            lines.append(f"| {m} | {score:.3f} | {target:.1f} | {status} |")

    # Per-topic breakdown
    if topic_map and "faithfulness" in ragas_df.columns and "answer_relevancy" in ragas_df.columns:
        lines += ["\n## Per-Topic Breakdown\n",
                  "| Topic | Faithfulness | Answer Relevancy |",
                  "|-------|-------------|-----------------|"]
        df = ragas_df.copy()
        df["topic"] = topic_map[: len(df)]
        for topic, grp in df.groupby("topic"):
            lines.append(f"| {str(topic)[:60]} | {grp['faithfulness'].mean():.3f} | {grp['answer_relevancy'].mean():.3f} |")

    # Latency
    lines += ["\n## Latency Analysis\n"]
    if latencies:
        arr  = sorted(latencies)
        mean_l = float(np.mean(arr))
        p50    = float(np.percentile(arr, 50))
        p95    = float(np.percentile(arr, 95))
        flag   = " ⚠️ **exceeds 5.0s target**" if p95 > 5.0 else " ✅"
        lines += [f"- **Mean**: {mean_l:.2f}s",
                  f"- **p50 (median)**: {p50:.2f}s",
                  f"- **p95**: {p95:.2f}s{flag}"]

    # Guardrail
    lines += ["\n## Guardrail Test Results\n",
              "| Question | Status | Answer Preview |",
              "|----------|--------|----------------|"]
    for g in guardrail_results:
        lines.append(f"| {g['question'][:50]} | {g['status']} | {g['answer'][:80].replace(chr(10),' ')} |")

    # Failed cases
    lines += ["\n## Failed Cases (any RAGAS score < 0.7)\n"]
    failed_any = False
    for i, row in ragas_df.iterrows():
        bad = [m for m in metrics if m in row and row[m] < 0.7]
        if bad:
            failed_any = True
            lines.append(f"**Q{i+1}**: {str(row.get('question', ''))[:80]}")
            for m in bad:
                lines.append(f"  - `{m}`: {row[m]:.3f} (target 0.7)")
    if not failed_any:
        lines.append("_No failed cases — all questions meet the 0.7 threshold._ ✅")

    # Recommendations
    lines += ["\n## Recommendations\n"]
    recs: list[str] = []
    if "faithfulness" in ragas_df.columns and ragas_df["faithfulness"].mean() < 0.7:
        recs.append("**Faithfulness low**: Tighten the RAG prompt — add *'Answer only using the provided context.'*")
    if "answer_relevancy" in ragas_df.columns and ragas_df["answer_relevancy"].mean() < 0.7:
        recs.append("**Answer relevancy low**: Tune retriever k or use step-back prompting to focus answers.")
    if "context_precision" in ragas_df.columns and ragas_df["context_precision"].mean() < 0.7:
        recs.append("**Context precision low**: Reduce retriever k or increase the guardrail threshold.")
    if "context_recall" in ragas_df.columns and ragas_df["context_recall"].mean() < 0.7:
        recs.append("**Context recall low**: Increase retriever k/fetch_k or re-ingest with smaller chunk_size.")
    if latencies and float(np.percentile(latencies, 95)) > 5.0:
        recs.append("**Latency p95 > 5.0s**: Switch to Ollama (local) or reduce retriever k.")
    if recs:
        for r in recs:
            lines.append(f"- {r}")
    else:
        lines.append("_All metrics meet targets. No immediate action required._ ✅")

    report_md = "\n".join(lines)
    _REPORT_FILE.write_text(report_md, encoding="utf-8")
    logger.info("Report saved -> %s", _REPORT_FILE)
    return report_md


# ===========================================================================
# CLI entry-point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIP AI Tutor evaluation pipeline.")
    parser.add_argument(
        "--phase", choices=["collect", "score"], required=True,
        help="collect=Phase A (local) | score=Phase B (Colab)",
    )
    parser.add_argument("--backend-url", default=os.getenv("EVAL_BACKEND_URL", "http://localhost:8000"))
    parser.add_argument("--intermediate", default=str(_INTERMEDIATE_FILE))
    args = parser.parse_args()

    if args.phase == "collect":
        data = collect_answers()
        gp = sum(1 for g in data["guardrail_results"] if g["passed"] is True)
        print(f"\n  DIP questions : {len(data['questions'])}")
        print(f"  Guardrail PASS: {gp} / {len(data['guardrail_results'])}")
        print(f"  Mean latency  : {sum(data['latencies'])/max(len(data['latencies']),1):.2f}s")
        print(f"  Saved         -> {_INTERMEDIATE_FILE}")

    elif args.phase == "score":
        with open(args.intermediate, encoding="utf-8") as f:
            intermediate = json.load(f)
        ragas_df = run_ragas_scoring(args.intermediate)
        report = generate_report(
            ragas_df,
            latencies=intermediate.get("latencies", []),
            guardrail_results=intermediate.get("guardrail_results", []),
            topic_map=intermediate.get("topic_map"),
        )
        print("\n" + "=" * 60)
        print(report)
        print(f"\n  Report saved -> {_REPORT_FILE}")

