# -*- coding: utf-8 -*-
"""
tests/test_metrics.py
---------------------
Unit tests for app/evaluation/metrics.py.

All tests are fully offline — no HTTP, no LLM, no disk IO beyond tmp_path.

Test inventory:
  1. test_collect_answers_file_not_found       — raises FileNotFoundError
  2. test_collect_answers_connection_refused   — returns empty answers gracefully
  3. test_generate_report_contains_sections   — report MD has all required headers
  4. test_generate_report_all_pass            — all-green report has no ❌
  5. test_generate_report_failed_case         — below-threshold score creates ❌
  6. test_generate_report_written_to_disk     — file is written to _REPORT_FILE path
  7. test_refusal_phrase_detected             — guardrail detection helper works
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ragas_df(faithfulness: float = 0.9, answer_relevancy: float = 0.9,
                   context_precision: float = 0.9, context_recall: float = 0.9):
    """Build a minimal pandas DataFrame with RAGAS metric columns."""
    import pandas as pd
    return pd.DataFrame({
        "question":          ["What is histogram equalization?"],
        "answer":            ["Histogram equalization is..."],
        "faithfulness":      [faithfulness],
        "answer_relevancy":  [answer_relevancy],
        "context_precision": [context_precision],
        "context_recall":    [context_recall],
    })


# ===========================================================================
# 1. collect_answers — raises FileNotFoundError for missing questions file
# ===========================================================================
def test_collect_answers_file_not_found():
    """collect_answers must raise FileNotFoundError for a non-existent path."""
    from app.evaluation.metrics import collect_answers

    with pytest.raises(FileNotFoundError):
        collect_answers(questions_path="/nonexistent/path/questions.json")


# ===========================================================================
# 2. collect_answers — ConnectionRefused → partial result (no crash)
# ===========================================================================
def test_collect_answers_connection_refused(tmp_path, monkeypatch):
    """collect_answers must handle a refused connection gracefully, not raise."""
    from app.evaluation import metrics as _metrics

    # Write a minimal questions file
    questions = [
        {
            "question": "What is edge detection?",
            "ground_truth": "Edge detection finds boundaries.",
            "topic": "edge_detection",
            "is_off_topic": False,
        }
    ]
    q_file = tmp_path / "test_questions.json"
    q_file.write_text(json.dumps(questions), encoding="utf-8")

    # Patch _preflight_quota_check to return True (skip real HTTP probe)
    monkeypatch.setattr(_metrics, "_preflight_quota_check", lambda: True)

    # Patch requests.post to raise ConnectionError
    import requests as _requests
    monkeypatch.setattr(
        _requests,
        "post",
        MagicMock(side_effect=_requests.exceptions.ConnectionError("refused")),
    )

    # Patch the intermediate file write to go into tmp_path
    monkeypatch.setattr(_metrics, "_INTERMEDIATE_FILE", tmp_path / "intermediate.json")

    result = _metrics.collect_answers(questions_path=q_file)

    # Should have one entry with empty answer (connection failed, but no crash)
    assert "answers" in result
    assert len(result["answers"]) == 1
    assert result["answers"][0] == "", (
        "Expected empty string answer on ConnectionError, got: "
        + repr(result["answers"][0])
    )


# ===========================================================================
# 3. generate_report — report contains all mandatory section headers
# ===========================================================================
def test_generate_report_contains_sections(tmp_path, monkeypatch):
    """generate_report must produce a Markdown string with all required sections."""
    from app.evaluation import metrics as _metrics

    # Route the file write to tmp_path
    monkeypatch.setattr(_metrics, "_REPORT_FILE", tmp_path / "evaluation_report.md")

    df = _make_ragas_df()
    report = _metrics.generate_report(
        ragas_df=df,
        latencies=[2.5, 3.1, 2.8],
        guardrail_results=[
            {"question": "What is tiramisu?", "answer": "out of focus", "passed": True, "status": "PASS"},
        ],
        topic_map=["histogram_equalization"],
    )

    required_sections = [
        "## Overall RAGAS Scores",
        "## Latency Analysis",
        "## Guardrail Test Results",
    ]
    for section in required_sections:
        assert section in report, f"Section '{section}' missing from report"


# ===========================================================================
# 4. generate_report — all metrics above target → no ❌ in report
# ===========================================================================
def test_generate_report_all_pass(tmp_path, monkeypatch):
    """When all metrics exceed 0.7, the report should contain no ❌ markers."""
    from app.evaluation import metrics as _metrics

    monkeypatch.setattr(_metrics, "_REPORT_FILE", tmp_path / "evaluation_report.md")

    df = _make_ragas_df(0.85, 0.90, 0.95, 0.80)
    report = _metrics.generate_report(
        ragas_df=df,
        latencies=[1.5],
        guardrail_results=[],
    )

    # No ❌ in the RAGAS scores table (table body only — header may have ❌ word)
    lines_with_red_x = [l for l in report.splitlines() if "❌" in l]
    assert not lines_with_red_x, (
        f"Unexpected ❌ found in all-passing report:\n"
        + "\n".join(lines_with_red_x)
    )


# ===========================================================================
# 5. generate_report — below-target score → ❌ in report table
# ===========================================================================
def test_generate_report_failed_case(tmp_path, monkeypatch):
    """A score below 0.7 must produce a ❌ in the RAGAS scores table."""
    from app.evaluation import metrics as _metrics

    monkeypatch.setattr(_metrics, "_REPORT_FILE", tmp_path / "evaluation_report.md")

    df = _make_ragas_df(faithfulness=0.50)  # faithfulness below target
    report = _metrics.generate_report(
        ragas_df=df,
        latencies=[3.0],
        guardrail_results=[],
    )

    assert "❌" in report, "Expected ❌ for faithfulness < 0.7 but it was not found"


# ===========================================================================
# 6. generate_report — writes the report file to disk
# ===========================================================================
def test_generate_report_written_to_disk(tmp_path, monkeypatch):
    """generate_report must create evaluation_report.md at _REPORT_FILE."""
    from app.evaluation import metrics as _metrics

    report_path = tmp_path / "evaluation_report.md"
    monkeypatch.setattr(_metrics, "_REPORT_FILE", report_path)

    df = _make_ragas_df()
    _metrics.generate_report(ragas_df=df, latencies=[2.0], guardrail_results=[])

    assert report_path.exists(), "evaluation_report.md was not created on disk"
    content = report_path.read_text(encoding="utf-8")
    assert len(content) > 50, "Written report is unexpectedly short"


# ===========================================================================
# 7. _is_daily_quota_exhausted — detects the known quota error ID
# ===========================================================================
def test_refusal_phrase_detected():
    """_REFUSAL_PHRASE must match the guardrail phrase used in the prompt."""
    from app.evaluation.metrics import _REFUSAL_PHRASE

    sample_refusal = (
        "While I'd love to always help, this question falls out of focus. "
        "Let's avoid distractions and get back to pixels & matrices!"
    )
    assert _REFUSAL_PHRASE.lower() in sample_refusal.lower(), (
        f"Guardrail phrase '{_REFUSAL_PHRASE}' not found in expected refusal message"
    )
