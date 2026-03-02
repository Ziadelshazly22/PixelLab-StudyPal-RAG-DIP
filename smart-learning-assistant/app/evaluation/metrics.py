"""
app/evaluation/metrics.py
--------------------------
RAG evaluation utilities using RAGAS and ROUGE.

Metrics computed:
  - Faithfulness      – answer grounded in retrieved context
  - Answer Relevancy  – answer addresses the question
  - Context Recall    – relevant context retrieved
  - ROUGE-L           – overlap between answer and reference
"""

from __future__ import annotations

from typing import Any


def evaluate_rag(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] | None = None,
) -> dict[str, Any]:
    """
    Evaluate a set of RAG responses with RAGAS + ROUGE.

    Parameters
    ----------
    questions : list[str]
        Input questions.
    answers : list[str]
        Generated answers from the RAG chain.
    contexts : list[list[str]]
        Retrieved context chunks for each question.
    ground_truths : list[str], optional
        Reference answers (required for context_recall and ROUGE).

    Returns
    -------
    dict
        Dictionary of metric names → scores.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_recall,
        faithfulness,
    )
    from rouge_score import rouge_scorer

    data: dict[str, list] = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    if ground_truths:
        data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(data)
    metrics = [faithfulness, answer_relevancy]
    if ground_truths:
        metrics.append(context_recall)

    ragas_results = evaluate(dataset, metrics=metrics)
    scores: dict[str, Any] = dict(ragas_results)

    # ROUGE-L (only when ground truths provided)
    if ground_truths:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_scores = [
            scorer.score(ref, hyp)["rougeL"].fmeasure
            for ref, hyp in zip(ground_truths, answers)
        ]
        scores["rouge_l"] = sum(rouge_scores) / len(rouge_scores)

    return scores
