"""Tests for the shared comparison-report renderer."""

from __future__ import annotations

from boe_rag.eval.metrics import RetrievalMetrics
from boe_rag.eval.report import render_comparison_report


def _metrics(recall: float) -> RetrievalMetrics:
    return RetrievalMetrics(
        k=10,
        num_queries=20,
        recall_at_k=recall,
        precision_at_k=0.09,
        hit_rate_at_k=0.9,
        mrr=0.75,
        ndcg_at_k=0.78,
    )


def test_render_includes_title_meta_and_rows() -> None:
    """The report contains the title, metadata bullets, and one row per retriever."""
    report = render_comparison_report(
        "Retrieval evaluation",
        {"dense": _metrics(0.900), "hybrid": _metrics(0.925)},
        meta=[("Queries", "20")],
        k=10,
    )
    assert "# Retrieval evaluation" in report
    assert "- **Queries:** 20" in report
    assert "| dense | 0.900 |" in report
    assert "| hybrid | 0.925 |" in report
    assert "## Metrics @10" in report
