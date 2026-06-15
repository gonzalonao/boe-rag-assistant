"""Tests for the UI presentation helpers (no Gradio import required)."""

from __future__ import annotations

from boe_rag.service.models import Source
from boe_rag.service.ui import format_sources, render_quality_markdown


def _source(text: str = "El tipo general del IVA es del 21%.") -> Source:
    return Source(
        chunk_id="c1",
        citation="Ley 1/2024, Artículo 1",
        text=text,
        url="https://www.boe.es/diario_boe/txt.php?id=BOE-A-2024-1",
        score=1.0,
    )


def test_format_sources_empty_returns_blank() -> None:
    """No sources renders to an empty string (nothing appended to the answer)."""
    assert format_sources([]) == ""


def test_format_sources_renders_numbered_links() -> None:
    """Each source becomes a numbered Markdown link with its citation and text."""
    rendered = format_sources([_source(), _source()])
    assert "**Sources**" in rendered
    assert "1. [Ley 1/2024, Artículo 1](https://www.boe.es" in rendered
    assert "2. [Ley 1/2024, Artículo 1]" in rendered
    assert "21%" in rendered


def test_format_sources_truncates_long_passages() -> None:
    """Long passages are truncated with an ellipsis to keep the chat readable."""
    rendered = format_sources([_source("palabra " * 100)])
    assert "…" in rendered
    assert len(rendered) < len("palabra " * 100) + 200


def test_render_quality_markdown_includes_metric_tables() -> None:
    """Both retrieval and end-to-end metrics appear in the rendered report."""
    retrieval = {
        "dense": {
            "k": 10,
            "num_queries": 20,
            "recall_at_k": 0.9,
            "hit_rate_at_k": 0.9,
            "mrr": 0.749,
            "ndcg_at_k": 0.783,
        },
        "hybrid + cross-encoder": {
            "k": 10,
            "num_queries": 20,
            "recall_at_k": 1.0,
            "hit_rate_at_k": 1.0,
            "mrr": 0.888,
            "ndcg_at_k": 0.913,
        },
    }
    e2e = {"mean_faithfulness": 0.99, "mean_correctness": 0.895, "refusal_rate": 0.05}

    rendered = render_quality_markdown(retrieval, e2e)

    assert "golden set of 20 questions" in rendered
    assert "k=10" in rendered
    assert "| dense | 0.900 | 0.900 | 0.749 | 0.783 |" in rendered
    assert "| hybrid + cross-encoder | 1.000 | 1.000 | 0.888 | 0.913 |" in rendered
    assert "| 0.990 | 0.895 | 0.050 |" in rendered


def test_render_quality_markdown_handles_missing_reports() -> None:
    """Missing report data degrades to an explicit note, not a crash."""
    rendered = render_quality_markdown({}, {})
    assert "Retrieval metrics unavailable" in rendered
    assert "End-to-end metrics unavailable" in rendered
