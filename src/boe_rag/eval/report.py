"""Render retriever-comparison reports as Markdown.

Shared by the ablation scripts so every before/after table has the same shape:
a metadata block followed by one metrics row per retriever.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from boe_rag.eval.metrics import RetrievalMetrics


def render_comparison_report(
    title: str,
    results: Mapping[str, RetrievalMetrics],
    *,
    meta: Sequence[tuple[str, str]],
    k: int,
) -> str:
    """Render a retriever comparison as a Markdown report.

    Args:
        title: Report heading.
        results: Ordered mapping of retriever name to its metrics.
        meta: ``(label, value)`` pairs rendered as a leading bullet list.
        k: The cut-off the ``@k`` metrics were computed at.

    Returns:
        The full Markdown report as a single string.
    """
    lines = [f"# {title}", ""]
    lines += [f"- **{label}:** {value}" for label, value in meta]
    lines += [
        "",
        f"## Metrics @{k}",
        "",
        "| Retriever | Recall | Precision | Hit rate | MRR | nDCG |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name} | {m.recall_at_k:.3f} | {m.precision_at_k:.3f} "
            f"| {m.hit_rate_at_k:.3f} | {m.mrr:.3f} | {m.ndcg_at_k:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)
