"""Compare two retrievers on the same gold set with paired significance.

A fine-tuned embedder is only worth shipping if it beats the baseline by more
than sampling noise on a 20-question gold set. This module turns two retrieval
runs over the *same* questions into an honest verdict: per-query recall@k and
reciprocal-rank series, a paired-bootstrap CI and sign-flip p-value for each
metric (via :mod:`boe_rag.eval.stats`), and a conservative ship/no-ship call.

It is pure — it consumes the :class:`~boe_rag.eval.runner.ExampleResult` lists the
eval runner already produces, so the decision logic is unit-tested without models
or a GPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from boe_rag.eval.metrics import recall_at_k, reciprocal_rank
from boe_rag.eval.runner import ExampleResult
from boe_rag.eval.stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DeltaSignificance,
    paired_delta_significance,
)

#: Default significance threshold below which an improvement is "real".
DEFAULT_ALPHA = 0.05


def per_query_recall_and_rr(
    results: Sequence[ExampleResult], k: int
) -> tuple[list[float], list[float]]:
    """Recompute the per-query recall@k and reciprocal-rank series.

    Args:
        results: Per-example retrieval outcomes from the eval runner.
        k: Cut-off rank for recall@k.

    Returns:
        Two equal-length lists, ``(recalls, reciprocal_ranks)``, in example order.
    """
    recalls = [recall_at_k(r.retrieved_ids, r.relevant_ids, k) for r in results]
    rrs = [reciprocal_rank(r.retrieved_ids, r.relevant_ids) for r in results]
    return recalls, rrs


@dataclass(frozen=True, slots=True)
class ModelComparison:
    """Paired baseline-vs-candidate verdict on the gold set.

    Attributes:
        k: Cut-off rank for the recall metric.
        recall: Paired significance of the recall@k difference.
        mrr: Paired significance of the reciprocal-rank (MRR) difference.
        num_queries: Number of gold queries both systems were scored on.
    """

    k: int
    recall: DeltaSignificance
    mrr: DeltaSignificance
    num_queries: int


def compare_models(
    baseline: Sequence[ExampleResult],
    candidate: Sequence[ExampleResult],
    *,
    k: int = 10,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> ModelComparison:
    """Compare two retrievers scored on the same, aligned gold examples.

    Args:
        baseline: Baseline retriever's per-example results.
        candidate: Candidate retriever's per-example results, aligned one-to-one
            with ``baseline`` (same example ids, same order).
        k: Cut-off rank for recall@k.
        confidence: Two-sided confidence level for the CIs.
        n_resamples: Resamples for the paired bootstrap and permutation test.
        seed: Seed for reproducible resampling.

    Returns:
        The paired significance of the recall@k and MRR differences.

    Raises:
        ValueError: If the two runs are empty or not aligned by example id.
    """
    if len(baseline) != len(candidate):
        raise ValueError(f"runs differ in length: {len(baseline)} vs {len(candidate)}")
    if not baseline:
        raise ValueError("cannot compare empty runs")
    for base, cand in zip(baseline, candidate, strict=True):
        if base.example_id != cand.example_id:
            raise ValueError(
                f"runs not aligned: '{base.example_id}' vs '{cand.example_id}'"
            )

    base_recall, base_rr = per_query_recall_and_rr(baseline, k)
    cand_recall, cand_rr = per_query_recall_and_rr(candidate, k)
    recall_sig = paired_delta_significance(
        base_recall,
        cand_recall,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )
    mrr_sig = paired_delta_significance(
        base_rr, cand_rr, confidence=confidence, n_resamples=n_resamples, seed=seed
    )
    return ModelComparison(
        k=k, recall=recall_sig, mrr=mrr_sig, num_queries=len(baseline)
    )


def recommend_ship(
    comparison: ModelComparison, *, alpha: float = DEFAULT_ALPHA
) -> tuple[bool, str]:
    """Decide whether the candidate should ship, with a one-line rationale.

    Conservative rule keyed on the headline metric: ship only if recall@k
    **improves** and the improvement is significant at ``alpha`` (the paired
    p-value is below it). A flat or negative recall delta, or an improvement
    within noise, is a no-ship — the off-the-shelf model stays.

    Args:
        comparison: The paired comparison verdict.
        alpha: Significance threshold for the recall improvement.

    Returns:
        ``(should_ship, reason)``.
    """
    recall = comparison.recall
    delta = recall.delta
    if delta <= 0:
        return False, f"recall@{comparison.k} did not improve (Δ={delta:+.3f})"
    if recall.p_value >= alpha:
        return (
            False,
            f"recall@{comparison.k} up Δ={delta:+.3f} but not significant "
            f"(p={recall.p_value:.3f} ≥ {alpha})",
        )
    return (
        True,
        f"recall@{comparison.k} up Δ={delta:+.3f}, significant "
        f"(p={recall.p_value:.3f} < {alpha})",
    )
