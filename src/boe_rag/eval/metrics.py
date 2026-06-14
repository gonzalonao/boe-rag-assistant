"""Information-retrieval ranking metrics.

Pure-Python, dependency-free implementations of the standard metrics used to
score a retriever against a golden set with binary relevance judgments. Kept
free of any model or I/O dependency so they run instantly in CI and form the
contract every retrieval change is measured against.

All per-query functions take the retrieved chunk ids in rank order and the set
of relevant chunk ids for that query.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass


def precision_at_k(
    ranked_ids: Sequence[str], relevant_ids: Collection[str], k: int
) -> float:
    """Fraction of the top-k results that are relevant.

    Args:
        ranked_ids: Retrieved chunk ids in descending score order.
        relevant_ids: Ground-truth relevant chunk ids.
        k: Cut-off rank.

    Returns:
        Precision@k in ``[0, 1]`` (0 when ``k <= 0``).
    """
    if k <= 0:
        return 0.0
    top = ranked_ids[:k]
    relevant = set(relevant_ids)
    hits = sum(1 for cid in top if cid in relevant)
    return hits / k


def recall_at_k(
    ranked_ids: Sequence[str], relevant_ids: Collection[str], k: int
) -> float:
    """Fraction of relevant items found within the top-k results.

    Args:
        ranked_ids: Retrieved chunk ids in descending score order.
        relevant_ids: Ground-truth relevant chunk ids.
        k: Cut-off rank.

    Returns:
        Recall@k in ``[0, 1]`` (0 when there are no relevant items).
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    top = set(ranked_ids[:k])
    return len(top & relevant) / len(relevant)


def hit_rate_at_k(
    ranked_ids: Sequence[str], relevant_ids: Collection[str], k: int
) -> float:
    """Whether at least one relevant item appears in the top-k results.

    Args:
        ranked_ids: Retrieved chunk ids in descending score order.
        relevant_ids: Ground-truth relevant chunk ids.
        k: Cut-off rank.

    Returns:
        ``1.0`` if any relevant item is in the top-k, else ``0.0``.
    """
    relevant = set(relevant_ids)
    return 1.0 if any(cid in relevant for cid in ranked_ids[:k]) else 0.0


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Collection[str]) -> float:
    """Reciprocal of the rank of the first relevant item.

    Args:
        ranked_ids: Retrieved chunk ids in descending score order.
        relevant_ids: Ground-truth relevant chunk ids.

    Returns:
        ``1 / rank`` of the first relevant hit (rank starting at 1), or ``0.0``
        if no relevant item was retrieved.
    """
    relevant = set(relevant_ids)
    for index, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(
    ranked_ids: Sequence[str], relevant_ids: Collection[str], k: int
) -> float:
    """Normalised discounted cumulative gain at k (binary relevance).

    Args:
        ranked_ids: Retrieved chunk ids in descending score order.
        relevant_ids: Ground-truth relevant chunk ids.
        k: Cut-off rank.

    Returns:
        nDCG@k in ``[0, 1]``; 0 when there are no relevant items.
    """
    relevant = set(relevant_ids)
    if not relevant or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(index + 1)
        for index, cid in enumerate(ranked_ids[:k], start=1)
        if cid in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Mean retrieval metrics aggregated over an evaluation set.

    Attributes:
        k: The primary cut-off rank these metrics were computed at.
        num_queries: Number of queries evaluated.
        recall_at_k: Mean recall@k.
        precision_at_k: Mean precision@k.
        hit_rate_at_k: Mean hit rate@k (a.k.a. success@k).
        mrr: Mean reciprocal rank (over the full ranking, not truncated).
        ndcg_at_k: Mean nDCG@k.
    """

    k: int
    num_queries: int
    recall_at_k: float
    precision_at_k: float
    hit_rate_at_k: float
    mrr: float
    ndcg_at_k: float

    def as_dict(self) -> dict[str, float | int]:
        """Return the metrics as a plain dict (for JSON/report serialisation)."""
        return asdict(self)


def evaluate_retrieval(
    results: Sequence[tuple[Sequence[str], Collection[str]]],
    k: int = 10,
) -> RetrievalMetrics:
    """Aggregate per-query metrics over an evaluation set.

    Args:
        results: One ``(ranked_ids, relevant_ids)`` pair per query.
        k: Cut-off rank for the @k metrics.

    Returns:
        The mean metrics across all queries (zeros when ``results`` is empty).

    Raises:
        ValueError: If ``k`` is not positive.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    n = len(results)
    if n == 0:
        return RetrievalMetrics(k, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    recall = sum(recall_at_k(r, rel, k) for r, rel in results) / n
    precision = sum(precision_at_k(r, rel, k) for r, rel in results) / n
    hit = sum(hit_rate_at_k(r, rel, k) for r, rel in results) / n
    mrr = sum(reciprocal_rank(r, rel) for r, rel in results) / n
    ndcg = sum(ndcg_at_k(r, rel, k) for r, rel in results) / n
    return RetrievalMetrics(k, n, recall, precision, hit, mrr, ndcg)
