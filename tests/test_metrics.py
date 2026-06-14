"""Tests for the retrieval ranking metrics."""

from __future__ import annotations

import math

import pytest

from boe_rag.eval.metrics import (
    evaluate_retrieval,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k() -> None:
    """Precision counts relevant hits in the top-k over k."""
    ranked = ["a", "b", "c", "d"]
    assert precision_at_k(ranked, {"a", "c"}, 4) == 0.5
    assert precision_at_k(ranked, {"a"}, 2) == 0.5
    assert precision_at_k(ranked, {"a"}, 0) == 0.0


def test_recall_at_k() -> None:
    """Recall counts relevant items found within the top-k over all relevant."""
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, {"a", "x"}, 3) == 0.5
    assert recall_at_k(ranked, {"a", "b"}, 3) == 1.0
    assert recall_at_k(ranked, set(), 3) == 0.0


def test_hit_rate_at_k() -> None:
    """Hit rate is 1 when any relevant item is in the top-k, else 0."""
    ranked = ["a", "b", "c"]
    assert hit_rate_at_k(ranked, {"c"}, 3) == 1.0
    assert hit_rate_at_k(ranked, {"c"}, 2) == 0.0
    assert hit_rate_at_k(ranked, {"z"}, 3) == 0.0


def test_reciprocal_rank() -> None:
    """Reciprocal rank is 1 over the rank of the first relevant item."""
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_ndcg_at_k_perfect_and_partial() -> None:
    """Ideal rankings score 1, while lower-ranked hits are discounted."""
    # Single relevant item at rank 1 -> perfect.
    assert ndcg_at_k(["a", "b"], {"a"}, 2) == 1.0
    # Single relevant item at rank 2 -> 1/log2(3) normalised by ideal (1.0).
    expected = (1.0 / math.log2(3)) / 1.0
    assert ndcg_at_k(["a", "b"], {"b"}, 2) == pytest.approx(expected)
    assert ndcg_at_k(["a"], set(), 2) == 0.0


def test_evaluate_retrieval_aggregates() -> None:
    """Aggregate metrics average correctly across queries."""
    results = [
        (["a", "b", "c"], {"a"}),  # hit@1
        (["x", "y", "z"], {"y"}),  # hit@2
    ]
    metrics = evaluate_retrieval(results, k=3)
    assert metrics.num_queries == 2
    assert metrics.hit_rate_at_k == 1.0
    assert metrics.mrr == pytest.approx((1.0 + 0.5) / 2)
    assert metrics.recall_at_k == 1.0


def test_evaluate_retrieval_empty() -> None:
    """An empty eval set yields zeroed metrics, not an error."""
    metrics = evaluate_retrieval([], k=5)
    assert metrics.num_queries == 0
    assert metrics.recall_at_k == 0.0


def test_evaluate_retrieval_rejects_bad_k() -> None:
    """A non-positive k is rejected."""
    with pytest.raises(ValueError, match="k must be positive"):
        evaluate_retrieval([(["a"], {"a"})], k=0)
