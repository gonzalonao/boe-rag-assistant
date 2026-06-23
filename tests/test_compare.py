"""Tests for the paired model-comparison verdict logic.

Builds ExampleResult runs by hand so the ship/no-ship rule and the alignment
guards are covered without models, a GPU, or the eval harness.
"""

from __future__ import annotations

import pytest

from boe_rag.eval.compare import (
    compare_models,
    per_query_recall_and_rr,
    recommend_ship,
)
from boe_rag.eval.runner import ExampleResult


def _result(example_id: str, rank: int | None) -> ExampleResult:
    """An ExampleResult whose single relevant id sits at 1-based ``rank``.

    The retrieved tuple pads with filler ids and places the relevant id at the
    given rank (or omits it entirely, for a miss), which is all recall@k / RR
    need to recompute.
    """
    relevant = "gold"
    if rank is None:
        retrieved: tuple[str, ...] = ("a", "b", "c", "d")
    else:
        retrieved = (*(f"f{i}" for i in range(rank - 1)), relevant)
    return ExampleResult(
        example_id=example_id,
        retrieved_ids=retrieved,
        relevant_ids=(relevant,),
        first_relevant_rank=rank,
    )


def test_per_query_series_match_ranks() -> None:
    """Recall@k is 1 when the hit is within k; RR is 1/rank."""
    results = [_result("e1", 1), _result("e2", 3), _result("e3", None)]
    recalls, rrs = per_query_recall_and_rr(results, k=2)
    assert recalls == [1.0, 0.0, 0.0]  # rank 3 is outside k=2
    assert rrs == [1.0, pytest.approx(1 / 3), 0.0]


def test_compare_flags_significant_recall_gain_as_ship() -> None:
    """A candidate that fixes every miss is a significant, shippable win."""
    baseline = [_result(f"e{i}", None) for i in range(12)]
    candidate = [_result(f"e{i}", 1) for i in range(12)]
    comparison = compare_models(baseline, candidate, k=10, n_resamples=2000)

    assert comparison.recall.delta == pytest.approx(1.0)
    assert comparison.num_queries == 12
    ship, reason = recommend_ship(comparison)
    assert ship is True
    assert "significant" in reason


def test_compare_no_change_is_no_ship() -> None:
    """Identical runs give a zero delta and a no-ship verdict."""
    runs = [_result(f"e{i}", 1) for i in range(10)]
    comparison = compare_models(runs, list(runs), k=10, n_resamples=2000)
    ship, reason = recommend_ship(comparison)
    assert ship is False
    assert "did not improve" in reason


def test_compare_regression_is_no_ship() -> None:
    """A candidate that is worse on recall never ships."""
    baseline = [_result(f"e{i}", 1) for i in range(10)]
    candidate = [_result(f"e{i}", None) for i in range(10)]
    comparison = compare_models(baseline, candidate, k=10, n_resamples=2000)
    ship, _ = recommend_ship(comparison)
    assert ship is False


def test_compare_rejects_misaligned_runs() -> None:
    """Runs whose example ids do not line up are rejected."""
    baseline = [_result("a", 1)]
    candidate = [_result("b", 1)]
    with pytest.raises(ValueError, match="not aligned"):
        compare_models(baseline, candidate, k=10)


def test_compare_rejects_empty_runs() -> None:
    """Empty runs cannot be compared."""
    with pytest.raises(ValueError, match="empty"):
        compare_models([], [], k=10)
