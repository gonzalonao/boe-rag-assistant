"""Tests for the retrieval regression gate (boe_rag.eval.regression)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from boe_rag.eval.regression import (
    GuardResult,
    MetricGuard,
    check_metrics,
    format_report,
    load_guards,
)

_METRICS: dict[str, float | int] = {
    "k": 10,
    "num_queries": 20,
    "recall_at_k": 0.9,
    "mrr": 0.749,
}


def test_guard_floor_is_baseline_minus_tolerance() -> None:
    """The floor is the baseline minus the allowed tolerance."""
    guard = MetricGuard(name="mrr", baseline=0.749, tolerance=0.05)
    assert guard.floor == pytest.approx(0.699)


def test_value_above_floor_passes() -> None:
    """A value above the floor clears the guard."""
    result = GuardResult(MetricGuard("mrr", 0.749, 0.05), value=0.72)
    assert result.passed


def test_value_below_floor_fails() -> None:
    """A value below the floor fails the guard."""
    result = GuardResult(MetricGuard("recall_at_k", 0.9, 0.05), value=0.84)
    assert not result.passed


def test_value_exactly_at_floor_passes() -> None:
    """The floor is inclusive (>=), so a value equal to it must pass."""
    result = GuardResult(MetricGuard("recall_at_k", 0.9, 0.05), value=0.85)
    assert result.passed


def test_check_metrics_passes_at_baseline() -> None:
    """Metrics equal to the baseline clear every guard."""
    guards = [
        MetricGuard("recall_at_k", 0.9, 0.05),
        MetricGuard("mrr", 0.749, 0.05),
    ]
    results = check_metrics(_METRICS, guards)
    assert [r.passed for r in results] == [True, True]


def test_check_metrics_flags_a_regression() -> None:
    """A metric that drops past its tolerance is flagged as failed."""
    regressed = {**_METRICS, "recall_at_k": 0.80}
    results = check_metrics(regressed, [MetricGuard("recall_at_k", 0.9, 0.05)])
    assert not results[0].passed


def test_check_metrics_raises_on_missing_metric() -> None:
    """A guarded metric absent from the report is an error, not a silent pass."""
    with pytest.raises(KeyError, match="ndcg_at_k"):
        check_metrics(_METRICS, [MetricGuard("ndcg_at_k", 0.5, 0.05)])


def test_load_guards_reads_a_baseline_file(tmp_path: Path) -> None:
    """A well-formed baseline file yields the declared guards."""
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "guards": {
                    "recall_at_k": {"baseline": 0.9, "tolerance": 0.05},
                    "mrr": {"baseline": 0.749, "tolerance": 0.05},
                }
            }
        ),
        encoding="utf-8",
    )
    guards = load_guards(baseline)
    assert {g.name for g in guards} == {"recall_at_k", "mrr"}
    recall = next(g for g in guards if g.name == "recall_at_k")
    assert recall.floor == pytest.approx(0.85)


def test_load_guards_rejects_a_file_without_guards(tmp_path: Path) -> None:
    """A baseline missing the 'guards' object is rejected."""
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"k": 10}), encoding="utf-8")
    with pytest.raises(ValueError, match="guards"):
        load_guards(baseline)


def test_load_guards_rejects_a_non_object_baseline(tmp_path: Path) -> None:
    """A baseline that is not a JSON object is rejected."""
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_guards(baseline)


def test_committed_baseline_file_is_valid() -> None:
    """The repo's real baseline must parse and guard recall@k and MRR."""
    guards = load_guards(Path("eval_data/retrieval_baseline.json"))
    assert {g.name for g in guards} == {"recall_at_k", "mrr"}


def test_format_report_marks_pass_and_fail() -> None:
    """The rendered report shows PASS for held metrics and FAIL for regressed."""
    results = check_metrics(
        {"recall_at_k": 0.70, "mrr": 0.80},
        [MetricGuard("recall_at_k", 0.9, 0.05), MetricGuard("mrr", 0.749, 0.05)],
    )
    report = format_report(results)
    assert "FAIL" in report  # recall regressed
    assert "PASS" in report  # mrr held
