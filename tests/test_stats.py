"""Tests for the bootstrap-CI and paired-significance helpers."""

from __future__ import annotations

import pytest

from boe_rag.eval.stats import (
    BootstrapCI,
    DeltaSignificance,
    bootstrap_mean_ci,
    paired_delta_significance,
)

# Small resample counts keep the tests fast while staying stable under a fixed seed.
_RESAMPLES = 2_000


def test_bootstrap_point_is_the_sample_mean() -> None:
    """The reported point estimate is exactly the sample mean."""
    ci = bootstrap_mean_ci([0.0, 0.5, 1.0], n_resamples=_RESAMPLES)
    assert ci.point == pytest.approx(0.5)


def test_bootstrap_interval_brackets_the_point() -> None:
    """The interval contains the point estimate and stores the confidence."""
    ci = bootstrap_mean_ci([0.2, 0.4, 0.6, 0.8, 1.0], n_resamples=_RESAMPLES)
    assert ci.low <= ci.point <= ci.high
    assert ci.confidence == 0.95


def test_bootstrap_is_deterministic_under_a_seed() -> None:
    """Same data and seed reproduce the same interval exactly."""
    values = [0.1, 0.9, 0.3, 0.7, 0.5]
    first = bootstrap_mean_ci(values, n_resamples=_RESAMPLES, seed=7)
    second = bootstrap_mean_ci(values, n_resamples=_RESAMPLES, seed=7)
    assert first == second


def test_bootstrap_constant_data_has_zero_width() -> None:
    """With no variance the interval collapses onto the point."""
    ci = bootstrap_mean_ci([0.5] * 50, n_resamples=_RESAMPLES)
    assert ci.low == pytest.approx(0.5)
    assert ci.high == pytest.approx(0.5)


def test_bootstrap_more_spread_widens_the_interval() -> None:
    """A higher-variance series (same mean) yields a wider interval."""
    low_var = bootstrap_mean_ci([0.4, 0.6] * 50, n_resamples=_RESAMPLES, seed=1)
    high_var = bootstrap_mean_ci([0.0, 1.0] * 50, n_resamples=_RESAMPLES, seed=1)
    assert low_var.point == pytest.approx(high_var.point)
    assert (high_var.high - high_var.low) > (low_var.high - low_var.low)


def test_bootstrap_as_dict_round_trips() -> None:
    """``as_dict`` exposes the four interval fields."""
    ci = bootstrap_mean_ci([0.0, 1.0], n_resamples=_RESAMPLES)
    assert ci.as_dict() == {
        "point": ci.point,
        "low": ci.low,
        "high": ci.high,
        "confidence": ci.confidence,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence": 0.0},
        {"confidence": 1.0},
        {"n_resamples": 0},
    ],
)
def test_bootstrap_rejects_bad_parameters(kwargs: dict[str, float]) -> None:
    """Out-of-range confidence or resample count raises ``ValueError``."""
    with pytest.raises(ValueError):
        bootstrap_mean_ci([0.1, 0.2, 0.3], **kwargs)  # type: ignore[arg-type]


def test_bootstrap_rejects_empty_values() -> None:
    """An empty series has no mean to bound."""
    with pytest.raises(ValueError):
        bootstrap_mean_ci([], n_resamples=_RESAMPLES)


def test_delta_identical_series_is_not_significant() -> None:
    """Two identical series give a zero difference and a p-value of 1."""
    series = [0.0, 0.5, 1.0, 0.25, 0.75]
    result = paired_delta_significance(series, series, n_resamples=_RESAMPLES)
    assert result.delta == pytest.approx(0.0)
    assert result.low == pytest.approx(0.0)
    assert result.high == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)


def test_delta_constant_improvement_is_significant() -> None:
    """A uniform improvement is detected: positive delta, tiny p, CI excludes 0."""
    baseline = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    candidate = [v + 0.5 for v in baseline]
    result = paired_delta_significance(baseline, candidate, n_resamples=_RESAMPLES)
    assert result.delta == pytest.approx(0.5)
    assert result.low > 0.0
    assert result.p_value < 0.05


def test_delta_sign_reflects_direction() -> None:
    """A worse candidate yields a negative mean difference."""
    baseline = [0.6, 0.7, 0.8, 0.9, 1.0]
    candidate = [0.1, 0.2, 0.3, 0.4, 0.5]
    result = paired_delta_significance(baseline, candidate, n_resamples=_RESAMPLES)
    assert result.delta < 0.0


def test_delta_is_deterministic_under_a_seed() -> None:
    """Same inputs and seed reproduce the same result."""
    a = [0.1, 0.4, 0.2, 0.9, 0.5]
    b = [0.3, 0.3, 0.6, 0.8, 0.7]
    first = paired_delta_significance(a, b, n_resamples=_RESAMPLES, seed=3)
    second = paired_delta_significance(a, b, n_resamples=_RESAMPLES, seed=3)
    assert first == second


def test_delta_rejects_mismatched_lengths() -> None:
    """Paired series must align one-to-one."""
    with pytest.raises(ValueError):
        paired_delta_significance([0.1, 0.2], [0.1], n_resamples=_RESAMPLES)


def test_delta_rejects_empty_series() -> None:
    """Empty series have nothing to compare."""
    with pytest.raises(ValueError):
        paired_delta_significance([], [], n_resamples=_RESAMPLES)


def test_delta_as_dict_round_trips() -> None:
    """``as_dict`` exposes all six result fields."""
    result = paired_delta_significance(
        [0.1, 0.2, 0.3], [0.2, 0.3, 0.4], n_resamples=_RESAMPLES
    )
    assert set(result.as_dict()) == {
        "delta",
        "low",
        "high",
        "p_value",
        "confidence",
        "n_resamples",
    }


def test_result_types() -> None:
    """The helpers return their documented dataclasses."""
    assert isinstance(
        bootstrap_mean_ci([0.5, 0.5], n_resamples=_RESAMPLES), BootstrapCI
    )
    assert isinstance(
        paired_delta_significance([0.1], [0.2], n_resamples=_RESAMPLES),
        DeltaSignificance,
    )
