"""Statistical tooling for evaluation: bootstrap CIs and paired significance.

Point-estimate metrics (recall@k, MRR, ...) hide how *certain* they are: on a
20-question gold set a 0.05 gap can be noise. These helpers put error bars on a
metric and test whether the difference between two systems is real, so a
retrieval change can be judged with the same rigor as the rest of the pipeline —
not just "the number went up".

Both procedures are non-parametric and make no normality assumption (apt for
bounded, skewed metrics like recall):

- :func:`bootstrap_mean_ci` — a percentile bootstrap confidence interval for the
  mean of a per-query metric series.
- :func:`paired_delta_significance` — for two systems scored on the *same*
  queries, a paired-bootstrap CI for the mean difference plus a two-sided
  sign-flip permutation test for its p-value.

Resampling is seeded, so every reported interval is reproducible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

#: Default number of resamples; 10k gives stable percentile and permutation estimates.
DEFAULT_RESAMPLES = 10_000
#: Default two-sided confidence level for intervals.
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    """A bootstrap confidence interval for the mean of a metric.

    Attributes:
        point: The observed sample mean.
        low: Lower bound of the confidence interval.
        high: Upper bound of the confidence interval.
        confidence: The two-sided confidence level (e.g. ``0.95``).
    """

    point: float
    low: float
    high: float
    confidence: float

    def as_dict(self) -> dict[str, float]:
        """Return the interval as a plain dict (for JSON/report serialisation)."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DeltaSignificance:
    """Significance of the mean difference between two paired metric series.

    Attributes:
        delta: Observed mean of ``candidate - baseline`` (positive = candidate
            is better when higher metric is better).
        low: Lower bound of the paired-bootstrap CI for the mean difference.
        high: Upper bound of the paired-bootstrap CI for the mean difference.
        p_value: Two-sided sign-flip permutation p-value for ``H0: delta == 0``.
        confidence: The two-sided confidence level of the interval.
        n_resamples: Resamples used for both the CI and the permutation test.
    """

    delta: float
    low: float
    high: float
    p_value: float
    confidence: float
    n_resamples: int

    def as_dict(self) -> dict[str, float | int]:
        """Return the result as a plain dict (for JSON/report serialisation)."""
        return asdict(self)


def _check_resamples(n_resamples: int) -> None:
    """Validate the resample count."""
    if n_resamples <= 0:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")


def _check_confidence(confidence: float) -> None:
    """Validate the confidence level lies strictly in ``(0, 1)``."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile-bootstrap confidence interval for the mean of ``values``.

    Args:
        values: Per-query metric values (e.g. one recall@k per question).
        confidence: Two-sided confidence level in ``(0, 1)``.
        n_resamples: Number of bootstrap resamples.
        seed: Seed for the resampling RNG (reproducibility).

    Returns:
        The sample mean and its confidence-interval bounds.

    Raises:
        ValueError: If ``values`` is empty, or the parameters are out of range.
    """
    _check_confidence(confidence)
    _check_resamples(n_resamples)
    data = np.asarray(values, dtype=float)
    n = data.size
    if n == 0:
        raise ValueError("values must be non-empty")

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, n, size=(n_resamples, n))
    resampled_means = data[sample_indices].mean(axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(resampled_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapCI(
        point=float(data.mean()),
        low=float(low),
        high=float(high),
        confidence=confidence,
    )


def paired_delta_significance(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> DeltaSignificance:
    """Test whether ``candidate`` beats ``baseline`` on the same queries.

    Computes the mean per-query difference ``candidate - baseline`` with a
    paired-bootstrap confidence interval, and a two-sided sign-flip permutation
    p-value for the null hypothesis that the mean difference is zero. The pairing
    (same queries, same order) removes between-query variance, so it detects
    smaller real differences than comparing the two means independently.

    Args:
        baseline: Per-query metric values for the baseline system.
        candidate: Per-query metric values for the candidate system, aligned
            one-to-one with ``baseline``.
        confidence: Two-sided confidence level in ``(0, 1)``.
        n_resamples: Resamples for both the CI and the permutation test.
        seed: Seed for the resampling RNG (reproducibility).

    Returns:
        The observed mean difference, its CI, and the permutation p-value.

    Raises:
        ValueError: If the series are empty, of different length, or the
            parameters are out of range.
    """
    _check_confidence(confidence)
    _check_resamples(n_resamples)
    base = np.asarray(baseline, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    if base.size == 0:
        raise ValueError("series must be non-empty")
    if base.size != cand.size:
        raise ValueError(
            f"series must be the same length, got {base.size} and {cand.size}"
        )

    diffs = cand - base
    n = diffs.size
    observed = float(diffs.mean())
    rng = np.random.default_rng(seed)

    # Paired bootstrap: resample (query) pairs to bound the mean difference.
    sample_indices = rng.integers(0, n, size=(n_resamples, n))
    resampled_deltas = diffs[sample_indices].mean(axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(resampled_deltas, [alpha / 2.0, 1.0 - alpha / 2.0])

    # Sign-flip permutation: under H0 the sign of each paired difference is
    # exchangeable, so randomly flipping signs builds the null distribution of
    # the mean. The +1 in numerator/denominator keeps the p-value unbiased and
    # never exactly zero.
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_resamples, n))
    null_means = (signs * diffs).mean(axis=1)
    extreme = int(np.count_nonzero(np.abs(null_means) >= abs(observed)))
    p_value = (extreme + 1) / (n_resamples + 1)

    return DeltaSignificance(
        delta=observed,
        low=float(low),
        high=float(high),
        p_value=p_value,
        confidence=confidence,
        n_resamples=n_resamples,
    )
