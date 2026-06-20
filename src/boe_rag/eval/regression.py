"""Retrieval regression gate: compare measured metrics to a committed baseline.

The CI eval-gate runs the gold-set retrieval evaluation and uses this module to
fail a pull request when a guarded metric drops more than its allowed tolerance
below the committed baseline. Keeping the comparison logic here (rather than in
the CLI script) puts the gate that protects retrieval quality under
``mypy --strict`` and pytest, so it is itself tested.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MetricGuard:
    """A guarded metric and how far it may fall before the gate fails.

    Attributes:
        name: The metric key as written in the metrics JSON (e.g. ``"mrr"``).
        baseline: The reference value the metric is expected to hold.
        tolerance: The largest drop below ``baseline`` tolerated before the
            guard fails. A guard passes when ``value >= baseline - tolerance``.
    """

    name: str
    baseline: float
    tolerance: float

    @property
    def floor(self) -> float:
        """The lowest value that still passes the guard."""
        return self.baseline - self.tolerance


@dataclass(frozen=True, slots=True)
class GuardResult:
    """The outcome of checking one metric against its guard.

    Attributes:
        guard: The guard that was applied.
        value: The measured metric value.
    """

    guard: MetricGuard
    value: float

    @property
    def passed(self) -> bool:
        """Whether the measured value cleared the guard's floor."""
        return self.value >= self.guard.floor


def _as_float(value: object, *, where: str) -> float:
    """Coerce a JSON scalar to ``float``, with a clear error on the wrong type."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected a number for {where}, got {value!r}")
    return float(value)


def load_guards(baseline_path: Path) -> list[MetricGuard]:
    """Load the guarded metrics from a baseline JSON file.

    Args:
        baseline_path: Path to the committed baseline JSON.

    Returns:
        The guards declared under the file's ``"guards"`` mapping.

    Raises:
        ValueError: If the file is malformed or declares no guards.
    """
    raw: object = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Baseline must be a JSON object, got {type(raw).__name__}")
    guards_raw = raw.get("guards")
    if not isinstance(guards_raw, Mapping) or not guards_raw:
        raise ValueError("Baseline must declare a non-empty 'guards' object")
    guards: list[MetricGuard] = []
    for name, spec in guards_raw.items():
        if not isinstance(spec, Mapping):
            raise ValueError(f"Guard {name!r} must be an object")
        guards.append(
            MetricGuard(
                name=str(name),
                baseline=_as_float(spec.get("baseline"), where=f"{name}.baseline"),
                tolerance=_as_float(spec.get("tolerance"), where=f"{name}.tolerance"),
            )
        )
    return guards


def check_metrics(
    metrics: Mapping[str, object], guards: list[MetricGuard]
) -> list[GuardResult]:
    """Apply each guard to the measured metrics.

    Args:
        metrics: The measured metrics (as produced by ``run_eval.py``).
        guards: The guards to apply.

    Returns:
        One :class:`GuardResult` per guard, in the given order.

    Raises:
        KeyError: If a guarded metric is absent from ``metrics``.
    """
    results: list[GuardResult] = []
    for guard in guards:
        if guard.name not in metrics:
            raise KeyError(f"Guarded metric {guard.name!r} not found in metrics")
        results.append(
            GuardResult(
                guard=guard, value=_as_float(metrics[guard.name], where=guard.name)
            )
        )
    return results


def format_report(results: list[GuardResult]) -> str:
    """Render the guard results as a compact, fixed-width text table."""
    header = f"{'metric':<16} {'value':>8} {'floor':>8} {'baseline':>9}  status"
    lines = [header, "-" * len(header)]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(
            f"{result.guard.name:<16} {result.value:>8.3f} "
            f"{result.guard.floor:>8.3f} {result.guard.baseline:>9.3f}  {status}"
        )
    return "\n".join(lines)
