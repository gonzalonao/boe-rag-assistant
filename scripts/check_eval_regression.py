"""Fail the build when gold-set retrieval metrics regress past the baseline.

Reads the metrics JSON produced by ``run_eval.py`` and the committed baseline
(``eval_data/retrieval_baseline.json``), then exits non-zero if any guarded
metric has dropped more than its tolerance below the baseline. Used by the CI
eval-gate (see ``.github/workflows/ci.yml``).

Example:
    python scripts/check_eval_regression.py \
        --metrics reports/ci_retrieval.json \
        --baseline eval_data/retrieval_baseline.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from boe_rag.eval.regression import check_metrics, format_report, load_guards

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Check retrieval metrics against a committed baseline."
    )
    parser.add_argument(
        "--metrics", type=Path, required=True, help="Metrics JSON from run_eval.py."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("eval_data/retrieval_baseline.json"),
        help="Committed baseline JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the regression check.

    Returns:
        0 if every guarded metric cleared its floor, 1 otherwise.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)

    metrics: object = json.loads(args.metrics.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        print(
            f"ERROR: metrics file must be a JSON object: {args.metrics}",
            file=sys.stderr,
        )
        return 1

    guards = load_guards(args.baseline)
    results = check_metrics(metrics, guards)
    print(format_report(results))

    failed = [r.guard.name for r in results if not r.passed]
    if failed:
        print(
            f"\nRetrieval regression gate FAILED: {', '.join(failed)}",
            file=sys.stderr,
        )
        return 1
    print("\nRetrieval regression gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
