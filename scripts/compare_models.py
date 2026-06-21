"""Compare a fine-tuned embedder against the baseline on the gold set (Arc 6).

The go/no-go gate for the fine-tune: encode the corpus with each model, score
both dense retrievers over the **same** gold questions, and judge the difference
with paired bootstrap CIs + a sign-flip permutation test (``eval/compare.py``).
Prints a before/after table and a conservative ship/no-ship verdict, and writes a
Markdown + JSON report. Ship only if recall@k improves significantly.

Requires the ``ml`` extra and a CUDA-matched torch (it encodes the corpus twice).

Example:
    python scripts/compare_models.py \
        --corpus data/corpus/boe-2015-present.parquet \
        --evalset eval_data/seed_evalset.jsonl \
        --candidate-model models/boe-e5-small \
        --out reports/finetune_compare
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from boe_rag.eval.compare import (
    ModelComparison,
    compare_models,
    per_query_recall_and_rr,
    recommend_ship,
)
from boe_rag.eval.dataset import EvalExample, load_evalset
from boe_rag.eval.embedding import DEFAULT_MODEL, E5Embedder
from boe_rag.eval.runner import ExampleResult, build_dense_searcher, evaluate_searcher
from boe_rag.eval.stats import DEFAULT_RESAMPLES

logger = logging.getLogger(__name__)


def _load_corpus(path: Path) -> tuple[list[str], list[str]]:
    """Load chunk ids and texts from a corpus Parquet file."""
    table = pq.read_table(path, columns=["chunk_id", "text"])  # type: ignore[no-untyped-call]
    data = table.to_pydict()
    return list(map(str, data["chunk_id"])), list(map(str, data["text"]))


def _evaluate(
    model_name: str,
    chunk_ids: list[str],
    texts: list[str],
    examples: list[EvalExample],
    *,
    k: int,
    retrieve_n: int,
) -> list[ExampleResult]:
    """Index the corpus with one model and score it over the gold examples."""
    logger.info("Encoding %d chunks with %s ...", len(chunk_ids), model_name)
    searcher = build_dense_searcher(chunk_ids, texts, E5Embedder(model_name))
    _, results = evaluate_searcher(searcher, examples, k=k, retrieve_n=retrieve_n)
    return results


def _means(results: list[ExampleResult], k: int) -> tuple[float, float]:
    """Return the mean recall@k and MRR over a run."""
    recalls, rrs = per_query_recall_and_rr(results, k)
    return sum(recalls) / len(recalls), sum(rrs) / len(rrs)


def _render_report(
    comparison: ModelComparison,
    *,
    base_model: str,
    candidate_model: str,
    base_means: tuple[float, float],
    cand_means: tuple[float, float],
    ship: bool,
    reason: str,
) -> str:
    """Render the before/after comparison as a Markdown report."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    recall, mrr = comparison.recall, comparison.mrr
    base_recall, base_mrr = base_means
    cand_recall, cand_mrr = cand_means
    verdict = "✅ SHIP" if ship else "❌ NO-SHIP"
    pct = round(recall.confidence * 100)
    return "\n".join(
        [
            "# Embedding fine-tune — baseline vs candidate",
            "",
            f"- **Generated:** {timestamp}",
            f"- **Baseline:** `{base_model}`",
            f"- **Candidate:** `{candidate_model}`",
            f"- **Gold queries:** {comparison.num_queries}",
            f"- **Verdict:** {verdict} — {reason}",
            "",
            f"Delta is candidate - baseline; CIs are {pct}% paired bootstrap, p-values "
            "two-sided sign-flip permutation.",
            "",
            "| Metric | Baseline | Candidate | Δ | 95% CI | p |",
            "|---|---|---|---|---|---|",
            f"| Recall@{comparison.k} | {base_recall:.3f} | {cand_recall:.3f} "
            f"| {recall.delta:+.3f} | [{recall.low:+.3f}, {recall.high:+.3f}] "
            f"| {recall.p_value:.3f} |",
            f"| MRR | {base_mrr:.3f} | {cand_mrr:.3f} | {mrr.delta:+.3f} "
            f"| [{mrr.low:+.3f}, {mrr.high:+.3f}] | {mrr.p_value:.3f} |",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    """Run the baseline-vs-candidate comparison.

    Returns:
        Process exit code: 0 if the candidate should ship, 2 if it should not,
        1 on a missing input file.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)
    if not args.corpus.is_file():
        print(f"ERROR: corpus not found: {args.corpus}", file=sys.stderr)
        return 1

    chunk_ids, texts = _load_corpus(args.corpus)
    examples = load_evalset(args.evalset)

    base_results = _evaluate(
        args.base_model,
        chunk_ids,
        texts,
        examples,
        k=args.k,
        retrieve_n=args.retrieve_n,
    )
    cand_results = _evaluate(
        args.candidate_model,
        chunk_ids,
        texts,
        examples,
        k=args.k,
        retrieve_n=args.retrieve_n,
    )

    comparison = compare_models(
        base_results,
        cand_results,
        k=args.k,
        n_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    ship, reason = recommend_ship(comparison, alpha=args.alpha)

    report = _render_report(
        comparison,
        base_model=args.base_model,
        candidate_model=args.candidate_model,
        base_means=_means(base_results, args.k),
        cand_means=_means(cand_results, args.k),
        ship=ship,
        reason=reason,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".md").write_text(report, encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "k": comparison.k,
                "num_queries": comparison.num_queries,
                "ship": ship,
                "reason": reason,
                "recall": comparison.recall.as_dict(),
                "mrr": comparison.mrr.as_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n" + report)
    logger.info("Wrote comparison to %s.{md,json}", args.out)
    return 0 if ship else 2


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Compare two embedders on gold.")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus .parquet.")
    parser.add_argument(
        "--evalset",
        type=Path,
        default=Path("eval_data/seed_evalset.jsonl"),
        help="Gold eval set (held out from training).",
    )
    parser.add_argument(
        "--base-model", default=DEFAULT_MODEL, help="Baseline embedder id."
    )
    parser.add_argument(
        "--candidate-model",
        required=True,
        help="Fine-tuned model dir or id to evaluate against the baseline.",
    )
    parser.add_argument("--k", type=int, default=10, help="Cut-off for recall@k.")
    parser.add_argument(
        "--retrieve-n", type=int, default=20, help="Candidates retrieved per query."
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_RESAMPLES,
        help="Resamples for the paired bootstrap / permutation test.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Resampling seed.")
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Significance threshold to ship."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/finetune_compare"),
        help="Output path stem (.md and .json are written).",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
