"""Run the retrieval evaluation and write a Markdown + JSON report.

Requires the ``ml`` extra (``pip install -e .[ml]``) for the embedding model.

Example:
    python scripts/run_eval.py \
        --corpus data/corpus/boe-2024.parquet \
        --evalset eval_data/seed_evalset.jsonl \
        --out reports/retrieval_baseline
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from boe_rag.eval.dataset import load_evalset
from boe_rag.eval.embedding import DEFAULT_MODEL, E5Embedder
from boe_rag.eval.metrics import RetrievalMetrics, recall_at_k, reciprocal_rank
from boe_rag.eval.qdrant_store import connect_searcher
from boe_rag.eval.retriever import FloatMatrix, load_embeddings
from boe_rag.eval.runner import (
    ExampleResult,
    evaluate_searcher,
    run_retrieval_eval,
)
from boe_rag.eval.stats import BootstrapCI, bootstrap_mean_ci

logger = logging.getLogger(__name__)


def _load_corpus(path: Path) -> tuple[list[str], list[str]]:
    """Load chunk ids and texts from a corpus Parquet file."""
    table = pq.read_table(path, columns=["chunk_id", "text"])  # type: ignore[no-untyped-call]
    data = table.to_pydict()
    return list(map(str, data["chunk_id"])), list(map(str, data["text"]))


def _confidence_intervals(
    results: list[ExampleResult], k: int, *, n_resamples: int, seed: int
) -> tuple[BootstrapCI, BootstrapCI]:
    """Bootstrap 95% CIs for the headline metrics (recall@k and MRR).

    Recomputes the per-query recall@k and reciprocal-rank series from the stored
    rankings, then bootstraps the mean of each so the report can show how much
    sampling noise sits behind the point estimates.
    """
    recalls = [recall_at_k(r.retrieved_ids, r.relevant_ids, k) for r in results]
    rrs = [reciprocal_rank(r.retrieved_ids, r.relevant_ids) for r in results]
    recall_ci = bootstrap_mean_ci(recalls, n_resamples=n_resamples, seed=seed)
    mrr_ci = bootstrap_mean_ci(rrs, n_resamples=n_resamples, seed=seed)
    return recall_ci, mrr_ci


def _render_report(
    metrics: RetrievalMetrics,
    results: list[ExampleResult],
    *,
    model: str,
    corpus: Path,
    num_chunks: int,
    retrieve_n: int,
    recall_ci: BootstrapCI,
    mrr_ci: BootstrapCI,
) -> str:
    """Render the evaluation results as a Markdown report."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    misses = [r for r in results if r.first_relevant_rank is None]
    pct = round(recall_ci.confidence * 100)
    lines = [
        "# Retrieval evaluation — baseline",
        "",
        f"- **Generated:** {timestamp}",
        f"- **Model:** `{model}`",
        f"- **Corpus:** `{corpus.name}` ({num_chunks} chunks)",
        f"- **Queries:** {metrics.num_queries}",
        f"- **Retrieved per query:** {retrieve_n}",
        "",
        f"## Metrics @{metrics.k}",
        "",
        f"Confidence intervals are {pct}% bootstrap (per-query resampling).",
        "",
        "| Metric | Value | 95% CI |",
        "|---|---|---|",
        f"| Recall@{metrics.k} | {metrics.recall_at_k:.3f} "
        f"| [{recall_ci.low:.3f}, {recall_ci.high:.3f}] |",
        f"| Precision@{metrics.k} | {metrics.precision_at_k:.3f} | — |",
        f"| Hit rate@{metrics.k} | {metrics.hit_rate_at_k:.3f} | — |",
        f"| MRR | {metrics.mrr:.3f} | [{mrr_ci.low:.3f}, {mrr_ci.high:.3f}] |",
        f"| nDCG@{metrics.k} | {metrics.ndcg_at_k:.3f} | — |",
        "",
        "## Per-question first-hit rank",
        "",
        "| Example | First relevant rank |",
        "|---|---|",
    ]
    for r in results:
        rank = r.first_relevant_rank if r.first_relevant_rank else "MISS"
        lines.append(f"| {r.example_id} | {rank} |")
    if misses:
        lines += [
            "",
            f"**Misses ({len(misses)}):** " + ", ".join(r.example_id for r in misses),
        ]
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run the retrieval evaluation.")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus .parquet.")
    parser.add_argument(
        "--evalset",
        type=Path,
        default=Path("eval_data/seed_evalset.jsonl"),
        help="Golden eval set .jsonl.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model id.")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Optional precomputed passage embeddings .npz. When its ids match "
        "the corpus, retrieval is scored without re-encoding (queries are still "
        "encoded live); a mismatch falls back to encoding.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Score the dense leg from a Qdrant collection at this URL instead "
        "of the in-memory NumPy index (proves backend parity). Requires "
        "--qdrant-collection and the `qdrant` extra.",
    )
    parser.add_argument(
        "--qdrant-collection",
        default=None,
        help="Qdrant collection to search when --qdrant-url is given.",
    )
    parser.add_argument("--k", type=int, default=10, help="Cut-off for @k metrics.")
    parser.add_argument(
        "--retrieve-n", type=int, default=20, help="Candidates retrieved per query."
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10_000,
        help="Bootstrap resamples for the recall@k / MRR confidence intervals.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the bootstrap resampling."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/retrieval_baseline"),
        help="Output path stem (.md and .json are written).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation and write the report.

    Returns:
        0 on success.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)

    chunk_ids, texts = _load_corpus(args.corpus)
    examples = load_evalset(args.evalset)
    precomputed: tuple[list[str], FloatMatrix] | None = None
    if args.embeddings is not None:
        logger.info("Loading precomputed embeddings from %s ...", args.embeddings)
        precomputed = load_embeddings(args.embeddings)
    else:
        logger.info("Embedding %d chunks with %s ...", len(chunk_ids), args.model)
    embedder = E5Embedder(args.model)
    if args.qdrant_url and args.qdrant_collection:
        logger.info(
            "Scoring the dense leg from Qdrant '%s' at %s ...",
            args.qdrant_collection,
            args.qdrant_url,
        )
        searcher = connect_searcher(args.qdrant_url, args.qdrant_collection, embedder)
        metrics, results = evaluate_searcher(
            searcher, examples, k=args.k, retrieve_n=args.retrieve_n
        )
    else:
        metrics, results = run_retrieval_eval(
            chunk_ids,
            texts,
            examples,
            embedder,
            k=args.k,
            retrieve_n=args.retrieve_n,
            precomputed=precomputed,
        )
    recall_ci, mrr_ci = _confidence_intervals(
        results, args.k, n_resamples=args.bootstrap_resamples, seed=args.seed
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = _render_report(
        metrics,
        results,
        model=args.model,
        corpus=args.corpus,
        num_chunks=len(chunk_ids),
        retrieve_n=args.retrieve_n,
        recall_ci=recall_ci,
        mrr_ci=mrr_ci,
    )
    payload: dict[str, object] = dict(metrics.as_dict())
    payload["recall_at_k_ci"] = recall_ci.as_dict()
    payload["mrr_ci"] = mrr_ci.as_dict()
    args.out.with_suffix(".md").write_text(report, encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    logger.info("Wrote report to %s.{md,json}", args.out)
    print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
