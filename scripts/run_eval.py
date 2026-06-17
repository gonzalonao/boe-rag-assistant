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
from boe_rag.eval.metrics import RetrievalMetrics
from boe_rag.eval.runner import ExampleResult, run_retrieval_eval

logger = logging.getLogger(__name__)


def _load_corpus(path: Path) -> tuple[list[str], list[str]]:
    """Load chunk ids and texts from a corpus Parquet file."""
    table = pq.read_table(path, columns=["chunk_id", "text"])  # type: ignore[no-untyped-call]
    data = table.to_pydict()
    return list(map(str, data["chunk_id"])), list(map(str, data["text"]))


def _render_report(
    metrics: RetrievalMetrics,
    results: list[ExampleResult],
    *,
    model: str,
    corpus: Path,
    num_chunks: int,
    retrieve_n: int,
) -> str:
    """Render the evaluation results as a Markdown report."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    misses = [r for r in results if r.first_relevant_rank is None]
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
        "| Metric | Value |",
        "|---|---|",
        f"| Recall@{metrics.k} | {metrics.recall_at_k:.3f} |",
        f"| Precision@{metrics.k} | {metrics.precision_at_k:.3f} |",
        f"| Hit rate@{metrics.k} | {metrics.hit_rate_at_k:.3f} |",
        f"| MRR | {metrics.mrr:.3f} |",
        f"| nDCG@{metrics.k} | {metrics.ndcg_at_k:.3f} |",
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
    parser.add_argument("--k", type=int, default=10, help="Cut-off for @k metrics.")
    parser.add_argument(
        "--retrieve-n", type=int, default=20, help="Candidates retrieved per query."
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
    logger.info("Embedding %d chunks with %s ...", len(chunk_ids), args.model)
    embedder = E5Embedder(args.model)
    metrics, results = run_retrieval_eval(
        chunk_ids, texts, examples, embedder, k=args.k, retrieve_n=args.retrieve_n
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = _render_report(
        metrics,
        results,
        model=args.model,
        corpus=args.corpus,
        num_chunks=len(chunk_ids),
        retrieve_n=args.retrieve_n,
    )
    args.out.with_suffix(".md").write_text(report, encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps(metrics.as_dict(), indent=2), encoding="utf-8"
    )
    logger.info("Wrote report to %s.{md,json}", args.out)
    print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
