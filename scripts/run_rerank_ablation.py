"""Measure cross-encoder reranking on top of hybrid retrieval.

Builds the first-stage retrievers (dense, hybrid) and a second-stage reranker,
then scores dense, hybrid, and hybrid+rerank on the golden set — the before/after
table for the Phase 3 reranking PR.

Requires the ``ml`` extra (``pip install -e .[ml]``); the cross-encoder is
downloaded on first run.

Example:
    python scripts/run_rerank_ablation.py \
        --corpus data/corpus/boe-2024.parquet \
        --out reports/retrieval_rerank
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from boe_rag.eval.cross_encoder import DEFAULT_RERANK_MODEL, CrossEncoderReranker
from boe_rag.eval.dataset import load_evalset
from boe_rag.eval.embedding import DEFAULT_MODEL, E5Embedder
from boe_rag.eval.hybrid import HybridRetriever
from boe_rag.eval.metrics import RetrievalMetrics
from boe_rag.eval.report import render_comparison_report
from boe_rag.eval.rerank import DEFAULT_RERANK_POOL, RerankingRetriever
from boe_rag.eval.retriever import DenseRetriever, Searcher
from boe_rag.eval.runner import evaluate_searcher
from boe_rag.eval.sparse import BM25Index

logger = logging.getLogger(__name__)


def _load_corpus(path: Path) -> tuple[list[str], list[str]]:
    """Load chunk ids and texts from a corpus Parquet file."""
    table = pq.read_table(path, columns=["chunk_id", "text"])  # type: ignore[no-untyped-call]
    data = table.to_pydict()
    return list(map(str, data["chunk_id"])), list(map(str, data["text"]))


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Measure cross-encoder reranking.")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus .parquet.")
    parser.add_argument(
        "--evalset",
        type=Path,
        default=Path("eval_data/seed_evalset.jsonl"),
        help="Golden eval set .jsonl.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model id.")
    parser.add_argument(
        "--rerank-model", default=DEFAULT_RERANK_MODEL, help="Cross-encoder model id."
    )
    parser.add_argument("--k", type=int, default=10, help="Cut-off for @k metrics.")
    parser.add_argument(
        "--retrieve-n", type=int, default=20, help="Candidates retrieved per query."
    )
    parser.add_argument(
        "--pool",
        type=int,
        default=DEFAULT_RERANK_POOL,
        help="First-stage candidates fed to the reranker.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/retrieval_rerank"),
        help="Output path stem (.md and .json are written).",
    )
    return parser


def _render_report(
    results: dict[str, RetrievalMetrics],
    *,
    model: str,
    rerank_model: str,
    corpus: Path,
    num_chunks: int,
    k: int,
    retrieve_n: int,
    pool: int,
) -> str:
    """Render the reranking comparison as a Markdown report."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    num_queries = next(iter(results.values())).num_queries
    meta = [
        ("Generated", timestamp),
        ("Embedding model", f"`{model}`"),
        ("Reranker", f"`{rerank_model}`"),
        ("Corpus", f"`{corpus.name}` ({num_chunks} chunks)"),
        ("Queries", str(num_queries)),
        ("Retrieved per query", str(retrieve_n)),
        ("Rerank pool", str(pool)),
    ]
    return render_comparison_report(
        "Retrieval evaluation — reranking", results, meta=meta, k=k
    )


def main(argv: list[str] | None = None) -> int:
    """Run the reranking comparison and write the report.

    Returns:
        0 on success.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)

    chunk_ids, texts = _load_corpus(args.corpus)
    examples = load_evalset(args.evalset)
    id_to_text = dict(zip(chunk_ids, texts, strict=True))

    logger.info("Embedding %d chunks with %s ...", len(chunk_ids), args.model)
    dense = DenseRetriever(E5Embedder(args.model))
    dense.index(chunk_ids, texts)
    bm25 = BM25Index()
    bm25.index(chunk_ids, texts)
    hybrid = HybridRetriever(dense, bm25)

    logger.info("Loading reranker %s ...", args.rerank_model)
    reranker = CrossEncoderReranker(args.rerank_model)
    reranked = RerankingRetriever(hybrid, reranker, id_to_text, pool=args.pool)

    retrievers: dict[str, Searcher] = {
        "dense": dense,
        "hybrid (RRF)": hybrid,
        "hybrid + cross-encoder": reranked,
    }
    results: dict[str, RetrievalMetrics] = {}
    for name, retriever in retrievers.items():
        logger.info("Evaluating %s ...", name)
        metrics, _ = evaluate_searcher(
            retriever, examples, k=args.k, retrieve_n=args.retrieve_n
        )
        results[name] = metrics

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = _render_report(
        results,
        model=args.model,
        rerank_model=args.rerank_model,
        corpus=args.corpus,
        num_chunks=len(chunk_ids),
        k=args.k,
        retrieve_n=args.retrieve_n,
        pool=args.pool,
    )
    args.out.with_suffix(".md").write_text(report, encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps({name: m.as_dict() for name, m in results.items()}, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote report to %s.{md,json}", args.out)
    print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
