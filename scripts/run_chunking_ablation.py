"""Compare chunking strategies (article vs fixed-size vs whole-document).

Re-chunks the corpus three ways and scores each with dense retrieval at
*document* granularity (a retrieved chunk is a hit when it belongs to a relevant
document), so strategies with different chunk boundaries are compared fairly.
This is the before/after for the Phase 3 chunking ablation; it validates the
production choice of structure-aware, article-level chunks.

Requires the ``ml`` extra (``pip install -e .[ml]``) for the embedding model.
No LLM API key is needed.

Example:
    python scripts/run_chunking_ablation.py \
        --corpus data/corpus/boe-2024.parquet --out reports/retrieval_chunking
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from boe_rag.eval.chunking import (
    CorpusChunk,
    StrategyChunks,
    article_strategy,
    document_strategy,
    fixed_size_strategy,
)
from boe_rag.eval.dataset import EvalExample, load_evalset
from boe_rag.eval.embedding import DEFAULT_MODEL, E5Embedder
from boe_rag.eval.metrics import RetrievalMetrics, evaluate_retrieval
from boe_rag.eval.report import render_comparison_report
from boe_rag.eval.retriever import DenseRetriever

logger = logging.getLogger(__name__)


def _load_corpus(path: Path) -> list[CorpusChunk]:
    """Load the article-level corpus rows needed for re-chunking."""
    table = pq.read_table(  # type: ignore[no-untyped-call]
        path, columns=["chunk_id", "document_id", "text", "ordinal"]
    )
    data = table.to_pydict()
    return [
        CorpusChunk(
            chunk_id=str(cid),
            document_id=str(doc),
            text=str(text),
            ordinal=int(ordinal),
        )
        for cid, doc, text, ordinal in zip(
            data["chunk_id"],
            data["document_id"],
            data["text"],
            data["ordinal"],
            strict=True,
        )
    ]


def _relevant_docs(
    example: EvalExample, chunk_to_doc: dict[str, str]
) -> frozenset[str]:
    """Map an example's relevant chunk ids to their document ids."""
    return frozenset(
        chunk_to_doc[cid] for cid in example.relevant_chunk_ids if cid in chunk_to_doc
    )


def _unique_ranked_docs(
    retrieved: Sequence[tuple[str, float]], chunk_doc: dict[str, str]
) -> list[str]:
    """Map retrieved chunk ids to their document ids, de-duplicated, in order."""
    ranked: list[str] = []
    seen: set[str] = set()
    for chunk_id, _ in retrieved:
        doc_id = chunk_doc[chunk_id]
        if doc_id not in seen:
            seen.add(doc_id)
            ranked.append(doc_id)
    return ranked


def _evaluate_strategy(
    strategy: StrategyChunks,
    examples: Sequence[EvalExample],
    relevant_by_example: Sequence[frozenset[str]],
    model: str,
    k: int,
    retrieve_n: int,
) -> RetrievalMetrics:
    """Index a strategy's chunks and score document-level retrieval."""
    retriever = DenseRetriever(E5Embedder(model))
    retriever.index(strategy.chunk_ids, strategy.texts)
    chunk_doc = dict(zip(strategy.chunk_ids, strategy.doc_ids, strict=True))

    scored: list[tuple[Sequence[str], frozenset[str]]] = []
    for example, relevant in zip(examples, relevant_by_example, strict=True):
        retrieved = retriever.search(example.question, retrieve_n)
        scored.append((_unique_ranked_docs(retrieved, chunk_doc), relevant))
    return evaluate_retrieval(scored, k=k)


def _render_report(
    results: dict[str, RetrievalMetrics],
    *,
    model: str,
    corpus: Path,
    num_docs: int,
    k: int,
    retrieve_n: int,
    window: int,
    overlap: int,
) -> str:
    """Render the chunking comparison as a Markdown report."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    num_queries = next(iter(results.values())).num_queries
    meta = [
        ("Generated", timestamp),
        ("Embedding model", f"`{model}`"),
        ("Corpus", f"`{corpus.name}` ({num_docs} documents)"),
        ("Queries", str(num_queries)),
        ("Relevance", "document-level (hit = chunk from a relevant document)"),
        ("Retrieved per query", str(retrieve_n)),
        ("Fixed window", f"{window} chars, {overlap} overlap"),
    ]
    return render_comparison_report(
        "Retrieval evaluation - chunking strategies", results, meta=meta, k=k
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Compare chunking strategies.")
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
        "--retrieve-n", type=int, default=50, help="Chunks retrieved per query."
    )
    parser.add_argument("--window", type=int, default=1000, help="Fixed window chars.")
    parser.add_argument(
        "--overlap", type=int, default=150, help="Fixed window overlap."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/retrieval_chunking"),
        help="Output path stem (.md and .json are written).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the chunking comparison and write the report.

    Returns:
        0 on success.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)

    chunks = _load_corpus(args.corpus)
    chunk_to_doc = {c.chunk_id: c.document_id for c in chunks}
    examples = load_evalset(args.evalset)
    relevant_by_example = [_relevant_docs(ex, chunk_to_doc) for ex in examples]

    strategies: dict[str, StrategyChunks] = {
        "article (current)": article_strategy(chunks),
        "fixed-size": fixed_size_strategy(chunks, args.window, args.overlap),
        "whole-document": document_strategy(chunks),
    }
    results: dict[str, RetrievalMetrics] = {}
    for name, strategy in strategies.items():
        logger.info("Evaluating %s (%d chunks) ...", name, len(strategy.chunk_ids))
        results[name] = _evaluate_strategy(
            strategy,
            examples,
            relevant_by_example,
            args.model,
            args.k,
            args.retrieve_n,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = _render_report(
        results,
        model=args.model,
        corpus=args.corpus,
        num_docs=len(set(chunk_to_doc.values())),
        k=args.k,
        retrieve_n=args.retrieve_n,
        window=args.window,
        overlap=args.overlap,
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
