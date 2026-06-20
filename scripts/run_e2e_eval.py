"""Run the end-to-end RAG evaluation (retrieve → generate → judge).

Requires the ``ml`` extra (``pip install -e .[ml]``) and at least one LLM API
key in the environment (``GEMINI_API_KEY``/``GOOGLE_API_KEY`` and/or
``GROQ_API_KEY``).

Example:
    python scripts/run_e2e_eval.py --corpus data/corpus/boe-2024.parquet \
        --out reports/e2e_baseline
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from boe_rag.eval.dataset import load_evalset
from boe_rag.eval.e2e import E2EExampleResult, E2EMetrics, run_e2e_eval
from boe_rag.eval.embedding import DEFAULT_MODEL, E5Embedder
from boe_rag.eval.retriever import DenseRetriever
from boe_rag.llm.base import LLMError
from boe_rag.llm.factory import FallbackProvider, build_available_providers
from boe_rag.settings import load_environment

logger = logging.getLogger(__name__)


def _load_corpus(path: Path) -> tuple[list[str], list[str], dict[str, tuple[str, str]]]:
    """Load chunk ids/texts and a chunk-id → (citation, text) lookup."""
    table = pq.read_table(  # type: ignore[no-untyped-call]
        path, columns=["chunk_id", "text", "citation"]
    )
    data = table.to_pydict()
    ids = list(map(str, data["chunk_id"]))
    texts = list(map(str, data["text"]))
    citations = list(map(str, data["citation"]))
    lookup = {
        cid: (cit, txt) for cid, txt, cit in zip(ids, texts, citations, strict=True)
    }
    return ids, texts, lookup


def _render_report(
    metrics: E2EMetrics,
    results: list[E2EExampleResult],
    *,
    provider_name: str,
    k: int,
) -> str:
    """Render the end-to-end results as a Markdown report."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# End-to-end evaluation — baseline",
        "",
        f"- **Generated:** {timestamp}",
        f"- **Provider:** `{provider_name}`",
        f"- **Passages per question (k):** {k}",
        f"- **Questions:** {metrics.num_queries}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Mean faithfulness | {metrics.mean_faithfulness:.3f} |",
        f"| Mean correctness | {metrics.mean_correctness:.3f} |",
        f"| Refusal rate | {metrics.refusal_rate:.3f} |",
        "",
        "## Per-question",
        "",
        "| Example | Faithful | Correct | Refused |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.example_id} | {r.faithfulness:.2f} | {r.correctness:.2f} | "
            f"{'yes' if r.refused else 'no'} |"
        )
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run the end-to-end RAG evaluation.")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus .parquet.")
    parser.add_argument(
        "--evalset",
        type=Path,
        default=Path("eval_data/seed_evalset.jsonl"),
        help="Golden eval set .jsonl.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model id.")
    parser.add_argument("--k", type=int, default=5, help="Passages per question.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/e2e_baseline"),
        help="Output path stem (.md and .json are written).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the end-to-end evaluation and write the report.

    Returns:
        0 on success, 1 if no LLM provider is configured.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_environment()  # pick up a local .env (real env vars still win)
    args = _build_parser().parse_args(argv)

    providers = build_available_providers()
    if not providers:
        logger.error(
            "No LLM provider configured. Set GEMINI_API_KEY/GOOGLE_API_KEY "
            "and/or GROQ_API_KEY."
        )
        return 1
    provider = FallbackProvider(providers)

    ids, texts, lookup = _load_corpus(args.corpus)
    examples = load_evalset(args.evalset)
    logger.info("Indexing %d chunks with %s ...", len(ids), args.model)
    retriever = DenseRetriever(E5Embedder(args.model))
    retriever.index(ids, texts)

    logger.info("Judging %d answers with %s ...", len(examples), provider.name)
    try:
        metrics, results = run_e2e_eval(retriever, lookup, examples, provider, k=args.k)
    except LLMError as err:
        logger.error("Evaluation failed: %s", err)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = _render_report(metrics, results, provider_name=provider.name, k=args.k)
    args.out.with_suffix(".md").write_text(report, encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps(metrics.as_dict(), indent=2), encoding="utf-8"
    )
    logger.info("Wrote report to %s.{md,json}", args.out)
    print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
