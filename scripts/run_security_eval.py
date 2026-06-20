"""Run the adversarial security evaluation against the grounded generator.

Sends a curated set of prompt-injection, exfiltration, citation-spoofing, and
out-of-corpus attacks through the real answer pipeline and scores each answer
with deterministic, rule-based checks (see ``boe_rag.eval.security``). Exercises
the generation guardrails, which is where these attacks land.

Requires the ``ml`` extra (``pip install -e .[ml]``) and at least one LLM API key
in the environment (``OPENROUTER_API_KEY`` / ``GROQ_API_KEY`` / ``GEMINI_API_KEY``).

Example:
    python scripts/run_security_eval.py --corpus data/corpus/boe-2024.parquet \
        --out reports/security_eval
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from boe_rag.eval.answerer import REFUSAL, SYSTEM_PROMPT_CANARY
from boe_rag.eval.embedding import DEFAULT_MODEL, E5Embedder
from boe_rag.eval.retriever import DenseRetriever
from boe_rag.eval.security import (
    SecurityReport,
    evaluate_security,
    load_adversarial_cases,
)
from boe_rag.llm.base import LLMError
from boe_rag.llm.factory import FallbackProvider, build_available_providers
from boe_rag.service.engine import ChunkInfo, RagEngine
from boe_rag.settings import load_environment

logger = logging.getLogger(__name__)


def _load_corpus(path: Path) -> tuple[list[str], list[str], dict[str, ChunkInfo]]:
    """Load chunk ids/texts and a chunk-id → :class:`ChunkInfo` lookup."""
    table = pq.read_table(  # type: ignore[no-untyped-call]
        path, columns=["chunk_id", "text", "citation", "url_html"]
    )
    data = table.to_pydict()
    ids = list(map(str, data["chunk_id"]))
    texts = list(map(str, data["text"]))
    citations = list(map(str, data["citation"]))
    urls = list(map(str, data["url_html"]))
    lookup = {
        cid: ChunkInfo(citation=cit, text=txt, url=url)
        for cid, txt, cit, url in zip(ids, texts, citations, urls, strict=True)
    }
    return ids, texts, lookup


def _render_report(report: SecurityReport, *, provider_name: str, k: int) -> str:
    """Render the security findings as a Markdown report."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Adversarial security evaluation",
        "",
        f"- **Generated:** {timestamp}",
        f"- **Provider:** `{provider_name}`",
        f"- **Passages per question (k):** {k}",
        f"- **Cases:** {report.num_cases}",
        f"- **Passed:** {report.num_passed}/{report.num_cases} "
        f"({report.pass_rate:.0%})",
        "",
        "## Pass rate by attack category",
        "",
        "| Category | Pass rate |",
        "|---|---|",
    ]
    for category, rate in report.pass_rate_by_category().items():
        lines.append(f"| {category} | {rate:.0%} |")
    lines += [
        "",
        "## Per-case findings",
        "",
        "| Case | Category | Expectation | Result | Detail |",
        "|---|---|---|---|---|",
    ]
    for f in report.findings:
        result = "PASS" if f.passed else "**FAIL**"
        lines.append(
            f"| {f.case_id} | {f.category} | {f.expectation} | {result} | {f.detail} |"
        )
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run the security evaluation.")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus .parquet.")
    parser.add_argument(
        "--adversarial",
        type=Path,
        default=Path("eval_data/adversarial_security.jsonl"),
        help="Adversarial cases .jsonl.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model id.")
    parser.add_argument("--k", type=int, default=5, help="Passages per question.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/security_eval"),
        help="Output path stem (.md and .json are written).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the security evaluation and write the report.

    Returns:
        0 on success, 1 if no LLM provider is configured.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_environment()  # pick up a local .env (real env vars still win)
    args = _build_parser().parse_args(argv)

    providers = build_available_providers()
    if not providers:
        logger.error(
            "No LLM provider configured. Set OPENROUTER_API_KEY (recommended) "
            "and/or GROQ_API_KEY, GEMINI_API_KEY."
        )
        return 1

    ids, texts, lookup = _load_corpus(args.corpus)
    cases = load_adversarial_cases(args.adversarial)
    logger.info("Indexing %d chunks with %s ...", len(ids), args.model)
    retriever = DenseRetriever(E5Embedder(args.model))
    retriever.index(ids, texts)
    provider = FallbackProvider(providers)
    engine = RagEngine(retriever=retriever, lookup=lookup, provider=provider)

    def answer_fn(question: str) -> tuple[str, int]:
        response = engine.answer(question, k=args.k)
        return response.answer, len(response.sources)

    logger.info("Running %d adversarial cases with %s ...", len(cases), provider.name)
    try:
        report = evaluate_security(
            cases, answer_fn, canary=SYSTEM_PROMPT_CANARY, refusal=REFUSAL
        )
    except LLMError as err:
        logger.error("Evaluation failed: %s", err)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rendered = _render_report(report, provider_name=provider.name, k=args.k)
    args.out.with_suffix(".md").write_text(rendered, encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Wrote report to %s.{md,json}", args.out)
    print("\n" + rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
