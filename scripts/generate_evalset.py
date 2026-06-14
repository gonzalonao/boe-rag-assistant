"""Generate a silver eval set by prompting an LLM over corpus chunks.

Samples substantive chunks, asks the LLM for a self-contained question + answer
grounded in each, filters out deictic or low-quality questions, and (by default)
keeps only answers the LLM judges faithful to their source chunk. The result is
written as JSONL in the same schema as the hand-curated seed set.

Requires the ``ml`` extra is *not* needed, but at least one LLM API key must be
set (``GROQ_API_KEY`` recommended; ``GEMINI_API_KEY``/``GOOGLE_API_KEY`` also
work). Groq is the reliable free option — Gemini's free tier rate-limits hard.

Example:
    python scripts/generate_evalset.py --corpus data/corpus/boe-2024.parquet \
        --out eval_data/generated_evalset.jsonl --limit 150
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from boe_rag.eval.dataset import EvalExample, save_evalset
from boe_rag.eval.generate import generate_qa, is_self_contained
from boe_rag.eval.judge import judge_faithfulness
from boe_rag.llm.base import LLMError
from boe_rag.llm.factory import FallbackProvider, build_available_providers

logger = logging.getLogger(__name__)

#: Chunks shorter than this (characters) are skipped as too thin to question.
DEFAULT_MIN_CHARS = 350


def _load_corpus(path: Path) -> list[tuple[str, str, str]]:
    """Load ``(chunk_id, text, citation)`` triples from a corpus Parquet file."""
    table = pq.read_table(  # type: ignore[no-untyped-call]
        path, columns=["chunk_id", "text", "citation"]
    )
    data = table.to_pydict()
    return [
        (str(cid), str(text), str(cit))
        for cid, text, cit in zip(
            data["chunk_id"], data["text"], data["citation"], strict=True
        )
    ]


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Generate a silver eval set.")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus .parquet.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval_data/generated_evalset.jsonl"),
        help="Destination .jsonl.",
    )
    parser.add_argument(
        "--limit", type=int, default=150, help="Number of chunks to sample."
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help="Skip chunks shorter than this many characters.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling RNG seed.")
    parser.add_argument(
        "--min-faithfulness",
        type=float,
        default=0.7,
        help="Drop pairs the judge scores below this (0 disables the filter).",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the LLM faithfulness filter (faster, lower quality).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate the silver eval set and write it to JSONL.

    Returns:
        0 on success, 1 if no LLM provider is configured.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)

    providers = build_available_providers()
    if not providers:
        logger.error(
            "No LLM provider configured. Set GROQ_API_KEY (recommended) "
            "and/or GEMINI_API_KEY/GOOGLE_API_KEY."
        )
        return 1
    provider = FallbackProvider(providers)

    corpus = _load_corpus(args.corpus)
    candidates = [item for item in corpus if len(item[1]) >= args.min_chars]
    sample_size = min(args.limit, len(candidates))
    sample = random.Random(args.seed).sample(candidates, sample_size)
    logger.info(
        "Generating from %d of %d eligible chunks with %s ...",
        sample_size,
        len(candidates),
        provider.name,
    )

    validate = not args.no_validate and args.min_faithfulness > 0.0
    examples: list[EvalExample] = []
    dropped_error = dropped_deictic = dropped_unfaithful = 0
    for chunk_id, text, citation in sample:
        try:
            qa = generate_qa(text, citation, provider)
        except LLMError as err:
            logger.warning("Generation failed for %s: %s", chunk_id, err)
            dropped_error += 1
            continue
        if not is_self_contained(qa.question):
            dropped_deictic += 1
            continue
        if validate:
            score = judge_faithfulness(qa.answer, [(citation, text)], provider).score
            if score < args.min_faithfulness:
                dropped_unfaithful += 1
                continue
        examples.append(
            EvalExample(
                example_id=f"gen-{chunk_id}",
                question=qa.question,
                relevant_chunk_ids=(chunk_id,),
                answer=qa.answer,
                category="generated",
                difficulty="auto",
            )
        )

    written = save_evalset(examples, args.out)
    logger.info(
        "Wrote %d examples to %s (dropped: %d errors, %d deictic, %d unfaithful).",
        written,
        args.out,
        dropped_error,
        dropped_deictic,
        dropped_unfaithful,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
