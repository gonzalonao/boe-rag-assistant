"""Generate a silver eval set by prompting an LLM over corpus chunks.

Samples substantive chunks, asks the LLM for a self-contained question + answer
grounded in each, filters out deictic or low-quality questions, and (by default)
keeps only answers the LLM-judge rates faithful to their source chunk. The result
is written as JSONL in the same schema as the hand-curated seed set.

The ``ml`` extra is *not* needed, but at least one LLM API key must be set
(``GROQ_API_KEY`` recommended; ``GEMINI_API_KEY``/``GOOGLE_API_KEY`` also work).
Groq is the reliable free option — Gemini's free tier rate-limits hard.

This is a long job against free-tier limits, so it is built to survive them and to
*resume*: on a rate limit it waits for the provider's cool-down and retries, and it
always writes whatever it has collected (even if interrupted or stopped early).

It is designed to be run many times until the corpus is covered. Kept examples are
merged into ``--out`` (never overwritten), and every chunk that has been *attempted*
(kept or dropped) is recorded in a ``<out>.processed`` ledger so later runs skip it
and never re-spend quota on it. ``--limit`` therefore means "new chunks to attempt
this run" -- set it to whatever your remaining daily quota allows, run, get rate-
limited, and run again later to pick up exactly where you left off. Keep ``--seed``
constant across runs so the chunk order stays stable.

Example (run this repeatedly):
    python scripts/generate_evalset.py --corpus data/corpus/boe-2024.parquet \
        --out eval_data/generated_evalset.jsonl --limit 150
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from boe_rag.eval.dataset import EvalExample, load_evalset, save_evalset
from boe_rag.eval.generate import generate_qa, is_self_contained
from boe_rag.eval.judge import judge_faithfulness
from boe_rag.llm.base import LLMError, LLMRateLimitError
from boe_rag.llm.factory import FallbackProvider, build_available_providers
from boe_rag.settings import load_environment

logger = logging.getLogger(__name__)

#: Chunks shorter than this (characters) are skipped as too thin to question.
DEFAULT_MIN_CHARS = 350
#: Seconds to wait before retrying an item after every provider is rate-limited.
RATE_LIMIT_WAIT_SECONDS = 65.0
#: How many times to wait-and-retry a single item before giving up the job.
MAX_RATE_LIMIT_RETRIES = 3


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


def _processed_path(out: Path) -> Path:
    """Sidecar ledger of attempted chunk ids, alongside the output file."""
    return out.with_name(f"{out.name}.processed")


def _load_processed(path: Path) -> set[str]:
    """Load the set of chunk ids already attempted in earlier runs."""
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def _save_processed(path: Path, ids: set[str]) -> None:
    """Persist the attempted-chunk ledger, one id per line, sorted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk_id in sorted(ids):
            handle.write(chunk_id)
            handle.write("\n")


def _make_example(
    chunk_id: str,
    text: str,
    citation: str,
    provider: FallbackProvider,
    *,
    validate: bool,
    min_faithfulness: float,
) -> tuple[EvalExample | None, str]:
    """Generate and filter one example.

    Returns:
        ``(example, status)`` where status is ``ok``/``deictic``/``unfaithful``.
        ``example`` is ``None`` unless the status is ``ok``.

    Raises:
        LLMRateLimitError: If every provider is currently rate-limited.
        LLMError: On any other provider failure.
    """
    qa = generate_qa(text, citation, provider)
    if not is_self_contained(qa.question):
        return None, "deictic"
    if validate:
        score = judge_faithfulness(qa.answer, [(citation, text)], provider).score
        if score < min_faithfulness:
            return None, "unfaithful"
    example = EvalExample(
        example_id=f"gen-{chunk_id}",
        question=qa.question,
        relevant_chunk_ids=(chunk_id,),
        answer=qa.answer,
        category="generated",
        difficulty="auto",
    )
    return example, "ok"


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
        "--limit",
        type=int,
        default=150,
        help="Max NEW chunks to attempt this run; already-processed chunks are "
        "skipped. Size it to your remaining quota and re-run to resume.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help="Skip chunks shorter than this many characters.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for chunk order; keep it constant across runs to resume.",
    )
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
    load_environment()  # pick up a local .env (real env vars still win)
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
    order = candidates[:]
    random.Random(args.seed).shuffle(order)

    out_path: Path = args.out
    processed_path = _processed_path(out_path)
    existing = load_evalset(out_path) if out_path.is_file() else []
    processed = _load_processed(processed_path)
    # Backfill the ledger for evalsets produced before resume support existed.
    processed |= {ex.relevant_chunk_ids[0] for ex in existing}

    pending = [item for item in order if item[0] not in processed]
    this_run = min(args.limit, len(pending))
    logger.info(
        "Resuming with %d kept example(s); %d of %d eligible chunk(s) already "
        "processed. Attempting up to %d new chunk(s) this run with %s ...",
        len(existing),
        len(processed),
        len(candidates),
        this_run,
        provider.name,
    )

    validate = not args.no_validate and args.min_faithfulness > 0.0
    new_examples: list[EvalExample] = []
    newly_processed: set[str] = set()
    dropped_error = dropped_deictic = dropped_unfaithful = 0
    stopped_early = False
    attempted = 0

    try:
        for chunk_id, text, citation in pending:
            if attempted >= args.limit:
                break
            example: EvalExample | None = None
            status = "error"
            resolved = False
            for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
                try:
                    example, status = _make_example(
                        chunk_id,
                        text,
                        citation,
                        provider,
                        validate=validate,
                        min_faithfulness=args.min_faithfulness,
                    )
                    resolved = True
                    break
                except LLMRateLimitError:
                    if attempt == MAX_RATE_LIMIT_RETRIES:
                        logger.error(
                            "Still rate-limited after %d waits; stopping early "
                            "with %d new example(s) kept.",
                            attempt,
                            len(new_examples),
                        )
                        stopped_early = True
                        break
                    logger.warning(
                        "Rate-limited; waiting %.0fs then retrying (%d/%d) ...",
                        RATE_LIMIT_WAIT_SECONDS,
                        attempt,
                        MAX_RATE_LIMIT_RETRIES,
                    )
                    time.sleep(RATE_LIMIT_WAIT_SECONDS)
                except LLMError as err:
                    logger.warning("Generation failed for %s: %s", chunk_id, err)
                    status = "error"
                    resolved = True
                    break

            if stopped_early:
                break
            if not resolved:
                continue
            # Only mark a chunk processed once we got a definitive outcome, so a
            # rate-limit stop leaves it for the next run rather than burning it.
            attempted += 1
            newly_processed.add(chunk_id)
            if status == "ok" and example is not None:
                new_examples.append(example)
                if attempted % 10 == 0:
                    logger.info(
                        "Kept %d new / attempted %d ...", len(new_examples), attempted
                    )
            elif status == "deictic":
                dropped_deictic += 1
            elif status == "unfaithful":
                dropped_unfaithful += 1
            else:
                dropped_error += 1
    except KeyboardInterrupt:
        logger.warning(
            "Interrupted; saving %d new example(s) collected so far.",
            len(new_examples),
        )

    written = save_evalset(list(existing) + new_examples, out_path)
    _save_processed(processed_path, processed | newly_processed)
    logger.info(
        "Wrote %d example(s) total to %s (+%d new this run; dropped this run: "
        "%d errors, %d deictic, %d unfaithful)%s",
        written,
        out_path,
        len(new_examples),
        dropped_error,
        dropped_deictic,
        dropped_unfaithful,
        " [stopped early: persistent rate limits]" if stopped_early else "",
    )
    logger.info(
        "Ledger now covers %d processed chunk(s): %s",
        len(processed | newly_processed),
        processed_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
