"""Split an eval set into leakage-free train/test partitions by document (Arc 6).

The go/no-go gate needs a held-out test set large enough to detect a real
retrieval gain — the 20-query gold set is too small. This carves a held-out test
split from the silver set, splitting by **source document** so no test positive
comes from a document seen in training (see :mod:`boe_rag.eval.split`).

The split is deterministic given ``--seed``, so the outputs are reproducible and
need not be committed. By default they land in ``eval_data/`` (git-ignored).

Example:
    python scripts/split_evalset.py \
        --in eval_data/generated_evalset.jsonl \
        --train-out eval_data/silver_train.jsonl \
        --test-out eval_data/silver_test.jsonl \
        --test-fraction 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from boe_rag.eval.dataset import load_evalset, save_evalset
from boe_rag.eval.split import document_id, split_by_document

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Split an eval set into leakage-free train/test partitions."
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=Path("eval_data/generated_evalset.jsonl"),
        help="Source eval set (JSONL).",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=Path("eval_data/silver_train.jsonl"),
        help="Destination for the training split.",
    )
    parser.add_argument(
        "--test-out",
        type=Path,
        default=Path("eval_data/silver_test.jsonl"),
        help="Destination for the held-out test split.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Target share of examples for the test split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split seed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Split the eval set and write both partitions.

    Returns:
        Process exit code (0 on success, 1 if the source file is missing).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)
    if not args.in_path.is_file():
        print(f"ERROR: eval set not found: {args.in_path}", file=sys.stderr)
        return 1

    examples = load_evalset(args.in_path)
    train, test = split_by_document(
        examples, test_fraction=args.test_fraction, seed=args.seed
    )
    save_evalset(train, args.train_out)
    save_evalset(test, args.test_out)

    train_docs = {document_id(c) for ex in train for c in ex.relevant_chunk_ids}
    test_docs = {document_id(c) for ex in test for c in ex.relevant_chunk_ids}
    overlap = train_docs & test_docs
    logger.info(
        "Split %d examples -> %d train / %d test (%.1f%% test); "
        "%d/%d train/test documents, %d overlapping",
        len(examples),
        len(train),
        len(test),
        100.0 * len(test) / len(examples),
        len(train_docs),
        len(test_docs),
        len(overlap),
    )
    if overlap:
        # Should be impossible given the document-level split; guard anyway.
        print(f"ERROR: {len(overlap)} documents leak across splits", file=sys.stderr)
        return 1
    logger.info("Wrote %s and %s", args.train_out, args.test_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
