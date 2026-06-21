"""Fine-tune the E5 embedding model on mined BOE retrieval pairs (Arc 6).

Domain-tunes ``multilingual-e5-small`` with a contrastive objective so the dense
retriever learns Spanish-legal phrasing the off-the-shelf model only approximates.
The pipeline is: load the corpus, mine (question, positive-chunk, hard-negatives)
pairs from the silver eval set (lexical near-misses as negatives), then train with
``MultipleNegativesRankingLoss`` — in-batch negatives plus the mined hard ones.

This is a **GPU workflow**, not run in CI: it needs the ``ml`` and ``train`` extras
(``pip install -e ".[ml,train]"``) and a CUDA-matched torch. Evaluate the result
honestly with ``scripts/compare_models.py`` (paired significance on the gold set)
and ship only if it wins.

Example (RTX 5070, 12 GB):
    python scripts/finetune_embeddings.py \
        --corpus data/corpus/boe-2015-present.parquet \
        --train-evalset eval_data/generated_evalset.jsonl \
        --out models/boe-e5-small \
        --epochs 1 --batch-size 64 --num-negatives 4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)

from boe_rag.eval.dataset import load_evalset
from boe_rag.eval.embedding import DEFAULT_MODEL
from boe_rag.eval.mine_pairs import (
    DEFAULT_NUM_NEGATIVES,
    DEFAULT_POOL,
    build_columnar_dataset,
    build_training_pairs,
    save_pairs,
)
from boe_rag.eval.sparse import BM25Index

logger = logging.getLogger(__name__)


def _load_corpus(path: Path) -> tuple[list[str], list[str]]:
    """Load chunk ids and texts from a corpus Parquet file."""
    table = pq.read_table(path, columns=["chunk_id", "text"])  # type: ignore[no-untyped-call]
    data = table.to_pydict()
    return list(map(str, data["chunk_id"])), list(map(str, data["text"]))


def _build_dataset(
    corpus: Path,
    train_evalset: Path,
    *,
    num_negatives: int,
    pool: int,
    pairs_out: Path | None,
) -> Dataset:
    """Mine training pairs and lay them out as a rectangular HF dataset."""
    chunk_ids, texts = _load_corpus(corpus)
    lookup = dict(zip(chunk_ids, texts, strict=True))
    logger.info("Loaded %d corpus chunks from %s", len(chunk_ids), corpus)

    bm25 = BM25Index()
    bm25.index(chunk_ids, texts)
    examples = load_evalset(train_evalset)
    logger.info("Mining pairs from %d examples ...", len(examples))
    pairs = build_training_pairs(
        examples, lookup, bm25, num_negatives=num_negatives, pool=pool
    )
    if pairs_out is not None:
        save_pairs(pairs, pairs_out)
        logger.info("Wrote %d mined pairs to %s", len(pairs), pairs_out)

    columns = build_columnar_dataset(pairs, num_negatives)
    n_rows = len(columns["anchor"])
    logger.info(
        "Built %d training rows (%d pairs, %d dropped for too few negatives)",
        n_rows,
        len(pairs),
        len(pairs) - n_rows,
    )
    if n_rows == 0:
        raise ValueError("no training rows produced; check the corpus/eval set")
    return Dataset.from_dict(columns)


def train(args: argparse.Namespace) -> Path:
    """Run the fine-tune and save the model, returning the output dir."""
    dataset = _build_dataset(
        args.corpus,
        args.train_evalset,
        num_negatives=args.num_negatives,
        pool=args.pool,
        pairs_out=args.pairs_out,
    )

    logger.info("Loading base model %s ...", args.model)
    model = SentenceTransformer(args.model, device=args.device)
    loss = losses.MultipleNegativesRankingLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        # MNRL must not see the same positive twice in a batch (it would become a
        # false negative for another anchor); this sampler guarantees uniqueness.
        batch_sampler="no_duplicates",  # type: ignore[arg-type]
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to=[],
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=training_args, train_dataset=dataset, loss=loss
    )
    logger.info(
        "Training: %d rows, batch %d, %g epoch(s), lr %g, fp16=%s",
        dataset.num_rows,
        args.batch_size,
        args.epochs,
        args.lr,
        args.fp16,
    )
    trainer.train()

    model.save_pretrained(str(args.out))
    logger.info("Saved fine-tuned model to %s", args.out)
    return args.out


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Fine-tune the E5 embedder.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corpus/boe-2015-present.parquet"),
        help="Corpus Parquet (chunk_id + text).",
    )
    parser.add_argument(
        "--train-evalset",
        type=Path,
        default=Path("eval_data/generated_evalset.jsonl"),
        help="Silver eval set supplying (question, relevant-chunk) positives.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Base model id to fine-tune."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/boe-e5-small"),
        help="Directory to save the fine-tuned model.",
    )
    parser.add_argument(
        "--pairs-out",
        type=Path,
        default=None,
        help="Optional path to also dump the mined pairs as JSONL.",
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=DEFAULT_NUM_NEGATIVES,
        help="Hard negatives mined per positive.",
    )
    parser.add_argument(
        "--pool",
        type=int,
        default=DEFAULT_POOL,
        help="Candidate pool retrieved per question before filtering positives.",
    )
    parser.add_argument("--epochs", type=float, default=1.0, help="Training epochs.")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Per-device batch size."
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument(
        "--warmup-ratio", type=float, default=0.1, help="LR warmup fraction."
    )
    parser.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        help="Disable mixed precision (fp16 is on by default for GPU runs).",
    )
    parser.add_argument(
        "--logging-steps", type=int, default=50, help="Trainer logging interval."
    )
    parser.add_argument(
        "--device", default=None, help="Torch device (e.g. cuda); auto if omitted."
    )
    parser.set_defaults(fp16=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the fine-tune.

    Returns:
        Process exit code (0 on success, 1 if an input file is missing).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)
    if not args.corpus.is_file():
        print(f"ERROR: corpus not found: {args.corpus}", file=sys.stderr)
        return 1
    if not args.train_evalset.is_file():
        print(f"ERROR: eval set not found: {args.train_evalset}", file=sys.stderr)
        return 1
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
