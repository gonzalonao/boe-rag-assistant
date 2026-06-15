"""Precompute E5 passage embeddings for a corpus and save them to disk.

Encoding the whole corpus is the slow part of serving cold-start. Running this
once (at Docker build time, or locally) and shipping the resulting ``.npz`` lets
the service skip re-embedding every passage on boot — it just loads the matrix
and uses the model only for query encoding.

Requires the ``ml`` extra (``pip install -e .[ml]``).

Example:
    python scripts/precompute_embeddings.py \
        --corpus data/corpus/boe-2024.parquet \
        --out data/corpus/boe-2024-embeddings.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from boe_rag.eval.embedding import DEFAULT_MODEL, E5Embedder
from boe_rag.eval.retriever import save_embeddings

logger = logging.getLogger(__name__)


def precompute(corpus: Path, out: Path, model_name: str) -> Path:
    """Embed every passage in ``corpus`` and save the matrix to ``out``.

    Args:
        corpus: Path to the corpus Parquet (needs ``chunk_id`` and ``text``).
        out: Destination ``.npz`` path for the embeddings.
        model_name: Sentence-Transformers model id to embed with.

    Returns:
        The destination path.
    """
    table = pq.read_table(corpus, columns=["chunk_id", "text"])  # type: ignore[no-untyped-call]
    data = table.to_pydict()
    chunk_ids = [str(cid) for cid in data["chunk_id"]]
    texts = [str(text) for text in data["text"]]
    logger.info("Embedding %d passages with %s ...", len(texts), model_name)

    embedder = E5Embedder(model_name=model_name)
    matrix = embedder.embed_passages(texts)

    save_embeddings(out, chunk_ids, matrix)
    logger.info("Saved %d x %d embeddings to %s", *matrix.shape, out)
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the precompute step.

    Returns:
        Process exit code (0 on success).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Precompute corpus embeddings.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corpus/boe-2024.parquet"),
        help="Corpus Parquet path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/corpus/boe-2024-embeddings.npz"),
        help="Destination .npz path.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Sentence-Transformers model id."
    )
    args = parser.parse_args(argv)

    if not args.corpus.is_file():
        print(f"ERROR: corpus not found: {args.corpus}", file=sys.stderr)
        return 1
    precompute(args.corpus, args.out, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
