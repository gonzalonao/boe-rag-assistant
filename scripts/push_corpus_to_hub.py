"""Publish a Parquet corpus and its dataset card to the Hugging Face Hub.

Requires the ``hub`` extra (``pip install -e .[hub]``) and an authenticated
session (``huggingface-cli login`` or the ``HF_TOKEN`` environment variable).

Example:
    python scripts/push_corpus_to_hub.py \
        --parquet data/corpus/boe-disposiciones.parquet \
        --repo-id gonzalonao/boe-corpus \
        --card docs/dataset_card.md
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from huggingface_hub import HfApi

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the publishing script."""
    parser = argparse.ArgumentParser(description="Push a BOE corpus to the HF Hub.")
    parser.add_argument(
        "--parquet", type=Path, required=True, help="Path to the corpus .parquet file."
    )
    parser.add_argument(
        "--repo-id", required=True, help="Target dataset repo, e.g. user/boe-corpus."
    )
    parser.add_argument(
        "--card",
        type=Path,
        default=Path("docs/dataset_card.md"),
        help="Markdown dataset card uploaded as the repo README.",
    )
    parser.add_argument(
        "--private", action="store_true", help="Create the dataset as private."
    )
    return parser


def push_corpus(parquet: Path, repo_id: str, card: Path, *, private: bool) -> None:
    """Create (if needed) the dataset repo and upload the corpus and its card.

    Args:
        parquet: Path to the corpus Parquet file.
        repo_id: Target Hugging Face dataset repository id.
        card: Path to the Markdown dataset card.
        private: Whether to create the repository as private.

    Raises:
        FileNotFoundError: If the Parquet file or the card does not exist.
    """
    if not parquet.is_file():
        raise FileNotFoundError(f"Corpus not found: {parquet}")
    if not card.is_file():
        raise FileNotFoundError(f"Dataset card not found: {card}")

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    logger.info("Uploading dataset card to %s", repo_id)
    api.upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    logger.info("Uploading corpus %s to %s", parquet.name, repo_id)
    api.upload_file(
        path_or_fileobj=str(parquet),
        path_in_repo=f"data/{parquet.name}",
        repo_id=repo_id,
        repo_type="dataset",
    )
    logger.info("Done: https://huggingface.co/datasets/%s", repo_id)


def main(argv: list[str] | None = None) -> int:
    """Run the publishing script.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        Process exit code (0 on success).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)
    push_corpus(args.parquet, args.repo_id, args.card, private=args.private)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
