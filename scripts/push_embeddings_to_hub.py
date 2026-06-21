"""Publish precomputed E5 passage embeddings to the corpus's HF dataset repo.

The embeddings ``.npz`` lives alongside the corpus Parquet in the same dataset
repo so the CI eval-gate and the Docker build can load it instead of re-encoding
the corpus on CPU. Republish whenever the corpus (or the embedding model) changes
so the matrix stays in lock-step with the published Parquet.

Requires the ``hub`` extra (``pip install -e .[hub]``) and an authenticated
session (``HF_TOKEN`` or ``huggingface-cli login``).

Example:
    python scripts/push_embeddings_to_hub.py \
        --embeddings data/corpus/boe-2015-present-embeddings.npz \
        --repo-id gonzalonao/boe-corpus
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from huggingface_hub import HfApi

logger = logging.getLogger(__name__)


def push_embeddings(embeddings: Path, repo_id: str) -> None:
    """Upload the embeddings ``.npz`` under ``embeddings/`` in the dataset repo.

    Args:
        embeddings: Path to the precomputed ``.npz`` file.
        repo_id: Target Hugging Face dataset repository id.

    Raises:
        FileNotFoundError: If the ``.npz`` file does not exist.
    """
    if not embeddings.is_file():
        raise FileNotFoundError(f"Embeddings not found: {embeddings}")

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    logger.info("Uploading embeddings %s to %s", embeddings.name, repo_id)
    api.upload_file(
        path_or_fileobj=str(embeddings),
        path_in_repo=f"embeddings/{embeddings.name}",
        repo_id=repo_id,
        repo_type="dataset",
    )
    logger.info("Done: https://huggingface.co/datasets/%s", repo_id)


def main(argv: list[str] | None = None) -> int:
    """Run the publishing script.

    Returns:
        Process exit code (0 on success).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Push precomputed BOE embeddings to the HF Hub."
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        required=True,
        help="Path to the precomputed .npz file.",
    )
    parser.add_argument(
        "--repo-id",
        default="gonzalonao/boe-corpus",
        help="Target dataset repo, e.g. user/boe-corpus.",
    )
    args = parser.parse_args(argv)
    push_embeddings(args.embeddings, args.repo_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
