"""Publish the evaluation set (seed + generated) and its card to the HF Hub.

Uploads the hand-curated ``seed`` split, the LLM-generated ``generated`` split,
and the dataset card as the repo README. Requires the ``hub`` extra
(``pip install -e .[hub]``) and an authenticated session (``huggingface-cli
login`` or the ``HF_TOKEN`` environment variable).

Example:
    python scripts/push_evalset_to_hub.py \
        --seed eval_data/seed_evalset.jsonl \
        --generated eval_data/generated_evalset.jsonl \
        --repo-id gonzalonao/boe-rag-evalset \
        --card docs/evalset_card.md
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from huggingface_hub import HfApi

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the publishing script."""
    parser = argparse.ArgumentParser(description="Push the BOE eval set to the HF Hub.")
    parser.add_argument(
        "--seed",
        type=Path,
        default=Path("eval_data/seed_evalset.jsonl"),
        help="Hand-curated gold split (uploaded as seed_evalset.jsonl).",
    )
    parser.add_argument(
        "--generated",
        type=Path,
        default=Path("eval_data/generated_evalset.jsonl"),
        help="LLM-generated silver split (uploaded as generated_evalset.jsonl).",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target dataset repo, e.g. user/boe-rag-evalset.",
    )
    parser.add_argument(
        "--card",
        type=Path,
        default=Path("docs/evalset_card.md"),
        help="Markdown dataset card uploaded as the repo README.",
    )
    parser.add_argument(
        "--private", action="store_true", help="Create the dataset as private."
    )
    return parser


def push_evalset(
    seed: Path,
    generated: Path,
    repo_id: str,
    card: Path,
    *,
    private: bool,
) -> None:
    """Create (if needed) the dataset repo and upload both splits and the card.

    Args:
        seed: Path to the hand-curated gold split JSONL.
        generated: Path to the LLM-generated silver split JSONL.
        repo_id: Target Hugging Face dataset repository id.
        card: Path to the Markdown dataset card.
        private: Whether to create the repository as private.

    Raises:
        FileNotFoundError: If any input file does not exist.
    """
    for path in (seed, generated, card):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    logger.info("Uploading dataset card to %s", repo_id)
    api.upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    for path in (seed, generated):
        logger.info("Uploading %s to %s", path.name, repo_id)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
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
    push_evalset(
        args.seed,
        args.generated,
        args.repo_id,
        args.card,
        private=args.private,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
