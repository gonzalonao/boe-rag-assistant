"""Trigger a factory rebuild of the Hugging Face Space (Phase 7 go-live step).

The Space bakes the corpus and its embeddings into the image at *build* time
(see the Dockerfile's ``fetch_corpus``/``fetch_embeddings`` step), so a republished
dataset only reaches the live demo when the image is rebuilt. A factory reboot
re-runs the Dockerfile, which re-fetches the freshly published corpus.

Used by the weekly refresh workflow *after* the eval-gate has passed and the new
corpus has been published. Requires the ``hub`` extra and an ``HF_TOKEN`` write
token in the environment.

Example:
    python scripts/redeploy_space.py --space-id gonzalonao/boe-rag-assistant
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from huggingface_hub import HfApi

logger = logging.getLogger(__name__)


def redeploy(space_id: str) -> None:
    """Factory-reboot the Space so it rebuilds with the latest published corpus.

    Args:
        space_id: The Space repo id, e.g. ``user/boe-rag-assistant``.
    """
    api = HfApi()
    logger.info("Requesting a factory rebuild of Space %s ...", space_id)
    api.restart_space(repo_id=space_id, factory_reboot=True)
    logger.info("Rebuild requested: https://huggingface.co/spaces/%s", space_id)


def main(argv: list[str] | None = None) -> int:
    """Run the redeploy.

    Returns:
        0 on success; 1 when no ``HF_TOKEN`` is configured.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Factory-reboot the HF Space.")
    parser.add_argument(
        "--space-id",
        default="gonzalonao/boe-rag-assistant",
        help="Target Space repo id (default: gonzalonao/boe-rag-assistant).",
    )
    args = parser.parse_args(argv)

    if not os.environ.get("HF_TOKEN"):
        print("ERROR: HF_TOKEN is not set; cannot rebuild the Space.", file=sys.stderr)
        return 1
    redeploy(args.space_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
