"""Pre-download the serving models into the local Hugging Face cache.

Run at Docker build time so the embedding and cross-encoder weights are baked
into the image. The running Space then loads them from cache instead of
downloading on first request, removing that from the cold-start path.

Requires the ``ml`` extra (``pip install -e .[ml]``).

Example:
    python scripts/warm_models.py
"""

from __future__ import annotations

import logging

from boe_rag.eval.cross_encoder import CrossEncoderReranker
from boe_rag.eval.embedding import E5Embedder

logger = logging.getLogger(__name__)


def warm_models() -> None:
    """Instantiate each serving model so its weights are fetched and cached."""
    logger.info("Warming the embedding model ...")
    E5Embedder()
    logger.info("Warming the cross-encoder model ...")
    CrossEncoderReranker()
    logger.info("Models cached.")


def main() -> int:
    """Download and cache the serving models.

    Returns:
        Process exit code (0 on success).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    warm_models()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
