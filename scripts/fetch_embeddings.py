"""Download published precomputed E5 passage embeddings from the Hugging Face Hub.

Used by the CI eval-gate and the Docker build to load a precomputed embedding
matrix instead of re-encoding the whole corpus on CPU (which scales with the
corpus and dominates both jobs once the corpus is large). Picks the largest
``.npz`` in the dataset repo, robust to the exact published filename.

Requires the ``hub`` extra (``pip install -e .[hub]``). Public datasets need no
token; private ones use ``HF_TOKEN``.

Example:
    python scripts/fetch_embeddings.py --repo-id gonzalonao/boe-corpus \
        --out data/corpus/boe-embeddings.npz
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


def fetch_embeddings(repo_id: str, out: Path) -> Path:
    """Download the dataset's embeddings and copy the largest ``.npz`` to ``out``.

    Args:
        repo_id: Hugging Face dataset repo id, e.g. ``user/boe-corpus``.
        out: Destination path for the ``.npz`` file.

    Returns:
        The destination path.

    Raises:
        FileNotFoundError: If the dataset contains no ``.npz`` file.
    """
    logger.info("Downloading embeddings from %s ...", repo_id)
    local_dir = snapshot_download(
        repo_id, repo_type="dataset", allow_patterns=["*.npz"]
    )
    matrices = sorted(Path(local_dir).rglob("*.npz"), key=lambda p: p.stat().st_size)
    if not matrices:
        raise FileNotFoundError(f"No .npz file found in dataset {repo_id!r}")
    source = matrices[-1]  # largest = the full-corpus matrix, not a sample
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, out)
    logger.info(
        "Wrote embeddings %s (%d bytes) to %s", source.name, out.stat().st_size, out
    )
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the embeddings fetch.

    Returns:
        Process exit code (0 on success).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Fetch precomputed BOE embeddings from the HF Hub."
    )
    parser.add_argument(
        "--repo-id", default="gonzalonao/boe-corpus", help="Source dataset repo id."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/corpus/boe-embeddings.npz"),
        help="Destination .npz path.",
    )
    args = parser.parse_args(argv)
    try:
        fetch_embeddings(args.repo_id, args.out)
    except (FileNotFoundError, OSError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
