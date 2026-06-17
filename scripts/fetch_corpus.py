"""Download a published BOE corpus Parquet from the Hugging Face Hub.

Used at Docker build time to bake the corpus into the image (so the running
Space does no runtime download). Picks the largest ``.parquet`` in the dataset
repo, which is robust to the exact published filename.

Requires the ``hub`` extra (``pip install -e .[hub]``). Public datasets need no
token; private ones use ``HF_TOKEN``.

Example:
    python scripts/fetch_corpus.py --repo-id gonzalonao/boe-corpus \
        --out data/corpus/boe-2024.parquet
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


def fetch_corpus(repo_id: str, out: Path) -> Path:
    """Download the corpus dataset and copy its largest Parquet to ``out``.

    Args:
        repo_id: Hugging Face dataset repo id, e.g. ``user/boe-corpus``.
        out: Destination path for the Parquet file.

    Returns:
        The destination path.

    Raises:
        FileNotFoundError: If the dataset contains no ``.parquet`` file.
    """
    logger.info("Downloading %s from the Hub ...", repo_id)
    local_dir = snapshot_download(
        repo_id, repo_type="dataset", allow_patterns=["*.parquet"]
    )
    parquets = sorted(
        Path(local_dir).rglob("*.parquet"), key=lambda p: p.stat().st_size
    )
    if not parquets:
        raise FileNotFoundError(f"No .parquet file found in dataset {repo_id!r}")
    source = parquets[-1]  # largest = the full corpus, not a sample
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, out)
    logger.info(
        "Wrote corpus %s (%d bytes) to %s", source.name, out.stat().st_size, out
    )
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the corpus fetch.

    Returns:
        Process exit code (0 on success).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Fetch a BOE corpus from the HF Hub.")
    parser.add_argument(
        "--repo-id", default="gonzalonao/boe-corpus", help="Source dataset repo id."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/corpus/boe-2024.parquet"),
        help="Destination Parquet path.",
    )
    args = parser.parse_args(argv)
    try:
        fetch_corpus(args.repo_id, args.out)
    except (FileNotFoundError, OSError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
