r"""Incrementally refresh the published corpus with the latest BOE issues (Phase 7).

This is the build half of the weekly auto-ingestion. It fetches the corpus and
its precomputed embeddings from the Hub, crawls a short trailing window of new
BOE issues, folds in only the genuinely new chunks, and embeds *only* those —
reusing every existing vector — so the decade already on the Hub is never
re-crawled or re-encoded. The refreshed Parquet and ``.npz`` are written to a
work directory under their published filenames (ready to overwrite in place), and
a JSON summary records how many chunks are new.

It deliberately does **not** publish or redeploy: the scheduled workflow
(`.github/workflows/refresh-corpus.yml`) runs the eval-gate on the output first
and only then republishes and rebuilds the Space, so a bad crawl can never reach
the live demo.

Requires the ``ml`` + ``hub`` extras (``pip install -e .[ml,hub]``) and, for the
fetch, network access (a token is only needed for a private dataset via
``HF_TOKEN``).

Example:
    python scripts/refresh_corpus.py \
        --corpus-repo gonzalonao/boe-corpus \
        --days 10 \
        --work-dir data/refresh \
        --summary-out reports/refresh_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from huggingface_hub import snapshot_download

from boe_rag.config import IngestionConfig
from boe_rag.eval.embedding import DEFAULT_MODEL, E5Embedder
from boe_rag.eval.retriever import load_embeddings, save_embeddings
from boe_rag.ingest.corpus import write_corpus
from boe_rag.ingest.incremental import (
    append_new_chunks,
    realign_embeddings,
    trailing_window,
)
from boe_rag.ingest.pipeline import date_range, ingest_dates

logger = logging.getLogger(__name__)


def _fetch_largest(repo_id: str, pattern: str, dest_dir: Path) -> Path:
    """Download the largest file matching ``pattern`` from a dataset repo.

    The published filename is preserved in ``dest_dir`` so a later re-push under
    the same name overwrites the canonical artifact in place rather than piling
    up stale copies.

    Args:
        repo_id: Hugging Face dataset repo id.
        pattern: Glob for the wanted files (e.g. ``"*.parquet"``).
        dest_dir: Local directory to copy the file into.

    Returns:
        The local path to the copied file.

    Raises:
        FileNotFoundError: If no file matches ``pattern`` in the repo.
    """
    local_dir = snapshot_download(
        repo_id, repo_type="dataset", allow_patterns=[pattern]
    )
    matches = sorted(Path(local_dir).rglob(pattern), key=lambda p: p.stat().st_size)
    if not matches:
        raise FileNotFoundError(f"No {pattern} in dataset {repo_id!r}")
    source = matches[-1]  # largest = the full corpus/matrix, not a sample
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source.name
    shutil.copyfile(source, dest)
    logger.info(
        "Fetched %s (%d bytes) from %s", source.name, dest.stat().st_size, repo_id
    )
    return dest


def _crawl_window(days: int, config: IngestionConfig, out: Path) -> pa.Table:
    """Crawl the trailing ``days``-day window and return the chunks as a table."""
    start, end = trailing_window(datetime.now(UTC).date(), days)
    logger.info("Crawling new BOE issues %s -> %s ...", start, end)
    chunks = ingest_dates(date_range(start, end), config)
    written = write_corpus(chunks, out)
    logger.info("Crawled %d chunk(s) in the window.", written)
    return pq.read_table(out)  # type: ignore[no-untyped-call]


def _embed_new(texts: list[str], model_name: str) -> object:
    """Encode the new passages with the E5 model (the only heavy step)."""
    logger.info("Embedding %d new passage(s) with %s ...", len(texts), model_name)
    embedder = E5Embedder(model_name=model_name)
    return embedder.embed_passages(texts)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Incrementally refresh the BOE corpus with the latest issues."
    )
    parser.add_argument(
        "--corpus-repo",
        default="gonzalonao/boe-corpus",
        help="Source/target HF dataset repo (default: gonzalonao/boe-corpus).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=10,
        help="Trailing window width in days; overlaps prior runs (default: 10).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("data/refresh"),
        help="Directory for fetched + refreshed artifacts (default: data/refresh).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Embedding model id for new chunks."
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("reports/refresh_summary.json"),
        help="Where to write the JSON run summary (default: reports/...).",
    )
    return parser


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    """Write the run summary JSON the workflow gates on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote refresh summary to %s", path)


def main(argv: list[str] | None = None) -> int:
    """Run the incremental refresh build step.

    Returns:
        0 on success (including the no-news case); 1 on a fetch/crawl error.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)

    try:
        corpus_in = _fetch_largest(args.corpus_repo, "*.parquet", args.work_dir)
        embeddings_in = _fetch_largest(args.corpus_repo, "*.npz", args.work_dir)
        existing = pq.read_table(corpus_in)  # type: ignore[no-untyped-call]
        crawled = _crawl_window(
            args.days, IngestionConfig(), args.work_dir / "crawled.parquet"
        )
    except (FileNotFoundError, OSError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1

    combined, new_ids = append_new_chunks(existing, crawled)
    summary: dict[str, object] = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "new_chunks": len(new_ids),
        "total_chunks": combined.num_rows,
        "corpus_repo": args.corpus_repo,
    }
    if not new_ids:
        logger.info("No new chunks in the window; nothing to publish.")
        _write_summary(args.summary_out, summary)
        return 0

    by_id = dict(
        zip(
            combined.column("chunk_id").to_pylist(),
            combined.column("text").to_pylist(),
            strict=True,
        )
    )
    new_texts = [str(by_id[cid]) for cid in new_ids]
    new_matrix = _embed_new(new_texts, args.model)
    old_ids, old_matrix = load_embeddings(embeddings_in)
    corpus_ids = [str(cid) for cid in combined.column("chunk_id").to_pylist()]
    matrix = realign_embeddings(
        corpus_ids, [(old_ids, old_matrix), (new_ids, new_matrix)]
    )

    corpus_out = args.work_dir / corpus_in.name
    embeddings_out = args.work_dir / embeddings_in.name
    pq.write_table(combined, corpus_out)  # type: ignore[no-untyped-call]
    save_embeddings(embeddings_out, corpus_ids, matrix)
    logger.info(
        "Refreshed corpus (%d chunks) and embeddings written to %s",
        combined.num_rows,
        args.work_dir,
    )

    summary["corpus_path"] = str(corpus_out)
    summary["embeddings_path"] = str(embeddings_out)
    _write_summary(args.summary_out, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
