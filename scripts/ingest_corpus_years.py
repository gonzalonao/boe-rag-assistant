r"""Resumable, year-by-year BOE corpus ingestion for the multi-year expansion.

Crawls Disposiciones Generales one calendar year at a time, writing a Parquet
shard per year (``boe-YYYY.parquet``). A year whose shard already exists with rows
is skipped, so a run interrupted by a dropped connection resumes where it left off.
With ``--merged-out`` the shards are then stitched into a single de-duplicated
corpus Parquet ready to publish, embed, and serve.

Examples:
    # Crawl 2015 through today, one shard per year, then merge.
    python scripts/ingest_corpus_years.py --start-year 2015 --end-year 2026 \\
        --out-dir data/corpus/years --merged-out data/corpus/boe-2015-2026.parquet

    # Just merge already-crawled shards (no network).
    python scripts/ingest_corpus_years.py --start-year 2015 --end-year 2026 \\
        --out-dir data/corpus/years --merged-out data/corpus/boe-2015-2026.parquet \\
        --merge-only
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, date, datetime
from pathlib import Path

from boe_rag.config import IngestionConfig
from boe_rag.ingest.corpus import write_corpus
from boe_rag.ingest.pipeline import date_range, ingest_dates
from boe_rag.ingest.years import is_complete, merge_shards, shard_path, year_spans

logger = logging.getLogger(__name__)


def _parse_date(value: str) -> date:
    """Parse a ``YYYY-MM-DD`` CLI argument into a date."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}: {err}") from err


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the year-by-year ingestion CLI."""
    parser = argparse.ArgumentParser(
        prog="ingest-corpus-years",
        description="Resumable year-by-year BOE corpus ingestion.",
    )
    parser.add_argument("--start-year", type=int, required=True, help="First year.")
    parser.add_argument("--end-year", type=int, required=True, help="Last year.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/corpus/years"),
        help="Directory for per-year shards (default: data/corpus/years).",
    )
    parser.add_argument(
        "--until",
        type=_parse_date,
        default=None,
        help="Cap ingestion at this YYYY-MM-DD (default: today, UTC).",
    )
    parser.add_argument(
        "--merged-out",
        type=Path,
        default=None,
        help="If set, merge all shards into this corpus Parquet after crawling.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip crawling; only merge existing shards (requires --merged-out).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: INFO).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the year-by-year ingestion (and optional merge).

    Returns:
        Process exit code: 0 on success, 2 on a usage error.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.merge_only and args.merged_out is None:
        logger.error("--merge-only requires --merged-out.")
        return 2

    until = args.until or datetime.now(UTC).date()
    spans = year_spans(args.start_year, args.end_year, until=until)
    config = IngestionConfig()

    if not args.merge_only:
        for span in spans:
            target = shard_path(args.out_dir, span.year)
            if is_complete(target):
                logger.info(
                    "Year %d already ingested (%s); skipping.", span.year, target
                )
                continue
            logger.info("Ingesting %d: %s -> %s ...", span.year, span.start, span.end)
            chunks = ingest_dates(date_range(span.start, span.end), config)
            written = write_corpus(chunks, target)
            logger.info("Year %d done: %d chunks.", span.year, written)

    if args.merged_out is not None:
        shards = [shard_path(args.out_dir, span.year) for span in spans]
        total = merge_shards(shards, args.merged_out)
        logger.info("Merged corpus: %d unique chunks at %s.", total, args.merged_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
