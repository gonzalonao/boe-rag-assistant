"""Command-line entrypoint for running a corpus ingestion.

Example:
    python -m boe_rag.ingest --start 2024-01-15 --end 2024-01-31 \
        --out data/corpus/boe-2024-01.parquet
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime
from pathlib import Path

from boe_rag.config import IngestionConfig
from boe_rag.ingest.corpus import write_corpus
from boe_rag.ingest.pipeline import date_range, ingest_dates


def _parse_date(value: str) -> date:
    """Parse a ``YYYY-MM-DD`` CLI argument into a date."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}: {err}") from err


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ingestion CLI."""
    parser = argparse.ArgumentParser(
        prog="boe-ingest",
        description="Ingest BOE documents into a Parquet corpus.",
    )
    parser.add_argument("--start", type=_parse_date, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=_parse_date, required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination .parquet path for the corpus.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: INFO).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ingestion CLI.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        Process exit code: 0 on success, 1 if no chunks were produced.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = IngestionConfig()
    chunks = ingest_dates(date_range(args.start, args.end), config)
    written = write_corpus(chunks, args.out)
    if written == 0:
        logging.getLogger(__name__).error("No chunks produced; check the date range.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
