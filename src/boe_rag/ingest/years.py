"""Helpers for a resumable, year-by-year corpus ingestion.

A 2015-present crawl spans thousands of daily BOE issues, so doing it as one run
risks losing hours to a dropped connection. These helpers split the span into
per-year shards (one Parquet each, skippable when already present) so the crawl can
be resumed, and merge the shards into a single de-duplicated corpus Parquet.

The network crawl itself lives in :mod:`boe_rag.ingest.pipeline`; this module only
plans the spans and stitches the shards, so the logic here is pure and unit-tested
without hitting the BOE API.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

#: Corpus column that uniquely identifies a chunk (used to de-duplicate shards).
_CHUNK_ID = "chunk_id"


@dataclass(frozen=True, slots=True)
class YearSpan:
    """The inclusive date range to ingest for one calendar year.

    Attributes:
        year: The calendar year.
        start: First date to ingest (Jan 1, or later if the span is clamped).
        end: Last date to ingest (Dec 31, or ``until`` for the final year).
    """

    year: int
    start: date
    end: date


def year_spans(
    start_year: int, end_year: int, *, until: date | None = None
) -> list[YearSpan]:
    """Build one :class:`YearSpan` per year in ``[start_year, end_year]``.

    The final year is capped at ``until`` (typically today), so the current,
    partial year is not requested past the present.

    Args:
        start_year: First calendar year, inclusive.
        end_year: Last calendar year, inclusive.
        until: Optional last date to ingest; years entirely after it are dropped
            and the year containing it is truncated to it.

    Returns:
        The spans in chronological order.

    Raises:
        ValueError: If ``end_year`` precedes ``start_year``.
    """
    if end_year < start_year:
        raise ValueError(f"end_year {end_year} precedes start_year {start_year}")
    spans: list[YearSpan] = []
    for year in range(start_year, end_year + 1):
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        if until is not None and until < end:
            end = until
        if end < start:
            continue  # `until` falls before this year entirely; skip it.
        spans.append(YearSpan(year=year, start=start, end=end))
    return spans


def shard_path(out_dir: Path, year: int) -> Path:
    """Return the Parquet path for a year's shard under ``out_dir``."""
    return out_dir / f"boe-{year}.parquet"


def is_complete(path: Path) -> bool:
    """Whether a shard already exists with at least one row (so it can be skipped).

    A missing, empty, or unreadable/partial file is treated as not complete, so a
    resumed run re-ingests it rather than trusting a truncated crawl.

    Args:
        path: The shard Parquet path.

    Returns:
        ``True`` only when the file exists and has one or more rows.
    """
    if not path.is_file():
        return False
    try:
        metadata = pq.read_metadata(path)  # type: ignore[no-untyped-call]
        return int(metadata.num_rows) > 0
    except (OSError, pa.ArrowInvalid):
        logger.warning("Shard %s is unreadable; will re-ingest.", path)
        return False


def merge_shards(paths: Sequence[Path], out: Path) -> int:
    """Concatenate shard Parquets into one corpus, de-duplicating by ``chunk_id``.

    The first occurrence of each ``chunk_id`` wins, so overlapping shards (e.g. a
    re-ingested year) collapse cleanly. Column order is preserved from the shards.

    Args:
        paths: Shard Parquet paths, in the desired concatenation order.
        out: Destination corpus Parquet path.

    Returns:
        The number of unique chunks written.
    """
    tables = [
        pq.read_table(path)  # type: ignore[no-untyped-call]
        for path in paths
        if path.is_file()
    ]
    if not tables:
        logger.warning("No shards to merge; nothing written.")
        return 0
    combined = pa.concat_tables(tables)
    seen: set[str] = set()
    keep: list[bool] = []
    for chunk_id in combined.column(_CHUNK_ID).to_pylist():
        is_new = chunk_id not in seen
        keep.append(is_new)
        seen.add(chunk_id)
    deduped = combined.filter(pa.array(keep))
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(deduped, out)  # type: ignore[no-untyped-call]
    logger.info(
        "Merged %d shard(s) into %s: %d unique chunks (%d duplicates dropped).",
        len(tables),
        out,
        deduped.num_rows,
        combined.num_rows - deduped.num_rows,
    )
    return int(deduped.num_rows)
