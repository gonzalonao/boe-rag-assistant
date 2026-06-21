"""Tests for the year-by-year ingestion planning and shard-merge helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from boe_rag.ingest.years import (
    YearSpan,
    is_complete,
    merge_shards,
    shard_path,
    year_spans,
)


def _write_shard(
    path: Path, chunk_ids: list[str], *, titulo_null: bool = False
) -> None:
    """Write a corpus-shaped Parquet shard with the given chunk ids.

    With ``titulo_null`` the optional ``titulo`` column is all-None and written
    *without* an explicit schema, so pyarrow infers a ``null`` type — reproducing
    the cross-shard schema mismatch the merge must tolerate.
    """
    n = len(chunk_ids)
    table = pa.table(
        {
            "chunk_id": chunk_ids,
            "document_id": ["doc"] * n,
            "document_title": ["Title"] * n,
            "text": [f"text-{cid}" for cid in chunk_ids],
            "ordinal": list(range(n)),
            "titulo": [None] * n if titulo_null else ["Titulo I"] * n,
            "capitulo": [None] * n,
            "seccion": [None] * n,
            "articulo": ["Articulo 1"] * n,
            "citation": ["Cite"] * n,
            "url_html": ["http://x"] * n,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)  # type: ignore[no-untyped-call]


# --- year_spans -----------------------------------------------------------


def test_year_spans_full_years() -> None:
    """Each whole year spans Jan 1 to Dec 31."""
    spans = year_spans(2015, 2016)
    assert spans == [
        YearSpan(2015, date(2015, 1, 1), date(2015, 12, 31)),
        YearSpan(2016, date(2016, 1, 1), date(2016, 12, 31)),
    ]


def test_year_spans_caps_final_year_at_until() -> None:
    """The last year is truncated to the `until` date."""
    spans = year_spans(2024, 2026, until=date(2026, 6, 20))
    assert spans[-1] == YearSpan(2026, date(2026, 1, 1), date(2026, 6, 20))


def test_year_spans_drops_years_after_until() -> None:
    """Years entirely after `until` are omitted."""
    spans = year_spans(2024, 2030, until=date(2025, 3, 1))
    assert [s.year for s in spans] == [2024, 2025]
    assert spans[-1].end == date(2025, 3, 1)


def test_year_spans_rejects_reversed_range() -> None:
    """end_year before start_year is an error."""
    with pytest.raises(ValueError, match="precedes"):
        year_spans(2026, 2015)


# --- is_complete ----------------------------------------------------------


def test_is_complete_false_when_missing(tmp_path: Path) -> None:
    """A missing shard is not complete."""
    assert not is_complete(tmp_path / "boe-2015.parquet")


def test_is_complete_true_when_rows_present(tmp_path: Path) -> None:
    """A shard with rows counts as complete."""
    path = tmp_path / "boe-2015.parquet"
    _write_shard(path, ["a::0001", "a::0002"])
    assert is_complete(path)


def test_is_complete_false_when_empty(tmp_path: Path) -> None:
    """A zero-row shard is treated as not complete (re-ingest it)."""
    path = tmp_path / "boe-2015.parquet"
    _write_shard(path, [])
    assert not is_complete(path)


def test_is_complete_false_when_unreadable(tmp_path: Path) -> None:
    """A corrupt/partial file is treated as not complete, not an error."""
    path = tmp_path / "boe-2015.parquet"
    path.write_bytes(b"not a parquet file")
    assert not is_complete(path)


# --- merge_shards ---------------------------------------------------------


def test_merge_shards_deduplicates_by_chunk_id(tmp_path: Path) -> None:
    """Overlapping shards collapse on chunk_id, first occurrence wins."""
    a = tmp_path / "boe-2015.parquet"
    b = tmp_path / "boe-2016.parquet"
    _write_shard(a, ["x::0001", "x::0002"])
    _write_shard(b, ["x::0002", "x::0003"])  # x::0002 overlaps
    out = tmp_path / "merged.parquet"

    total = merge_shards([a, b], out)

    assert total == 3
    merged = pq.read_table(out).to_pydict()  # type: ignore[no-untyped-call]
    assert merged["chunk_id"] == ["x::0001", "x::0002", "x::0003"]


def test_merge_shards_unifies_null_typed_columns(tmp_path: Path) -> None:
    """An all-null column in one shard still merges (schema-mismatch regression)."""
    a = tmp_path / "boe-2015.parquet"
    b = tmp_path / "boe-2016.parquet"
    _write_shard(a, ["x::0001"], titulo_null=True)  # `titulo` inferred as null type
    _write_shard(b, ["y::0001"])  # `titulo` is string
    out = tmp_path / "merged.parquet"

    total = merge_shards([a, b], out)

    assert total == 2
    merged = pq.read_table(out)  # type: ignore[no-untyped-call]
    assert merged.schema.field("titulo").type == pa.string()


def test_merge_shards_skips_missing_paths(tmp_path: Path) -> None:
    """Absent shard paths are ignored, present ones still merge."""
    a = tmp_path / "boe-2015.parquet"
    _write_shard(a, ["x::0001"])
    out = tmp_path / "merged.parquet"
    total = merge_shards([a, tmp_path / "absent.parquet"], out)
    assert total == 1


def test_merge_shards_no_inputs_writes_nothing(tmp_path: Path) -> None:
    """With no existing shards, nothing is written and zero is returned."""
    out = tmp_path / "merged.parquet"
    assert merge_shards([tmp_path / "absent.parquet"], out) == 0
    assert not out.exists()


def test_shard_path_format(tmp_path: Path) -> None:
    """Shard paths follow the boe-YYYY.parquet convention."""
    assert shard_path(tmp_path, 2019).name == "boe-2019.parquet"
