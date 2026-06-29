"""Tests for the incremental-refresh helpers (no BOE API or model needed)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pyarrow as pa
import pytest

from boe_rag.ingest.corpus import CORPUS_SCHEMA
from boe_rag.ingest.incremental import (
    append_new_chunks,
    realign_embeddings,
    trailing_window,
)


def _corpus(chunk_ids: list[str]) -> pa.Table:
    """Build a minimal corpus table with the canonical schema."""
    n = len(chunk_ids)
    columns: dict[str, list[object]] = {name: [""] * n for name in CORPUS_SCHEMA.names}
    columns["chunk_id"] = list(chunk_ids)
    columns["text"] = [f"text-{cid}" for cid in chunk_ids]
    columns["ordinal"] = list(range(n))
    return pa.table(columns, schema=CORPUS_SCHEMA)


def test_trailing_window_is_inclusive_and_ends_today() -> None:
    """A 10-day window spans today and the nine prior days, inclusive."""
    start, end = trailing_window(date(2026, 6, 29), days=10)
    assert end == date(2026, 6, 29)
    assert start == date(2026, 6, 20)


def test_trailing_window_rejects_nonpositive_days() -> None:
    """A non-positive window is a programming error, not a silent no-op."""
    with pytest.raises(ValueError, match="positive"):
        trailing_window(date(2026, 6, 29), days=0)


def test_append_new_chunks_adds_only_unseen_ids() -> None:
    """Overlapping crawls contribute only their genuinely new chunk ids."""
    existing = _corpus(["a", "b"])
    crawled = _corpus(["b", "c", "d"])  # b overlaps; c, d are new

    combined, new_ids = append_new_chunks(existing, crawled)

    assert new_ids == ["c", "d"]
    assert combined.column("chunk_id").to_pylist() == ["a", "b", "c", "d"]


def test_append_new_chunks_preserves_existing_order_prefix() -> None:
    """Existing rows stay first and in order, so their embeddings still align."""
    existing = _corpus(["a", "b", "c"])
    crawled = _corpus(["d"])

    combined, _ = append_new_chunks(existing, crawled)

    assert combined.column("chunk_id").to_pylist()[:3] == ["a", "b", "c"]


def test_append_new_chunks_dedupes_within_crawl() -> None:
    """A chunk id repeated within the same crawl is added at most once."""
    existing = _corpus(["a"])
    crawled = _corpus(["b", "b"])

    combined, new_ids = append_new_chunks(existing, crawled)

    assert new_ids == ["b"]
    assert combined.column("chunk_id").to_pylist() == ["a", "b"]


def test_append_new_chunks_no_news_returns_existing() -> None:
    """When nothing is new the existing corpus is returned untouched."""
    existing = _corpus(["a", "b"])
    crawled = _corpus(["a", "b"])

    combined, new_ids = append_new_chunks(existing, crawled)

    assert new_ids == []
    assert combined.num_rows == 2


def test_realign_embeddings_prefers_earlier_source_and_orders_by_corpus() -> None:
    """Reused vectors come from the first source; rows follow the corpus order."""
    old = (["a", "b"], np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32))
    delta = (["c"], np.array([[3.0, 3.0]], dtype=np.float32))

    matrix = realign_embeddings(["a", "b", "c"], [old, delta])

    assert matrix.shape == (3, 2)
    np.testing.assert_array_equal(matrix[2], np.array([3.0, 3.0], dtype=np.float32))
    # 'a' is taken from the first source even if a later one also had it.
    np.testing.assert_array_equal(matrix[0], np.array([1.0, 1.0], dtype=np.float32))


def test_realign_embeddings_raises_on_missing_id() -> None:
    """A corpus id with no embedding anywhere is a hard error, not a silent gap."""
    old = (["a"], np.array([[1.0, 1.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="no embedding"):
        realign_embeddings(["a", "b"], [old])
