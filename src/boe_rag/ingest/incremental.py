"""Pure helpers for an incremental, scheduled corpus refresh (Phase 7).

The weekly refresh crawls a short trailing window of new BOE issues and folds the
new chunks into the published corpus without re-crawling or re-embedding the
decade already on the Hub. This module holds the dependency-free core of that
flow — the date-window plan, the chunk-id-keyed append, and the embedding
realignment — so it is unit-tested without touching the BOE API or a model.

The network crawl lives in :mod:`boe_rag.ingest.pipeline` and the embedding model
in :mod:`boe_rag.eval.embedding`; the orchestration that stitches them together
is :mod:`scripts.refresh_corpus`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from boe_rag.ingest.corpus import CORPUS_SCHEMA

logger = logging.getLogger(__name__)

#: Corpus column that uniquely (and stably) identifies a chunk.
_CHUNK_ID = "chunk_id"

#: A matrix of L2-normalised row vectors (mirrors ``eval.retriever.FloatMatrix``).
FloatMatrix = npt.NDArray[np.float32]


def trailing_window(today: date, days: int) -> tuple[date, date]:
    """Return the ``[start, today]`` window to crawl on a refresh run.

    The window deliberately overlaps previous runs (``days`` is larger than the
    weekly cadence) so a missed or late run never leaves a gap; the chunk-id
    de-duplication in :func:`append_new_chunks` makes the overlap free.

    Args:
        today: The run date (typically ``datetime.now(UTC).date()``).
        days: Width of the trailing window in days; must be positive.

    Returns:
        The inclusive ``(start, end)`` dates, with ``end == today``.

    Raises:
        ValueError: If ``days`` is not positive.
    """
    if days <= 0:
        raise ValueError(f"days must be positive, got {days}")
    return today - timedelta(days=days - 1), today


def append_new_chunks(
    existing: pa.Table, crawled: pa.Table
) -> tuple[pa.Table, list[str]]:
    """Fold freshly crawled chunks into the existing corpus, deduped by id.

    Rows of ``crawled`` whose ``chunk_id`` is already present in ``existing`` (or
    repeated within ``crawled``) are dropped; the survivors are appended after
    the existing rows, so the existing order — and therefore its precomputed
    embedding rows — is preserved unchanged.

    Both tables are coerced to :data:`~boe_rag.ingest.corpus.CORPUS_SCHEMA` before
    concatenation so a shard with an all-null optional column cannot break the
    concat on a type mismatch.

    Args:
        existing: The current published corpus.
        crawled: Newly crawled chunks (may overlap ``existing`` or repeat ids).

    Returns:
        A ``(combined, new_ids)`` pair: the appended corpus table and the list of
        newly added chunk ids in append order (empty when nothing is new).
    """
    existing = existing.select(list(CORPUS_SCHEMA.names)).cast(CORPUS_SCHEMA)
    crawled = crawled.select(list(CORPUS_SCHEMA.names)).cast(CORPUS_SCHEMA)

    seen: set[str] = {str(cid) for cid in existing.column(_CHUNK_ID).to_pylist()}
    keep: list[bool] = []
    new_ids: list[str] = []
    for cid in crawled.column(_CHUNK_ID).to_pylist():
        chunk_id = str(cid)
        is_new = chunk_id not in seen
        keep.append(is_new)
        if is_new:
            seen.add(chunk_id)
            new_ids.append(chunk_id)

    if not new_ids:
        return existing, []
    new_rows = crawled.filter(pa.array(keep))
    combined = pa.concat_tables([existing, new_rows])
    logger.info(
        "Appended %d new chunk(s); corpus grew %d -> %d.",
        len(new_ids),
        existing.num_rows,
        combined.num_rows,
    )
    return combined, new_ids


def realign_embeddings(
    corpus_ids: Sequence[str],
    sources: Sequence[tuple[Sequence[str], FloatMatrix]],
) -> FloatMatrix:
    """Assemble one embedding matrix aligned row-for-row with ``corpus_ids``.

    Each corpus id takes its vector from the first source that carries it, so the
    refresh can pass the existing precomputed matrix first and the freshly encoded
    delta second: reused passages keep their stored vectors and only the new ones
    are encoded. The result matches the corpus order, which the serving path
    requires (a mismatch makes it re-encode the whole corpus).

    Args:
        corpus_ids: The chunk ids of the combined corpus, in on-disk order.
        sources: ``(ids, matrix)`` pairs searched in priority order; within a
            source ``ids`` is aligned row-for-row with ``matrix``.

    Returns:
        A matrix with one row per id in ``corpus_ids``.

    Raises:
        ValueError: If any id in ``corpus_ids`` is absent from every source.
    """
    location: dict[str, tuple[FloatMatrix, int]] = {}
    for ids, matrix in sources:
        for row, cid in enumerate(ids):
            location.setdefault(str(cid), (matrix, row))

    missing = [cid for cid in corpus_ids if cid not in location]
    if missing:
        raise ValueError(
            f"{len(missing)} corpus id(s) have no embedding "
            f"(first: {missing[0]!r}); cannot realign."
        )
    rows = [location[cid][0][location[cid][1]] for cid in corpus_ids]
    return np.asarray(np.vstack(rows), dtype=np.float32)
