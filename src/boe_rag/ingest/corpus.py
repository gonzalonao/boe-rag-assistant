"""Serialise chunk streams to a Parquet corpus.

Parquet is the lingua franca of the Hugging Face Hub datasets viewer, so the
corpus is written in that format. This module isolates the only third-party
typing gap in the package (``pyarrow`` ships incomplete stubs) behind a small,
fully annotated surface.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from boe_rag.models import Chunk

logger = logging.getLogger(__name__)

#: Column order of the on-disk corpus, kept stable for reproducible datasets.
_COLUMNS: tuple[str, ...] = (
    "chunk_id",
    "document_id",
    "document_title",
    "text",
    "ordinal",
    "titulo",
    "capitulo",
    "seccion",
    "articulo",
    "citation",
    "url_html",
)


def write_corpus(chunks: Iterable[Chunk], path: Path) -> int:
    """Write chunks to a Parquet file, creating parent directories as needed.

    Args:
        chunks: The chunks to serialise.
        path: Destination ``.parquet`` path.

    Returns:
        The number of chunks written.
    """
    records = [chunk.model_dump() for chunk in chunks]
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {name: [record[name] for record in records] for name in _COLUMNS}
    table = pa.table(columns)
    # pyarrow ships stubs but leaves these functions untyped; isolate the gap.
    pq.write_table(table, path)  # type: ignore[no-untyped-call]
    logger.info("Wrote %d chunks to %s", len(records), path)
    return len(records)
