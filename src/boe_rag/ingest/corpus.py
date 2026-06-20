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

#: Canonical on-disk schema, pinned so every shard and corpus has identical column
#: types. Without it, a slice in which an optional column (e.g. ``titulo``) is
#: entirely null makes pyarrow infer a ``null``-typed column, which then fails to
#: concatenate against a ``string``-typed column from another slice.
CORPUS_SCHEMA: pa.Schema = pa.schema(
    [
        ("chunk_id", pa.string()),
        ("document_id", pa.string()),
        ("document_title", pa.string()),
        ("text", pa.string()),
        ("ordinal", pa.int64()),
        ("titulo", pa.string()),
        ("capitulo", pa.string()),
        ("seccion", pa.string()),
        ("articulo", pa.string()),
        ("citation", pa.string()),
        ("url_html", pa.string()),
    ]
)

#: Column order of the on-disk corpus, kept stable for reproducible datasets.
_COLUMNS: tuple[str, ...] = tuple(CORPUS_SCHEMA.names)


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
    # Pin the schema so an all-null optional column never collapses to a `null`
    # type that breaks cross-shard concatenation downstream.
    table = pa.table(columns, schema=CORPUS_SCHEMA)
    # pyarrow ships stubs but leaves these functions untyped; isolate the gap.
    pq.write_table(table, path)  # type: ignore[no-untyped-call]
    logger.info("Wrote %d chunks to %s", len(records), path)
    return len(records)
