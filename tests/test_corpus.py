"""Tests for the Parquet corpus writer."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from boe_rag.ingest.corpus import write_corpus
from boe_rag.models import Chunk


def _chunk(ordinal: int) -> Chunk:
    """Build a minimal valid chunk for serialisation tests."""
    return Chunk(
        chunk_id=f"BOE-A-2024-1::{ordinal:04d}",
        document_id="BOE-A-2024-1",
        document_title="Documento de prueba",
        text=f"Contenido del fragmento {ordinal}.",
        ordinal=ordinal,
        titulo="TÍTULO I",
        capitulo=None,
        seccion=None,
        articulo=f"Artículo {ordinal + 1}.",
        citation=f"Orden, Artículo {ordinal + 1}",
        url_html="https://example/doc.html",
    )


def test_write_corpus_roundtrip(tmp_path: Path) -> None:
    """Chunks written to Parquet read back with the same content and schema."""
    chunks = [_chunk(0), _chunk(1)]
    out = tmp_path / "nested" / "corpus.parquet"

    count = write_corpus(chunks, out)

    assert count == 2
    assert out.exists()
    table = pq.read_table(out)  # type: ignore[no-untyped-call]
    assert table.num_rows == 2
    assert table.column_names[0] == "chunk_id"
    texts = table.column("text").to_pylist()
    assert texts == ["Contenido del fragmento 0.", "Contenido del fragmento 1."]
    # Nullable columns survive as nulls, not as the string "None".
    assert table.column("capitulo").to_pylist() == [None, None]


def test_write_corpus_creates_parent_dirs(tmp_path: Path) -> None:
    """Missing parent directories are created."""
    out = tmp_path / "a" / "b" / "c.parquet"
    write_corpus([_chunk(0)], out)
    assert out.exists()
