"""Tests for the structure-aware chunker."""

from __future__ import annotations

from boe_rag.config import IngestionConfig
from boe_rag.ingest.chunker import chunk_document
from boe_rag.ingest.parser import parse_document
from boe_rag.models import BoeDocument, DocumentMetadata, TextBlock


def test_chunks_are_article_scoped(structured_xml: str) -> None:
    """Each article becomes its own chunk carrying the full hierarchy."""
    doc = parse_document(structured_xml)
    chunks = chunk_document(doc)
    articulos = [c.articulo for c in chunks]
    assert articulos == [
        "Artículo 1. Objeto de la Ley.",
        "Artículo 2. Ámbito subjetivo de aplicación.",
        "Artículo 3. Capacidad de obrar.",
    ]


def test_hierarchy_context_is_tracked(structured_xml: str) -> None:
    """A deep article carries its título and capítulo context."""
    doc = parse_document(structured_xml)
    chunks = chunk_document(doc)
    art3 = next(c for c in chunks if c.articulo and c.articulo.startswith("Artículo 3"))
    assert art3.titulo == "TÍTULO I - De los interesados en el procedimiento"
    assert (
        art3.capitulo
        == "CAPÍTULO I - La capacidad de obrar y el concepto de interesado"
    )


def test_first_article_keeps_titulo_preliminar(structured_xml: str) -> None:
    """The first article belongs to the preliminary title, with no capítulo."""
    doc = parse_document(structured_xml)
    chunks = chunk_document(doc)
    art1 = next(c for c in chunks if c.articulo and c.articulo.startswith("Artículo 1"))
    assert art1.titulo == "TÍTULO PRELIMINAR - Disposiciones generales"
    assert art1.capitulo is None


def test_citation_uses_norm_reference_and_article(structured_xml: str) -> None:
    """Citations combine the norm reference with a short article label."""
    doc = parse_document(structured_xml)
    chunks = chunk_document(doc)
    art1 = next(c for c in chunks if c.articulo and c.articulo.startswith("Artículo 1"))
    assert art1.citation == "Ley 39/2015, Artículo 1"


def test_chunk_ids_are_deterministic_and_ordered(structured_xml: str) -> None:
    """Chunk ids are stable and follow the document order."""
    doc = parse_document(structured_xml)
    chunks = chunk_document(doc)
    assert [c.chunk_id for c in chunks] == [
        "BOE-A-2015-10565::0000",
        "BOE-A-2015-10565::0001",
        "BOE-A-2015-10565::0002",
    ]
    assert [c.ordinal for c in chunks] == [0, 1, 2]


def test_article_body_is_included_in_text(structured_xml: str) -> None:
    """Article body paragraphs are concatenated into the chunk text."""
    doc = parse_document(structured_xml)
    chunks = chunk_document(doc)
    art2 = next(c for c in chunks if c.articulo and c.articulo.startswith("Artículo 2"))
    assert "Ámbito subjetivo" in art2.text
    assert "Comunidades Autónomas" in art2.text


def test_document_without_articles_still_chunks(simple_xml: str) -> None:
    """A flat resolution yields a chunk with no article context."""
    doc = parse_document(simple_xml)
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.articulo is None
    assert chunk.citation == "Resolución"
    assert "precios de venta" in chunk.text


def _doc_with_long_article(body_paragraphs: list[str]) -> BoeDocument:
    """Build a one-article document with the given body paragraphs."""
    blocks = [TextBlock(css_class="articulo", text="Artículo 1. Prueba.")]
    blocks += [TextBlock(css_class="parrafo", text=p) for p in body_paragraphs]
    meta = DocumentMetadata(
        identifier="BOE-A-2024-1", title="Documento de prueba", rango="Orden"
    )
    return BoeDocument(metadata=meta, blocks=tuple(blocks))


def test_long_article_is_split_into_parts() -> None:
    """An article exceeding the size limit is split into multiple chunks."""
    paragraphs = [
        f"Párrafo número {i} con suficiente longitud textual." for i in range(8)
    ]
    doc = _doc_with_long_article(paragraphs)
    config = IngestionConfig(max_chunk_chars=120, min_chunk_chars=10)
    chunks = chunk_document(doc, config)
    assert len(chunks) > 1
    # Every split part keeps the same article scope.
    assert all(c.articulo == "Artículo 1. Prueba." for c in chunks)
    assert all(len(c.text) <= 200 for c in chunks)


def test_tiny_trailing_fragment_merges_back() -> None:
    """A sub-minimum trailing split merges into the previous same-scope chunk."""
    paragraphs = [
        "Un párrafo razonablemente largo que supera el umbral mínimo con holgura.",
        "x",
    ]
    doc = _doc_with_long_article(paragraphs)
    config = IngestionConfig(max_chunk_chars=80, min_chunk_chars=30)
    chunks = chunk_document(doc, config)
    # The lone 'x' fragment must not survive as its own chunk.
    assert all(len(c.text) >= 30 for c in chunks)
    assert chunks[-1].text.endswith("x")
