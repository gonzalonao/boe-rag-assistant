"""Pydantic data models for the BOE ingestion pipeline.

These models are the contract between the pipeline stages: the parser produces
them, the chunker consumes documents and emits chunks, and later phases
(indexing, retrieval) read chunks. Keeping them strict and explicit means a
malformed upstream payload fails loudly here rather than corrupting the corpus.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class SumarioItem(BaseModel):
    """A single document entry as listed in a daily BOE sumario.

    This is the lightweight index record; the full text is fetched separately
    via :attr:`url_xml`.

    Attributes:
        identifier: Stable BOE document id, e.g. ``BOE-A-2024-714``.
        title: Official document title as shown in the sumario.
        url_xml: URL of the document's machine-readable XML.
        url_html: URL of the document's human-readable HTML page.
        section_code: Code of the section the item appears under.
        section_name: Human-readable section name.
        department_code: Code of the issuing department/ministry.
        department_name: Human-readable issuing department name.
    """

    model_config = ConfigDict(frozen=True)

    identifier: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url_xml: str
    url_html: str
    section_code: str
    section_name: str
    department_code: str
    department_name: str


class DocumentMetadata(BaseModel):
    """Structured metadata extracted from a document's ``<metadatos>`` block.

    Attributes:
        identifier: Stable BOE document id.
        title: Full official title.
        rango: Legal rank/type (e.g. ``Ley``, ``Real Decreto``, ``Resolución``).
        department: Issuing department/ministry name.
        publication_date: Date the document was published in the BOE.
        disposition_date: Date the disposition itself was signed, when present.
        official_number: Official number of the norm (e.g. ``39/2015``), if any.
        section_code: Section code within the daily issue.
        url_eli: European Legislation Identifier URL, when assigned.
        url_pdf: URL of the official PDF.
    """

    model_config = ConfigDict(frozen=True)

    identifier: str = Field(min_length=1)
    title: str = Field(min_length=1)
    rango: str | None = None
    department: str | None = None
    publication_date: date | None = None
    disposition_date: date | None = None
    official_number: str | None = None
    section_code: str | None = None
    url_eli: str | None = None
    url_pdf: str | None = None


class TextBlock(BaseModel):
    """An ordered paragraph from a document body, tagged with its style class.

    The ``css_class`` (e.g. ``articulo``, ``titulo_num``, ``parrafo``) is what
    the chunker reads to reconstruct the document's legal hierarchy.

    Attributes:
        css_class: Value of the ``class`` attribute on the source ``<p>`` tag.
        text: Normalised text content of the paragraph.
    """

    model_config = ConfigDict(frozen=True)

    css_class: str
    text: str = Field(min_length=1)


class BoeDocument(BaseModel):
    """A fully parsed BOE document: metadata plus its ordered body blocks.

    Attributes:
        metadata: Structured document metadata.
        blocks: Body paragraphs in document order.
    """

    model_config = ConfigDict(frozen=True)

    metadata: DocumentMetadata
    blocks: tuple[TextBlock, ...]


class Chunk(BaseModel):
    """A retrieval unit: a coherent span of one document with its legal context.

    Attributes:
        chunk_id: Deterministic id, ``{document_id}::{ordinal:04d}``.
        document_id: Identifier of the source document.
        document_title: Title of the source document (denormalised for display).
        text: The chunk's text content.
        ordinal: Zero-based position of the chunk within its document.
        titulo: Enclosing TÍTULO heading, when the document has one.
        capitulo: Enclosing CAPÍTULO heading, when present.
        seccion: Enclosing SECCIÓN heading, when present.
        articulo: Enclosing article heading (e.g. ``Artículo 3.``), when present.
        citation: Human-readable citation string for this span.
        url_html: Link back to the official document page.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    document_title: str
    text: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    titulo: str | None = None
    capitulo: str | None = None
    seccion: str | None = None
    articulo: str | None = None
    citation: str
    url_html: str
