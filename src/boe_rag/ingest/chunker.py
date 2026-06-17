"""Structure-aware chunking of BOE documents.

Rather than slicing text into fixed-size windows, this walks the document's
legal structure (títulos, capítulos, secciones, artículos) and emits one chunk
per article, carrying the full hierarchy as metadata. That yields precise,
human-trustworthy citations ("Ley 39/2015, Artículo 3") and keeps semantically
coherent units together, which measurably improves retrieval quality.

Articles longer than the configured limit are split into numbered parts; stray
fragments shorter than the minimum are merged back into the preceding chunk so
the corpus is not polluted with noise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from boe_rag.config import DOCUMENT_HTML_URL, IngestionConfig
from boe_rag.models import BoeDocument, Chunk, TextBlock

#: Block classes that open or refine the document hierarchy rather than content.
_TITULO_NUM = "titulo_num"
_TITULO_TIT = "titulo_tit"
_CAPITULO_NUM = "capitulo_num"
_CAPITULO_TIT = "capitulo_tit"
_SECCION = "seccion"
_ARTICULO = "articulo"

#: Classes that begin a new heading level (chunk boundary, context reset).
_HIERARCHY_OPENERS: frozenset[str] = frozenset({_TITULO_NUM, _CAPITULO_NUM, _SECCION})

#: Matches the short article label at the start of an article heading.
_ARTICLE_LABEL_RE = re.compile(
    r"^(Art[íi]culo\s+[\w.]+(?:\s+(?:bis|ter|quáter|quinquies))?)",
    re.IGNORECASE,
)


@dataclass
class _Context:
    """Mutable cursor over the document hierarchy while walking blocks."""

    titulo: str | None = None
    capitulo: str | None = None
    seccion: str | None = None
    articulo: str | None = None


@dataclass
class _PendingChunk:
    """A chunk under construction, kept mutable so fragments can be merged."""

    titulo: str | None
    capitulo: str | None
    seccion: str | None
    articulo: str | None
    texts: list[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        """Current combined character length of the buffered text."""
        return sum(len(t) for t in self.texts) + max(0, len(self.texts) - 1)

    def same_scope(self, other: _PendingChunk) -> bool:
        """Whether this chunk shares the hierarchy scope of ``other``."""
        return (
            self.titulo == other.titulo
            and self.capitulo == other.capitulo
            and self.seccion == other.seccion
            and self.articulo == other.articulo
        )


def chunk_document(
    document: BoeDocument,
    config: IngestionConfig | None = None,
) -> list[Chunk]:
    """Split a parsed document into retrieval chunks with legal context.

    Args:
        document: The parsed document to chunk.
        config: Chunking configuration; defaults are used when omitted.

    Returns:
        Chunks in document order, each tagged with its hierarchy and a citation.
    """
    cfg = config or IngestionConfig()
    builder = _ChunkBuilder(cfg)
    for block in document.blocks:
        builder.consume(block)
    builder.flush()
    return builder.finalize(document)


class _ChunkBuilder:
    """Accumulates blocks into pending chunks following the hierarchy rules."""

    def __init__(self, config: IngestionConfig) -> None:
        """Initialise an empty builder bound to ``config``."""
        self._config = config
        self._ctx = _Context()
        self._pending: list[_PendingChunk] = []
        self._current: _PendingChunk | None = None

    def consume(self, block: TextBlock) -> None:
        """Route a single block to the right handler by its style class."""
        css = block.css_class
        if css in _HIERARCHY_OPENERS:
            self._open_hierarchy(css, block.text)
        elif css == _TITULO_TIT:
            self._ctx.titulo = _join_heading(self._ctx.titulo, block.text)
        elif css == _CAPITULO_TIT:
            self._ctx.capitulo = _join_heading(self._ctx.capitulo, block.text)
        elif css == _ARTICULO:
            self._open_article(block.text)
        else:
            self._append_body(block.text)

    def _open_hierarchy(self, css: str, text: str) -> None:
        """Start a new heading level, resetting everything below it."""
        self.flush()
        if css == _TITULO_NUM:
            self._ctx.titulo = text
            self._ctx.capitulo = None
            self._ctx.seccion = None
        elif css == _CAPITULO_NUM:
            self._ctx.capitulo = text
            self._ctx.seccion = None
        else:  # _SECCION
            self._ctx.seccion = text
        self._ctx.articulo = None

    def _open_article(self, text: str) -> None:
        """Start a new article chunk seeded with the article heading text."""
        self.flush()
        self._ctx.articulo = text
        self._current = self._new_pending()
        self._current.texts.append(text)

    def _append_body(self, text: str) -> None:
        """Append a body paragraph, splitting if the chunk grows too large.

        The size cap is a soft bound: an article heading is always kept with at
        least its first paragraph (a chunk only splits once it already holds two
        or more text pieces), so no heading is ever orphaned into its own chunk.
        """
        if self._current is None:
            self._current = self._new_pending()
        prospective = self._current.length + len(text) + 1
        over_limit = prospective > self._config.max_chunk_chars
        if len(self._current.texts) >= 2 and over_limit:
            self.flush()
            self._current = self._new_pending()
        self._current.texts.append(text)

    def _new_pending(self) -> _PendingChunk:
        """Create a pending chunk bound to the current hierarchy context."""
        return _PendingChunk(
            titulo=self._ctx.titulo,
            capitulo=self._ctx.capitulo,
            seccion=self._ctx.seccion,
            articulo=self._ctx.articulo,
        )

    def flush(self) -> None:
        """Finalise the current pending chunk, merging tiny fragments back."""
        if self._current is None or not self._current.texts:
            self._current = None
            return
        chunk = self._current
        self._current = None
        if (
            chunk.length < self._config.min_chunk_chars
            and self._pending
            and self._pending[-1].same_scope(chunk)
        ):
            self._pending[-1].texts.extend(chunk.texts)
            return
        self._pending.append(chunk)

    def finalize(self, document: BoeDocument) -> list[Chunk]:
        """Materialise pending chunks into immutable :class:`Chunk` models."""
        chunks: list[Chunk] = []
        for ordinal, pending in enumerate(self._pending):
            chunks.append(_build_chunk(document, pending, ordinal))
        return chunks


def _join_heading(prefix: str | None, suffix: str) -> str:
    """Combine a numbered heading with its title, e.g. 'TITULO I - ...'."""
    if not prefix:
        return suffix
    return f"{prefix} - {suffix}"


def _article_label(articulo: str | None) -> str | None:
    """Extract a short article label ('Artículo 3') from a full heading."""
    if not articulo:
        return None
    match = _ARTICLE_LABEL_RE.match(articulo)
    if match:
        return match.group(1).rstrip(".")
    return None


def _norm_reference(document: BoeDocument) -> str:
    """Build the norm reference for a citation, e.g. 'Ley 39/2015'."""
    meta = document.metadata
    if meta.rango and meta.official_number:
        return f"{meta.rango} {meta.official_number}"
    if meta.rango:
        return meta.rango
    return meta.identifier


def _build_citation(document: BoeDocument, pending: _PendingChunk) -> str:
    """Compose a human-readable citation for a chunk."""
    reference = _norm_reference(document)
    label = _article_label(pending.articulo)
    if label:
        return f"{reference}, {label}"
    return reference


def _build_chunk(
    document: BoeDocument,
    pending: _PendingChunk,
    ordinal: int,
) -> Chunk:
    """Assemble an immutable :class:`Chunk` from a pending chunk."""
    document_id = document.metadata.identifier
    return Chunk(
        chunk_id=f"{document_id}::{ordinal:04d}",
        document_id=document_id,
        document_title=document.metadata.title,
        text=" ".join(pending.texts),
        ordinal=ordinal,
        titulo=pending.titulo,
        capitulo=pending.capitulo,
        seccion=pending.seccion,
        articulo=pending.articulo,
        citation=_build_citation(document, pending),
        url_html=DOCUMENT_HTML_URL.format(identifier=document_id),
    )
