"""Chunking strategies for the chunking ablation.

The production corpus is chunked one-chunk-per-article (structure-aware). This
module lets the eval harness compare that choice against the common alternatives
— fixed-size windows and whole-document chunks — so the design decision is backed
by a measured before/after rather than asserted.

Because relevance in the golden set is defined per *document*, every strategy is
evaluated at document granularity (a retrieved chunk counts as a hit when it
belongs to a relevant document). That keeps the comparison fair across strategies
whose chunk boundaries — and therefore chunk ids — differ.

The alternative strategies re-chunk text reconstructed from the existing
article chunks (concatenated in order), so no re-fetch from the BOE API is needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: Default fixed window size in characters.
DEFAULT_WINDOW_CHARS = 1000
#: Default overlap between consecutive fixed windows, in characters.
DEFAULT_WINDOW_OVERLAP = 150


@dataclass(frozen=True, slots=True)
class CorpusChunk:
    """A row of the article-level corpus needed for re-chunking.

    Attributes:
        chunk_id: The article chunk's stable id.
        document_id: The id of the document the chunk belongs to.
        text: The chunk text.
        ordinal: The chunk's position within its document.
    """

    chunk_id: str
    document_id: str
    text: str
    ordinal: int


@dataclass(frozen=True, slots=True)
class StrategyChunks:
    """Chunks produced by a strategy, as aligned parallel lists.

    Attributes:
        chunk_ids: Stable ids for each produced chunk.
        texts: Chunk texts, aligned with ``chunk_ids``.
        doc_ids: Source document id for each chunk, aligned with ``chunk_ids``.
    """

    chunk_ids: list[str]
    texts: list[str]
    doc_ids: list[str]


def reconstruct_documents(chunks: Sequence[CorpusChunk]) -> dict[str, str]:
    """Rebuild per-document text by joining its article chunks in order.

    Args:
        chunks: The article-level corpus rows.

    Returns:
        A mapping of document id to its reconstructed full text.
    """
    by_doc: dict[str, list[CorpusChunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.document_id, []).append(chunk)
    documents: dict[str, str] = {}
    for doc_id, doc_chunks in by_doc.items():
        ordered = sorted(doc_chunks, key=lambda c: c.ordinal)
        documents[doc_id] = "\n".join(c.text for c in ordered)
    return documents


def article_strategy(chunks: Sequence[CorpusChunk]) -> StrategyChunks:
    """Use the existing article-level chunks unchanged."""
    return StrategyChunks(
        chunk_ids=[c.chunk_id for c in chunks],
        texts=[c.text for c in chunks],
        doc_ids=[c.document_id for c in chunks],
    )


def fixed_windows(
    text: str,
    size: int = DEFAULT_WINDOW_CHARS,
    overlap: int = DEFAULT_WINDOW_OVERLAP,
) -> list[str]:
    """Split text into fixed-size overlapping character windows.

    Args:
        text: The text to split.
        size: Window size in characters.
        overlap: Overlap between consecutive windows, in characters.

    Returns:
        The windows in order; an empty list for blank text.

    Raises:
        ValueError: If ``size <= 0`` or ``overlap >= size``.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")
    if not text.strip():
        return []
    step = size - overlap
    windows: list[str] = []
    start = 0
    while start < len(text):
        windows.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += step
    return windows


def fixed_size_strategy(
    chunks: Sequence[CorpusChunk],
    size: int = DEFAULT_WINDOW_CHARS,
    overlap: int = DEFAULT_WINDOW_OVERLAP,
) -> StrategyChunks:
    """Re-chunk each document into fixed-size overlapping windows."""
    chunk_ids: list[str] = []
    texts: list[str] = []
    doc_ids: list[str] = []
    for doc_id, full_text in reconstruct_documents(chunks).items():
        for index, window in enumerate(fixed_windows(full_text, size, overlap)):
            chunk_ids.append(f"{doc_id}::w{index}")
            texts.append(window)
            doc_ids.append(doc_id)
    return StrategyChunks(chunk_ids=chunk_ids, texts=texts, doc_ids=doc_ids)


def document_strategy(chunks: Sequence[CorpusChunk]) -> StrategyChunks:
    """Use one chunk per whole document (coarse parent-document retrieval)."""
    chunk_ids: list[str] = []
    texts: list[str] = []
    doc_ids: list[str] = []
    for doc_id, full_text in reconstruct_documents(chunks).items():
        chunk_ids.append(f"{doc_id}::doc")
        texts.append(full_text)
        doc_ids.append(doc_id)
    return StrategyChunks(chunk_ids=chunk_ids, texts=texts, doc_ids=doc_ids)
