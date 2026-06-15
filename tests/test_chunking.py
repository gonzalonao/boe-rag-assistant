"""Tests for the chunking strategies used in the chunking ablation."""

from __future__ import annotations

import pytest

from boe_rag.eval.chunking import (
    CorpusChunk,
    article_strategy,
    document_strategy,
    fixed_size_strategy,
    fixed_windows,
    reconstruct_documents,
)

_CHUNKS = [
    CorpusChunk(chunk_id="d1-a2", document_id="d1", text="segundo", ordinal=1),
    CorpusChunk(chunk_id="d1-a1", document_id="d1", text="primero", ordinal=0),
    CorpusChunk(chunk_id="d2-a1", document_id="d2", text="otro documento", ordinal=0),
]


def test_reconstruct_documents_orders_by_ordinal() -> None:
    """Document text is rebuilt by joining chunks in ordinal order."""
    docs = reconstruct_documents(_CHUNKS)
    assert docs["d1"] == "primero\nsegundo"
    assert docs["d2"] == "otro documento"


def test_article_strategy_is_passthrough() -> None:
    """The article strategy keeps the original chunks and their doc ids."""
    strategy = article_strategy(_CHUNKS)
    assert strategy.chunk_ids == ["d1-a2", "d1-a1", "d2-a1"]
    assert strategy.doc_ids == ["d1", "d1", "d2"]


def test_document_strategy_one_chunk_per_document() -> None:
    """The document strategy yields exactly one chunk per document."""
    strategy = document_strategy(_CHUNKS)
    assert len(strategy.chunk_ids) == 2
    assert set(strategy.doc_ids) == {"d1", "d2"}
    # Each chunk's text is the full reconstructed document.
    by_doc = dict(zip(strategy.doc_ids, strategy.texts, strict=True))
    assert by_doc["d1"] == "primero\nsegundo"


def test_fixed_windows_overlap_and_cover() -> None:
    """Fixed windows are the right size, overlap, and cover the whole text."""
    text = "abcdefghij"  # 10 chars
    windows = fixed_windows(text, size=4, overlap=1)
    # step = 3 -> windows start at 0, 3, 6; the third already reaches the end.
    assert windows == ["abcd", "defg", "ghij"]
    # Consecutive windows overlap by `overlap` characters.
    assert windows[0][-1] == windows[1][0]


def test_fixed_windows_blank_text_is_empty() -> None:
    """Blank text produces no windows."""
    assert fixed_windows("   ", size=4, overlap=1) == []


def test_fixed_windows_rejects_bad_params() -> None:
    """Non-positive size or too-large overlap is rejected."""
    with pytest.raises(ValueError, match="size"):
        fixed_windows("abc", size=0, overlap=0)
    with pytest.raises(ValueError, match="overlap"):
        fixed_windows("abc", size=4, overlap=4)


def test_fixed_size_strategy_tags_document() -> None:
    """Every fixed window keeps the source document id."""
    strategy = fixed_size_strategy(_CHUNKS, size=4, overlap=1)
    assert set(strategy.doc_ids) <= {"d1", "d2"}
    assert len(strategy.chunk_ids) == len(strategy.texts) == len(strategy.doc_ids)
    # All produced ids are unique.
    assert len(set(strategy.chunk_ids)) == len(strategy.chunk_ids)
