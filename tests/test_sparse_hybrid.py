"""Tests for the BM25 sparse retriever and the RRF hybrid retriever.

All dependency-free (the fake embedder avoids torch), so the lexical, fusion,
and hybrid logic is covered in CI without the `ml` extra.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pytest

from boe_rag.eval.hybrid import HybridRetriever, reciprocal_rank_fusion
from boe_rag.eval.retriever import DenseRetriever, FloatMatrix
from boe_rag.eval.sparse import BM25Index, tokenize_es

_DIM = 64

_CORPUS_IDS = ["c1", "c2", "c3"]
_CORPUS_TEXTS = [
    "impuesto sobre el valor añadido iva tipo reducido",
    "consulado honorario en bengasi libia",
    "presupuesto general de la comunidad de navarra importe",
]


# --- tokenizer ---------------------------------------------------------------


def test_tokenize_lowercases_and_drops_stopwords() -> None:
    """Tokenisation lowercases and removes stopwords and 1-char tokens."""
    tokens = tokenize_es("El IVA y la Ley de Navarra a")
    assert tokens == ["iva", "ley", "navarra"]


def test_tokenize_preserves_accents_and_enye() -> None:
    """Accented letters and ``ñ`` are kept as part of tokens."""
    assert tokenize_es("añadido pequeño") == ["añadido", "pequeño"]


# --- BM25 --------------------------------------------------------------------


def _bm25() -> BM25Index:
    """Build a BM25 index over the toy corpus."""
    index = BM25Index()
    index.index(_CORPUS_IDS, _CORPUS_TEXTS)
    return index


def test_bm25_ranks_exact_term_match_first() -> None:
    """A query with distinctive terms ranks the matching chunk first."""
    results = _bm25().search("iva valor añadido", k=3)
    assert results[0][0] == "c1"


def test_bm25_excludes_zero_score_chunks() -> None:
    """Chunks matching no query term are not returned."""
    results = _bm25().search("criptomoneda inexistente", k=3)
    assert results == []


def test_bm25_term_frequency_increases_score() -> None:
    """More occurrences of the query term yield a higher score."""
    index = BM25Index()
    index.index(["hi", "lo"], ["gas gas gas precio", "gas energia"])
    results = index.search("gas", k=2)
    assert results[0][0] == "hi"


def test_bm25_respects_k() -> None:
    """The number of results is capped at k."""
    results = _bm25().search("de la comunidad navarra importe consulado", k=1)
    assert len(results) == 1


def test_bm25_search_before_index_raises() -> None:
    """Searching before indexing is an error."""
    with pytest.raises(RuntimeError, match="index"):
        BM25Index().search("x")


def test_bm25_index_length_mismatch_raises() -> None:
    """Mismatched ids/texts lengths are rejected."""
    with pytest.raises(ValueError, match="same length"):
        BM25Index().index(["a"], ["x", "y"])


# --- RRF fusion --------------------------------------------------------------


def test_rrf_rewards_agreement_across_rankings() -> None:
    """An id ranked highly by both lists wins over single-list ids."""
    fused = reciprocal_rank_fusion([["a", "b"], ["a", "c"]])
    assert fused[0][0] == "a"
    ids = [cid for cid, _ in fused]
    assert set(ids) == {"a", "b", "c"}


def test_rrf_higher_rank_beats_lower_rank() -> None:
    """Within a single ranking, an earlier id scores higher."""
    fused = dict(reciprocal_rank_fusion([["first", "second", "third"]]))
    assert fused["first"] > fused["second"] > fused["third"]


# --- hybrid ------------------------------------------------------------------


def _vector(text: str) -> FloatMatrix:
    """Map text to a deterministic bag-of-words unit vector."""
    vec = np.zeros(_DIM, dtype=np.float32)
    for token in text.lower().split():
        digest = hashlib.md5(token.encode()).hexdigest()
        vec[int(digest, 16) % _DIM] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


class _FakeEmbedder:
    """Deterministic embedder that ignores query/passage distinctions."""

    def embed_passages(self, texts: Sequence[str]) -> FloatMatrix:
        """Embed passages as bag-of-words unit vectors."""
        return np.vstack([_vector(t) for t in texts])

    def embed_queries(self, texts: Sequence[str]) -> FloatMatrix:
        """Embed queries as bag-of-words unit vectors."""
        return np.vstack([_vector(t) for t in texts])


def _hybrid() -> HybridRetriever:
    """Build a hybrid retriever over the toy corpus."""
    dense = DenseRetriever(_FakeEmbedder())
    dense.index(_CORPUS_IDS, _CORPUS_TEXTS)
    sparse = BM25Index()
    sparse.index(_CORPUS_IDS, _CORPUS_TEXTS)
    return HybridRetriever(dense, sparse)


def test_hybrid_ranks_relevant_chunk_first() -> None:
    """A query matching one chunk lexically and semantically ranks it first."""
    results = _hybrid().search("iva valor añadido tipo reducido", k=3)
    assert results[0][0] == "c1"


def test_hybrid_respects_k() -> None:
    """The number of fused results is capped at k."""
    results = _hybrid().search("navarra presupuesto importe", k=1)
    assert len(results) == 1


def test_hybrid_recovers_lexical_match_dense_alone_ranks_lower() -> None:
    """Hybrid surfaces a strong lexical hit even when dense ranks it lower."""
    ids = ["doc_lex", "doc_a", "doc_b", "doc_c"]
    texts = [
        "expediente sancionador 12345 abc",
        "alfa beta gamma delta epsilon",
        "alfa beta gamma delta zeta",
        "alfa beta gamma delta eta",
    ]
    dense = DenseRetriever(_FakeEmbedder())
    dense.index(ids, texts)
    sparse = BM25Index()
    sparse.index(ids, texts)
    hybrid = HybridRetriever(dense, sparse)
    # The query term only occurs in doc_lex; the lexical leg pins it to rank 1.
    results = hybrid.search("expediente sancionador 12345", k=4)
    assert results[0][0] == "doc_lex"
