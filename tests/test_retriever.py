"""Tests for the dense retriever and eval runner, using a fake embedder.

The fake embedder is deterministic and dependency-free (no torch), so the
ranking and evaluation logic is covered in CI without the `ml` extra.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pytest

from boe_rag.eval.dataset import EvalExample
from boe_rag.eval.retriever import DenseRetriever, FloatMatrix
from boe_rag.eval.runner import run_retrieval_eval

_DIM = 64


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


def _retriever() -> DenseRetriever:
    """Build a retriever indexed over a tiny toy corpus."""
    retriever = DenseRetriever(_FakeEmbedder())
    retriever.index(
        ["c1", "c2", "c3"],
        [
            "precio del gas licuado por canalizacion",
            "consulado honorario en bengasi libia",
            "presupuesto general de navarra importe",
        ],
    )
    return retriever


def test_search_ranks_lexical_match_first() -> None:
    """A query sharing words with a chunk ranks that chunk first."""
    retriever = _retriever()
    results = retriever.search("precio del gas por canalizacion", k=3)
    assert results[0][0] == "c1"
    assert len(results) == 3


def test_search_respects_k() -> None:
    """The number of results is capped at k."""
    retriever = _retriever()
    assert len(retriever.search("gas", k=2)) == 2


def test_search_before_index_raises() -> None:
    """Searching before indexing is an error."""
    with pytest.raises(RuntimeError, match="index"):
        DenseRetriever(_FakeEmbedder()).search("x")


def test_index_length_mismatch_raises() -> None:
    """Mismatched ids/texts lengths are rejected."""
    with pytest.raises(ValueError, match="same length"):
        DenseRetriever(_FakeEmbedder()).index(["a"], ["x", "y"])


def test_run_retrieval_eval_scores_perfectly_on_aligned_set() -> None:
    """An eval set whose questions match chunks yields a perfect hit rate."""
    chunk_ids = ["c1", "c2", "c3"]
    texts = [
        "precio del gas licuado por canalizacion",
        "consulado honorario en bengasi libia",
        "presupuesto general de navarra importe",
    ]
    examples = [
        EvalExample(
            example_id="e1",
            question="precio gas canalizacion",
            relevant_chunk_ids=("c1",),
        ),
        EvalExample(
            example_id="e2",
            question="consulado bengasi libia",
            relevant_chunk_ids=("c2",),
        ),
    ]
    metrics, results = run_retrieval_eval(
        chunk_ids, texts, examples, _FakeEmbedder(), k=3
    )
    assert metrics.hit_rate_at_k == 1.0
    assert all(r.first_relevant_rank == 1 for r in results)
