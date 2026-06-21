"""Tests for the Qdrant-backed dense searcher, using a fake client.

The fake client mimics the slice of ``qdrant_client.QdrantClient`` that
:class:`QdrantSearcher` calls, so the search-result mapping is covered in CI
without the ``qdrant`` extra or a running Qdrant. The query embedder is the same
deterministic, torch-free fake used by the dense-retriever tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pytest

from boe_rag.eval.qdrant_store import CHUNK_ID_FIELD, QdrantSearcher
from boe_rag.eval.retriever import FloatMatrix

_DIM = 8


@dataclass(frozen=True)
class _FakePoint:
    """Stand-in for a Qdrant ``ScoredPoint``."""

    score: float
    payload: Mapping[str, object] | None


@dataclass(frozen=True)
class _FakeResponse:
    """Stand-in for a Qdrant ``query_points`` response."""

    points: Sequence[_FakePoint]


class _FakeClient:
    """Records the last query and returns a canned, ranked response.

    The canned hits are returned in score order regardless of the query vector,
    which is enough to test that :class:`QdrantSearcher` maps payload → chunk id
    and preserves Qdrant's ordering; the ranking itself is Qdrant's concern.
    """

    def __init__(self, points: Sequence[_FakePoint]) -> None:
        """Store the points the fake collection will return."""
        self._points = points
        self.last_query: list[float] | None = None
        self.last_limit: int | None = None

    def query_points(
        self,
        collection_name: str,
        *,
        query: Sequence[float],
        limit: int,
        with_payload: bool,
    ) -> _FakeResponse:
        """Return up to ``limit`` canned points, recording the call."""
        self.last_query = list(query)
        self.last_limit = limit
        return _FakeResponse(points=list(self._points)[:limit])


class _FakeEmbedder:
    """Deterministic embedder that records how many queries it encoded."""

    def __init__(self) -> None:
        """Start with no query-encode calls recorded."""
        self.query_calls = 0

    def embed_passages(self, texts: Sequence[str]) -> FloatMatrix:
        """Unused here, but required by the Embedder protocol."""
        return np.ones((len(texts), _DIM), dtype=np.float32)

    def embed_queries(self, texts: Sequence[str]) -> FloatMatrix:
        """Return a fixed unit vector per query and count the call."""
        self.query_calls += 1
        return np.ones((len(texts), _DIM), dtype=np.float32)


def _points() -> list[_FakePoint]:
    """Three hits in descending score, each carrying its chunk id."""
    return [
        _FakePoint(score=0.91, payload={CHUNK_ID_FIELD: "c1"}),
        _FakePoint(score=0.42, payload={CHUNK_ID_FIELD: "c2"}),
        _FakePoint(score=0.13, payload={CHUNK_ID_FIELD: "c3"}),
    ]


def test_search_maps_payload_to_chunk_id_and_score() -> None:
    """Each hit becomes a (chunk_id, score) pair in Qdrant's order."""
    searcher = QdrantSearcher(_FakeClient(_points()), "boe", _FakeEmbedder())
    assert searcher.search("precio del gas", k=3) == [
        ("c1", 0.91),
        ("c2", 0.42),
        ("c3", 0.13),
    ]


def test_search_passes_k_as_limit_and_encodes_query_once() -> None:
    """K is forwarded as the Qdrant limit and the query is encoded one time."""
    client = _FakeClient(_points())
    embedder = _FakeEmbedder()
    searcher = QdrantSearcher(client, "boe", embedder)

    results = searcher.search("gas", k=2)

    assert len(results) == 2
    assert client.last_limit == 2
    assert client.last_query is not None and len(client.last_query) == _DIM
    assert embedder.query_calls == 1


def test_search_raises_when_payload_missing_chunk_id() -> None:
    """A point without the chunk-id payload signals a mis-built collection."""
    bad = [_FakePoint(score=0.5, payload={"other": "x"})]
    searcher = QdrantSearcher(_FakeClient(bad), "boe", _FakeEmbedder())
    with pytest.raises(ValueError, match=CHUNK_ID_FIELD):
        searcher.search("q", k=1)


def test_search_raises_when_payload_is_none() -> None:
    """A point with no payload at all is likewise rejected."""
    searcher = QdrantSearcher(
        _FakeClient([_FakePoint(score=0.5, payload=None)]), "boe", _FakeEmbedder()
    )
    with pytest.raises(ValueError, match=CHUNK_ID_FIELD):
        searcher.search("q", k=1)
