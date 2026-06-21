"""Qdrant-backed dense retriever — an on-disk alternative to the NumPy index.

Once the corpus grows past a few thousand chunks, holding every embedding in a
process-local NumPy matrix stops being the obvious choice: it cannot be shared
across processes, has to be rebuilt on every boot, and scales only with RAM. A
vector database is the honest production answer. :class:`QdrantSearcher` is a
drop-in for :class:`~boe_rag.eval.retriever.DenseRetriever` — it satisfies the
same :class:`~boe_rag.eval.retriever.Searcher` contract, so the hybrid retriever,
the eval runner, and the serving engine accept either one unchanged.

Only the dense leg moves to Qdrant; BM25 stays in memory and RRF fusion is
untouched (see :mod:`boe_rag.eval.hybrid`). Because the same E5 vectors and the
same cosine metric are used, swapping in Qdrant is a *backend* change, not a
quality change — the parity is proven by re-running the retrieval eval against
the Qdrant leg.

The ``qdrant-client`` dependency is intentionally **not** imported here: the
client is injected behind the small :class:`VectorSearchClient` protocol, so this
module (and its unit tests) stay free of the optional ``qdrant`` extra. The
concrete client is constructed only at the edges — :func:`connect_searcher` and
``scripts/build_qdrant_index.py`` — where the real library is available.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, cast

from boe_rag.eval.retriever import Embedder

#: Payload key under which each point stores its stable corpus chunk id. Qdrant
#: point ids must be unsigned ints or UUIDs, so the BOE chunk id (an arbitrary
#: string) rides in the payload instead and is recovered at search time.
CHUNK_ID_FIELD = "chunk_id"


class ScoredPoint(Protocol):
    """The subset of a Qdrant search hit this module reads."""

    @property
    def score(self) -> float:
        """Similarity score of the hit (cosine, higher is better)."""
        ...

    @property
    def payload(self) -> Mapping[str, object] | None:
        """Stored payload; carries the chunk id under :data:`CHUNK_ID_FIELD`."""
        ...


class QueryResponse(Protocol):
    """The subset of a Qdrant ``query_points`` response this module reads."""

    @property
    def points(self) -> Sequence[ScoredPoint]:
        """The scored hits, best first."""
        ...


class VectorSearchClient(Protocol):
    """Minimal view of ``qdrant_client.QdrantClient`` used for searching.

    Declaring only ``query_points`` keeps this module decoupled from the
    ``qdrant`` extra: any object with a compatible method — including a test
    fake — satisfies it.
    """

    def query_points(
        self,
        collection_name: str,
        *,
        query: Sequence[float],
        limit: int,
        with_payload: bool,
    ) -> QueryResponse:
        """Return the nearest points to ``query`` in ``collection_name``."""
        ...


class QdrantSearcher:
    """Dense retriever that ranks chunks via a Qdrant collection.

    Implements the :class:`~boe_rag.eval.retriever.Searcher` protocol. The query
    is embedded locally with the same :class:`Embedder` used to build the index,
    then nearest-neighbour search is delegated to Qdrant; each hit's stable chunk
    id is read back from its payload.

    Args:
        client: A connected Qdrant client (anything matching
            :class:`VectorSearchClient`).
        collection: Name of the collection to search.
        embedder: Embedder for query encoding; must match the model whose
            vectors populate the collection.
    """

    def __init__(
        self,
        client: VectorSearchClient,
        collection: str,
        embedder: Embedder,
    ) -> None:
        """Bind the client, target collection, and query embedder."""
        self._client = client
        self._collection = collection
        self._embedder = embedder

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return the top-k chunk ids and scores for ``query``.

        Args:
            query: The search query.
            k: Maximum number of results to return.

        Returns:
            Up to ``k`` ``(chunk_id, score)`` pairs, highest score first.

        Raises:
            ValueError: If a returned point is missing its chunk-id payload,
                which would mean the collection was built incorrectly.
        """
        query_vec = [float(x) for x in self._embedder.embed_queries([query])[0]]
        response = self._client.query_points(
            self._collection,
            query=query_vec,
            limit=k,
            with_payload=True,
        )
        results: list[tuple[str, float]] = []
        for point in response.points:
            payload = point.payload or {}
            chunk_id = payload.get(CHUNK_ID_FIELD)
            if chunk_id is None:
                raise ValueError(
                    f"Qdrant point is missing the '{CHUNK_ID_FIELD}' payload; "
                    "rebuild the collection with scripts/build_qdrant_index.py"
                )
            results.append((str(chunk_id), float(point.score)))
        return results


def connect_searcher(
    url: str,
    collection: str,
    embedder: Embedder,
    *,
    api_key: str | None = None,
) -> QdrantSearcher:
    """Connect to a running Qdrant and return a ready :class:`QdrantSearcher`.

    Imports ``qdrant-client`` lazily so the dependency is needed only when a
    Qdrant backend is actually requested (the ``qdrant`` extra).

    Args:
        url: Base URL of the Qdrant instance (e.g. ``http://localhost:6333``).
        collection: Name of the collection to search.
        embedder: Embedder for query encoding.
        api_key: Optional API key for a secured Qdrant deployment.

    Returns:
        A searcher bound to a live client.

    Raises:
        ModuleNotFoundError: If the ``qdrant`` extra is not installed.
    """
    from qdrant_client import QdrantClient

    client = QdrantClient(url=url, api_key=api_key)
    # QdrantClient.query_points has a far broader signature than we use; this
    # searcher only ever calls it with (collection, query=, limit=, with_payload=),
    # which the real client accepts, so narrow it to the protocol we depend on.
    return QdrantSearcher(cast(VectorSearchClient, client), collection, embedder)
